################################################################################
# Library
################################################################################

library(tidyverse)
library(arules)
library(mclust)
library(rpart)
library(parallel)
library(bnlearn)
library(igraph)
library(stringr)
library(purrr)
library(readxl)
library(tidyr)
library(dplyr)
library(optparse)

################################################################################
# Helper functions #############################################################
################################################################################

create_dir <- function(dir) {
  dir <- dirname(dir)
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
  cat("Directory created: ", dir, "\n")
}

add_s_z_links <- function(node_names){
  # There are z nodes with pattern _z$ or _z[0-9]+$ and s nodes with pattern _s$
  # THe link for s --> z can always be added

  # Create empty list to store expanded constraints
  edge_list <- list()

  # Loop through each node in the node_names vector
  for (from_node in node_names) {
    # check if the node is a s node
    if (grepl("_s$", from_node)) {
      for (to_node in node_names) {
        # check if the node is a z node
        if (grepl("_z$", to_node) || grepl("_z[0-9]+$", to_node)) {
          base_from_name <- gsub("_s$", "", from_node)
          base_to_name <- ifelse(grepl("_z$", to_node), gsub("_z$", "", to_node), gsub("_z[0-9]+$", "", to_node))
          if (base_from_name == base_to_name) {
            edge_list <- c(edge_list, list(data.frame(from = from_node, to = to_node)))
          }
        }
      }
    }
  }

  # Combine all expanded constraints into a single dataframe
  edge_list <- data.table::rbindlist(edge_list)
  return(edge_list)  
}

fix_constraints <- function(constraint, column_names, z_suffixes = NULL) {
  # Suffixes can be _z, _z[0-9]+ and _s
  # Prefixes can be SA_
  # All constraints are not expanded and need to be expanded
  # Create all and then subset for column names

  # Create vectors of prefixes and suffixes
  sa_prefix <- "SA_"
  s_suffix <- "_s"

  # Set default z_suffixes if NULL
  if (is.null(z_suffixes) || z_suffixes == 0) {
    z_suffixes <- c("_z")
  } else {
    z_suffixes <- paste0("_z", seq(0, z_suffixes - 1))
  }

  # Create empty list to store expanded constraints
  expanded_constraints <- list()

  # Function to generate constraint combinations
  generate_constraints <- function(from_col, to_col, prefix, suffix) {
    expanded_constraints <- list(
      data.frame(from = paste0(prefix, from_col), to = paste0(prefix, to_col)),
      data.frame(from = paste0(prefix, to_col), to = paste0(prefix, from_col)),
      data.frame(from = paste0(prefix, from_col), to = paste0(to_col, suffix)),
      data.frame(from = paste0(to_col, suffix), to = paste0(prefix, from_col)),
      data.frame(from = paste0(from_col, suffix), to = paste0(prefix, to_col)),
      data.frame(from = paste0(prefix, to_col), to = paste0(from_col, suffix)),
      data.frame(from = paste0(from_col, suffix), to = paste0(to_col, suffix)),
      data.frame(from = paste0(to_col, suffix), to = paste0(from_col, suffix))
    )
    return(expanded_constraints)
  }

  # Loop through each row in the constraint dataframe
  for (i in seq_len(nrow(constraint))) {
    # Extract from and to columns from the current row
    from_col <- constraint[i, "from"]
    to_col <- constraint[i, "to"]

    # Generate constraint combinations for each z suffix
    for (z_suffix in z_suffixes) {
      expanded_constraints <- c(expanded_constraints, generate_constraints(from_col, to_col, sa_prefix, z_suffix))
    }

    # Generate constraint combinations for the s suffix
    expanded_constraints <- c(expanded_constraints, generate_constraints(from_col, to_col, sa_prefix, s_suffix))
  }

  # Combine all expanded constraints into a single dataframe
  expanded_constraints <- data.table::rbindlist(expanded_constraints)

  # Filter expanded constraints for column names in column_names vector
  expanded_constraints <- expanded_constraints[expanded_constraints$from %in% column_names & expanded_constraints$to %in% column_names, ]

  if (nrow(expanded_constraints) > 0) {
    print("Reading in BL.")
  } else {
    expanded_constraints <- NULL
  }

  return(expanded_constraints)
}

find_modules_stalone_variables <- function(column_names) {
  # divides the data column names into either module names or standalone variable names
  s_idx <- grep("_s$", column_names)
  z_idx <- grep("_z\\d*$", column_names)
  stalone_idx <- grep("^SA_", column_names)

  l <- list(
    s_idx = s_idx,
    z_idx = z_idx,
    stalone_idx = stalone_idx
  )

  return(l)
}

allow_edges <- function(wl, column_names) {
  if (nrow(wl) == 0) {
    return(NULL)
  } else {
    print("Reading in WL.")
    # Create empty list to store expanded constraints
    expanded_whitelist <- list()

    # Loop through each row in the whitelist dataframe and allow edges
    idx <- find_modules_stalone_variables(column_names)

    for (i in seq_len(nrow(wl))) {
      # Extract from and to columns from the current row
      from_col <- wl[i, "from"]
      to_col <- wl[i, "to"]

      # Check if col is stalone variable or module
      ## stalone
      if (any(grepl(paste0("^SA_", from_col, "$"), column_names))) {
        from_type <- "stalone"
      } else {
        from_type <- "module"
      }

      if (any(grepl(paste0("^SA_", to_col, "$"), column_names))) {
        to_type <- "stalone"
      } else {
        to_type <- "module"
      }

      if (from_type == "stalone" && to_type == "stalone") {
        # Create new row with expanded format: SA_{old_name} - SA_{old_name}
        new_row <- data.frame(
          from = paste0("SA_", from_col),
          to = paste0("SA_", to_col)
        )
        expanded_whitelist[[length(expanded_whitelist) + 1]] <- new_row
      } else if (from_type == "stalone" && to_type == "module") {
        # Create new rows with expanded format: SA_{old_name} - {old_name}_z\d*
        z_cols <- grep(paste0("^", to_col, "_z\\d*$"), column_names, value = TRUE)
        new_rows <- data.frame(
          from = paste0("SA_", from_col),
          to = z_cols
        )
        expanded_whitelist <- c(expanded_whitelist, split(new_rows, seq_len(nrow(new_rows))))
      } else if (from_type == "module" && to_type == "stalone") {
        # Create new rows with expanded format: {old_name}_z\d* - SA_{old_name}
        z_cols <- grep(paste0("^", from_col, "_z\\d*$"), column_names, value = TRUE)
        new_rows <- data.frame(
          from = z_cols,
          to = paste0("SA_", to_col)
        )
        expanded_whitelist <- c(expanded_whitelist, split(new_rows, seq_len(nrow(new_rows))))
      } else if (from_type == "module" && to_type == "module") {
        # Create new rows with expanded format: {old_name}_z\d* - {old_name}_z\d*
        from_z_cols <- grep(paste0("^", from_col, "_z\\d*$"), column_names, value = TRUE)
        to_z_cols <- grep(paste0("^", to_col, "_z\\d*$"), column_names, value = TRUE)
        new_rows <- expand.grid(from = from_z_cols, to = to_z_cols)
        expanded_whitelist <- c(expanded_whitelist, split(new_rows, seq_len(nrow(new_rows))))
      }
    }

    # Combine all expanded constraints into a single dataframe
    expanded_whitelist <- data.table::rbindlist(expanded_whitelist)
    # Filter expanded constraint for column names in column_names vector
    expanded_whitelist <- expanded_whitelist[expanded_whitelist$from %in% column_names & expanded_whitelist$to %in% column_names, ]
    return(expanded_whitelist)
  }
}


read_and_concat <- function(files) {
  # Read in all files and concatenate them
  data <- files %>%
    lapply(read.csv) %>%
    reduce(merge, by = "SUBJID")
  return(data)
}

merge_data <- function(input_data, metaenc) {
  # Read standalone data
  data_stalone <- read.csv(input_data)

  if (length(unique(data_stalone$VISIT)) > 1 ){
    stop("Data contains multiple visits. Please provide data with only one visit.")
  }

  data_stalone <- data_stalone %>%
    select(-VISIT)

  if (nrow(data_stalone) > 0) {
    data_stalone <- data_stalone %>%
      mutate_if(is.character, as.factor) %>%
      mutate_if(is.integer, as.numeric)
    data_stalone_flag <- TRUE
  } else {
    data_stalone <- NULL
    data_stalone_flag <- FALSE
  }

  # Read auxiliary data (TODO: fix AUX part)
  aux <- c()
  if (length(aux) > 0) {
    data_aux <- read_and_concat(aux) %>%
      as.data.frame() %>%
      lapply(as.factor) %>%
      as.data.frame()
    data_aux_flag <- TRUE
  } else {
    data_aux <- NULL
    data_aux_flag <- FALSE
  }

  # Read meta data
  data_meta <- read.csv(metaenc)

  # Merge all data
  merge_list <- list(data_meta)
  if (data_stalone_flag) merge_list <- c(merge_list, list(data_stalone))
  if (data_aux_flag) merge_list <- c(merge_list, list(data_aux))

  data <- merge_list %>%
    reduce(merge, by = "SUBJID")

  return(data)
}


create_start_dag <- function(args, start_nodes) {
  # Input path leads to a CSV file with columns "from" and "to"
  if (is.null(args$start_dag)) {
    return(NULL)
  }

  # Read in start DAG input
  start_df <- read.csv(args$start_dag)
  if (nrow(start_df) == 0) {
    return(NULL)
  }

  # Filter for nodes existing in data
  filtered_start_df <- start_df[start_df$from %in% start_nodes & start_df$to %in% start_nodes, ]

  # Get unique nodes from the filtered data frame
  unique_nodes <- unique(c(filtered_start_df$from, filtered_start_df$to))

  # Create an empty DAG with the unique nodes
  start_dag <- bnlearn::empty.graph(unique_nodes)

  # Add arcs to the DAG based on the filtered data frame
  arcs(start_dag) <- filtered_start_df[, c("from", "to")]

  return(start_dag)
}

addnoise <- function(dat, noise) {
  # Add noise to the data that is no factor/string ==> continous
  # when static
  if (noise == 0) {
    return(dat)
  }

  for (i in 1:ncol(dat)) {
    if (is.numeric(dat[[i]])) {
      dat[[i]] <- dat[[i]] + rnorm(nrow(dat), 0, noise)
    }
  }
  return(dat)
}
