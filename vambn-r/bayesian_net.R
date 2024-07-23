################################################################################
# Dependencies #################################################################
################################################################################

library(tidyverse)
library(bnlearn)
library(igraph)
library(parallel)
library(optparse)
library(ggplot2)
library(ggraph)
library(RColorBrewer)

source("vambn-r/helpers.R") # Load helper functions

################################################################################
# CLI Parsing ##################################################################
################################################################################
options(warn = 1)
option_list <- list(
  make_option("--stalone_data", type = "character", help = "Input file path for imputed standalone data"),
  make_option("--grouping_file", type = "character", help = "Input file path for grouping and types"),
  make_option("--metaenc", type = "character", help = "Input file path for meta-encodings data"),
  make_option("--blacklist", type = "character", help = "Input file path for blacklist data"),
  make_option("--whitelist", type = "character", help = "Input file path for whitelist data", default = NULL),
  make_option("--start_dag", type = "character", help = "Input file path for the starting DAG"),
  make_option("--bnl_maxp", type = "integer", default = 5, help = "Maximum number of parents for each node in BNL (default: 5)"),
  make_option("--n_bootstrap", type = "integer", default = 100, help = "Number of bootstrap samples (default: 10)"),
  make_option("--seed", type = "integer", help = "Random seed for reproducibility", default = 42),
  make_option("--output_dir", type = "character", default = "./", help = "Output directory (default: current directory)"),
  make_option("--bn_out", type = "character", default = "bn.rds", help = "BN object output file name (default: 'bn.rds')"),
  make_option("--likelihood_out", type = "character", default = "likelihood.csv", help = "Likelihood output file name (default: 'likelihood.csv')"),
  make_option("--bn_data", type = "character", help = "Output file path for the BN data"),
  make_option("--cores", type = "integer", default = 1, help = "Number of cores to use")
)

run_cv <- function(data, blacklist, whitelist, max_p, n_bootstrap, cores) {
  cat("Starting cross-validation.\n")

  # Define the assessed algorithms depending on the numeric or discrete nature of the data
  algorithms <- if (all(sapply(data, is.numeric))) {
    c("hc", "tabu", "mmhc", "rsmax2")
  } else {
    c("hc", "tabu", "mmhc", "rsmax2", "h2pc")
  }

  cv_results <- list()
  for (algorithm in algorithms) {
    algorithm.args <- list(
      blacklist = blacklist,
      whitelist = whitelist
    )
    # add max.parents only if the algorithm supports it
    if (algorithm %in% c("hc", "tabu")) {
      algorithm.args$max.parents <- max_p
    }

    loss <- switch(algorithm,
      hc = "bic-cg",
      tabu = "bic-cg",
      mmhc = "bic-cg",
      rsmax2 = "bic-cg",
      h2pc = "bic-cg"
    )

    bn_loss <- NULL

    tryCatch(
      {
        bn_cv <- bnlearn::bn.cv(
          data = data,
          bn = algorithm,
          algorithm.args = algorithm.args,
        )
        cat("Cross-validation for ", algorithm, " completed successfully.\n")
        bn_loss <- mean(loss(bn_cv))
      },
      error = function(e) {
        cat("Error in ", algorithm, ": ", e$message, "\n")
        bn_loss <- NA
      }
    )
    cv_results[[algorithm]] <- bn_loss
  }

  # Evaluate the CV and select the method with the lowest loss
  cv_means <- unlist(cv_results)
  best_method <- names(cv_results)[which.min(cv_means)]
  cat("Cross-validation results:\n")
  print(cv_means)
  cat("Best method: ", best_method, "\n")

  return(best_method)
}

fit_bn <- function(data, best_method, blacklist, whitelist, maxp, cores, bootstrap_samples = 1000, threshold = 0.5) {
  cat("Fitting the final BN.\n")

  # Perform bootstrapping for the final BN
  algorithm_args <- list(
    blacklist = blacklist,
    whitelist = whitelist
  )
  if (best_method %in% c("hc", "tabu")) {
    algorithm_args$maxp <- maxp
  }

  cat("Fitting the final BN using ", best_method, " with ", bootstrap_samples, " bootstrap samples.\n")
  cl <- makeCluster(cores)
  boot_strength <- boot.strength(data,
    algorithm = best_method, R = bootstrap_samples,
    algorithm.args = algorithm_args,
    cluster = cl
  )
  stopCluster(cl)
  cat("Bootstrapping completed.\n")

  # Filter out the edges with a strength below the specified threshold
  boot_strength_filtered <- boot_strength[boot_strength$strength > threshold, ]

  # Generate the averaged network from the filtered bootstrap strength
  averaged_network <- averaged.network(boot_strength_filtered, threshold = threshold)
  extended_network <- cextend(averaged_network)


  # Fit the parameters of the averaged network using MLE
  cat("Fitting the parameters of the final BN using the averaged network.\n")
  final_bn <- bn.fit(extended_network, data)
  cat("Final BN fitted.\n")

  cat("Calculate likelihood of the final BN.\n")
  likelihood <- logLik(final_bn, data, by.sample = TRUE)

  return(list(
    final_bn = final_bn,
    boot_strength = boot_strength,
    averaged_network = averaged_network,
    likelihood = likelihood
  ))
}

main <- function() {
  parser <- OptionParser(option_list = option_list)
  args <- parse_args(parser)
  set.seed(args$seed)
  cat("Parsed arguments:\n")
  print(args)

  # raise error if start_dag is provided
  if (!is.null(args$start_dag)) {
    # stop("start_dag is not supported yet.")
    cat("Warning: start_dag is ignored!\n")
  }

  ################################################################################
  # Load Data ####################################################################
  ################################################################################

  data <- merge_data(args$stalone_data, args$metaenc)

  # Extract z_suffixes from the column names of the merged data
  z_suffixes <- unique(str_extract(colnames(data), "_z\\d*$"))
  z_suffixes <- z_suffixes[!is.na(z_suffixes)]
  cat("Existing z_suffixes:", z_suffixes, "\n")

  # Detect and refactor categorical variables
  categorical_variables <- read.csv(args$grouping_file) %>%
    filter(str_detect(technical_group_name, "stalone")) %>%
    filter(hivae_types == "categorical") %>%
    pull(column_names)

  # Refactor all factor columns (so there are no empty levels)
  for (col in colnames(data)) {
    if (is.factor(data[, col]) || grepl("_s", col) || col %in% categorical_variables) {
      data[, col] <- factor(data[, col])
    }
  }

  # Check for NaN values and drop rows with NaN if present
  if (any(is.na(data))) {
    cat("Data contains NaN values. Dropping rows with NaN values.\n")
    data <- data %>% drop_na()
  }

  # drop columns SUBJID
  subjid <- data$SUBJID
  data <- data %>% select(-SUBJID)

  ################################################################################
  # Bayesian Network #############################################################
  ################################################################################

  # Read blacklist and whitelist constraints
  bl <- if (!is.null(args$blacklist)) {
    bl_input <- ifelse(!is.null(z_suffixes) && length(z_suffixes) > 1, length(z_suffixes), 0)
    read.csv(args$blacklist) %>%
      fix_constraints(column_names = colnames(data), z_suffixes = bl_input)
  } else {
    NULL
  }

  wl <- if (!is.null(args$whitelist)) {
    read.csv(args$whitelist) %>%
      allow_edges(column_names = colnames(data))
  } else {
    NULL
  }

  sz_whitelist <- add_s_z_links(colnames(data))
  if (!is.null(wl)) {
    wl <- rbind(wl, sz_whitelist)
  } else {
    wl <- sz_whitelist
  }

  cat("Saving the data to ", args$bn_data, "\n")
  saveRDS(data, args$bn_data)

  cat("Starting Bayesian Network analysis.\n")
  best_method <- run_cv(
    data = data,
    blacklist = bl,
    whitelist = wl,
    max_p = args$bnl_maxp,
    n_bootstrap = args$n_bootstrap,
    cores = args$cores
  )
  bootstrap_threshold <- 0.3
  results_list <- fit_bn(
    data = data,
    best_method = best_method,
    blacklist = bl,
    whitelist = wl,
    maxp = args$bnl_maxp,
    cores = args$cores,
    bootstrap_samples = args$n_bootstrap,
    threshold = bootstrap_threshold
  )
  final_bn <- results_list$final_bn

  # Save the final BN object
  saveRDS(final_bn, args$bn_out)

  graph <- as.igraph(final_bn)
  boot_strength_df <- results_list$boot_strength
  edge_list <- boot_strength_df[, c("from", "to")]

  # Filter the edge list based on the existing edges in the graph
  filtered_edge_list <- edge_list[paste(edge_list$from, edge_list$to) %in%
    paste(ends(graph, E(graph))[, 1], ends(graph, E(graph))[, 2]), ]

  # Extract the bootstrap strength values for the filtered edges
  edge_labels <- round(boot_strength_df$strength[paste(boot_strength_df$from, boot_strength_df$to) %in%
    paste(filtered_edge_list$from, filtered_edge_list$to)] * 100, 1)

  # Set the Fruchterman-Reingold layout and plot the graph
  # layout <- layout_with_fr(graph)
  layout <- create_layout(graph, layout = "fr")

  # assign colors for the different types of nodes
  # assign red for nodes starting with "SA_"
  available_modules <- c()
  for (node in V(graph)$name) {
    if (grepl("^SA_", node)) {
      available_modules <- c(available_modules, "stalone")
    } else {
      module_name <- strsplit(node, "_")[[1]][1]
      available_modules <- c(available_modules, module_name)
    }
  }
  V(graph)$module <- factor(available_modules)

  # Plot the graph with bootstrap strength as edge weights
  ggraph_plot <- ggraph(layout) +
    geom_edge_link(aes(label = edge_labels),
      arrow = arrow(length = unit(0.3, "cm")),
      end_cap = circle(3, "mm"),
      color = "black"
    ) +
    geom_node_point(aes(color = available_modules), size = 5) +
    geom_node_text(aes(label = name), repel = TRUE, size = 4) +
    labs(
      title = "Bayesian Network",
      subtitle = sprintf("Edges with bootstrap strength > %d", bootstrap_threshold * 100),
      color = "Module"
    ) +
    theme_graph() +
    theme(plot.margin = margin(20, 20, 20, 20), legend.position = "bottom")

  ggsave(paste(args$output_dir, "graph.svg", sep = "/"),
    plot = ggraph_plot, width = 10, height = 7, dpi = 300
  )

  # Save the bootstrapping results
  boot_strength <- results_list$boot_strength
  write.csv(boot_strength, paste(args$output_dir, "bootstrap_strength.csv", sep = "/"))
  g <- ggplot(boot_strength, aes(x = strength)) +
    geom_density(fill = "#69b3a2", color = "#e9ecef", alpha = 0.8)
  ggsave(paste(args$output_dir, "bootstrap_dist.svg", sep = "/"), plot = g)

  # filter the bootstrapped edges based on the threshold
  boot_strength_with_threshold <- boot_strength[boot_strength$strength > 0.3, ]
  write.csv(boot_strength_with_threshold, paste(args$output_dir, "bootstrap_strength_threshold.csv", sep = "/"))

  # likelihood out
  likelihood_df <- data.frame(likelihood = results_list$likelihood, SUBJID = subjid)
  write.csv(likelihood_df, args$likelihood_out)

  cat("Done.\n")
}

main()
