############# README
# This is the main analysis file.
# 1. run the R files clean_data->format_data->impute_aux (scripts with fixed settings)
# 2. run HI-VAE jupyter notebook (up until VP decoding)
# 3. run full script below for bayesian network and VirtualPatient validation
# 4. decode generated VPs in the HI-VAE notebook
# 5. Additional analyses:
#     - hi-vae_decoded.R (get all the comparison plots with confidence intervals)
#     - bnet_likelihoods.R (get likelihoods of VP/RP under fitted model)
#     - counterfactuals_bnlearn (interventions - decoded plots in HI-VAE notebook)
#############

############################## Dependencies

rm(list = ls())
library(config)
library(tidyverse)
library(arules)
library(mclust)
library(rpart)
# hc might be overwritten by arules or some such package "bnlearn::hc" if so; not currently used though
library(parallel)
library(bnlearn)
library(stringr)
library(igraph)

############################## Settings and preprocessing

########## Name output files

config <- config::get(file = "/vambn/02_config/config_r.yml")
path_data_in <- config$path_data_in
path_data_out <- config$path_data_out
path_hivae <- config$path_hivae
name <- config$name

# create list visits with column names
visit_names <- strsplit(config$visit_names, split = " ")
visits <- c()
n <- c()
for (i in 1:length(visit_names[[1]])) {
  visits <- c(visits, i)
  n <- c(n, strtoi(visit_names[[1]][i]))
}
names(visits) <- n

path_data_final <- paste0(path_data_out, "data_final.rds")
path_data_out_main <- paste0(path_data_out, name)

blname <- paste0(path_data_in, "main_bl.csv")
wlname <- paste0(path_data_in, "main_wl.csv")
path_data_aux <- paste0(path_data_in, "data_aux.rds")
path_data_all_imp <- paste0(path_data_in, "data_all_imp.rds")

path_final_bn <- paste0(path_data_out_main, "_finalBN.rds")
path_final_bn_fitted <- paste0(path_data_out_main, "_finalBN_fitted.rds")
path_boot_bn <- paste0(path_data_out_main, "_bootBN.rds")
path_virtual_ppts_rds <- paste0(path_data_out_main, "_VirtualPPts.rds")
path_virtual_ppts_csv <- paste0(path_data_out_main, "_VirtualPPts.csv")
path_real_ppts_rds <- paste0(path_data_out_main, "_RealPPts.rds")
path_real_ppts_csv <- paste0(path_data_out_main, "_RealPPts.csv")
path_real <- paste0(path_data_out_main, "_real_df.rds")
path_pt <- paste0(path_data_out_main, "_pt.rds")

path_vp_misslist <- paste0(path_hivae, "VP_misslist/VP_misslist.csv")
path_metaenc <- paste0(path_hivae, "metaenc.csv")

scr <- "bic-cg" # 'bic' for basic autoencoder and fully discretized
mth <- "mle" # 'bayes' for basic autoencoder and fully discretized
bl <- read.csv(blname) # Make bl/wl
wl <- read.csv(wlname) # Make bl/wl

########## Main functions ###############################

########## Load data & remaining formatting of standalone
run_bnet <- function() {
  # could load just basic autoencoded with T (if so need to fully discretize)
  data <- merge_data()
  data <- data[, !grepl("DX|APOE", colnames(data))]

  # refactor all factor columns (so there are no empty levels)
  for (col in colnames(data)) {
    if (is.factor(data[, col]) | grepl("scode", col))
      data[, col] <- factor(data[, col])
  }

  # remove subject variable
  pt <- data$SUBJID
  saveRDS(pt, path_pt)

  data$SUBJID <- NULL

  ######### Discretize & set score
  discdata <- data
  discdata <- addnoise(discdata, 0.01)

  ######### Add AUX superior
  rm <- c()

  for (i in 2:length(visits)) {
    visit_name <- names(visits)[i]
    p <- paste0("_VIS", visit_name)
    value <- factor(ifelse(apply(discdata[, grepl("AUX_", colnames(discdata)) & grepl(p, colnames(discdata))], 1, function(x) (all(x==1))), 1, 0))

    # make string as variable name, then assign value to it
    col_name <- paste0("visitmiss_VIS", visit_name)
    discdata[col_name] <- value

    dfv <- as.data.frame(t(discdata[, (grepl("AUX_", colnames(discdata)) & grepl(p, colnames(discdata))) | grepl(col_name, colnames(discdata))]))

    rmv <- rownames(dfv)[duplicated(dfv, fromLast = TRUE)]
    rm <- c(rm, rmv)
  }

  lowaux <- discdata[, grepl("AUX_", colnames(discdata)) & !(colnames(discdata) %in% rm)]
  lowaux <- colnames(lowaux)[sapply(colnames(lowaux), function(x) sum(as.numeric(as.character(lowaux[,x]))) <= 5)]
  discdata <- discdata[, !(names(discdata) %in% rm)]
  discdata <- discdata[, !(names(discdata) %in% lowaux)]
  orphans <- gsub("AUX_", "", rm)
  orphans <- unname(sapply(orphans, function(x) ifelse(!grepl("SA_", x), paste0("zcode_", x), x)))

  ############################## Bnet

  ######### Final bayesian network
  finalBN <- tabu(discdata, maxp = 5, blacklist = bl, whitelist = wl, score = scr)
  saveRDS(finalBN, path_final_bn)

  ######### Bootstrapped network
  cores <- detectCores()
  cl <-  makeCluster(cores)
  boot.stren = boot.strength(discdata, algorithm = "tabu", R = 1000, algorithm.args = list(maxp = 5, blacklist = bl, whitelist = wl, score = scr), cluster = cl)
  stopCluster(cl)
  boot.strenwithThreshold <- boot.stren[boot.stren$strength > 0.5 & boot.stren$direction >= 0.5, ]
  saveRDS(boot.stren, path_boot_bn)

  # save fitted network
  real <- discdata
  saveRDS(real, path_real)

  #real$SUBJID<-NULL
  finalBN <- readRDS(path_final_bn)
  fitted <- bn.fit(finalBN, real, method = mth)
  saveRDS(fitted, path_final_bn_fitted)

  print("[*] BNET model training script completed.")
}

simulate_virtual_patient <- function() {
  ############################## VP vs RP
  ############################

  print("[*] 1. Simulate virtual patients.")
  real <- readRDS(path_real)
  finalBN <- readRDS(path_final_bn)
  pt <- readRDS(path_pt)

  # Virtual Patient Generation
  virtual <- simulate_VPs(real, finalBN, iterative = FALSE, scr, mth, wl, bl, n = nrow(real))

  ############################
  ############################ save out all data
  ############################

  # save out real and virtual patients
  print("[*] 2. Save virtual patients as rds and csv.")
  real$SUBJID <- pt
  saveRDS(virtual, path_virtual_ppts_rds)
  write.csv(virtual, path_virtual_ppts_csv, row.names = FALSE)
  saveRDS(real, path_real_ppts_rds)
  real$SUBJID <- NULL
  write.csv(real, path_real_ppts_csv, row.names = FALSE)

  # save out VP misslist (for HIVAE decoding, tells HIVAE which zcodes the BN considers missing)
  print("[*] 3. Save vp misslist.")
  save_VPmisslist(virtual, path_data_in, path_hivae)

  ######### Virtual Patient Validation
  #roc<-validate_VP(real=real,virtual=virtual,proc=F) # full AUC rather than partial AUC
  #proc<-validate_VP(real=real,virtual=virtual,proc=T) # partial AUC with focus on sensitivity

  # if validation of ARAE/VAE, saveout is in jupyter notebooks 
  # (VAE_decoded and mclust_decode(R notebook! could make it a script if easier)/ARAE_decoded)

  print("[*] BNET virtual patient simulation script completed.")
}

########## Helper files

# originally from helper/plot_bn.R
# save graph as gml for cytoscape
cyt_graph <- function(g, name, boot) {
  #type <- ifelse(boot, "_bootstrapBN", "_finalBN")

  if (boot) {
    boot_subgraph <- g[g$strength > 0.1 & g$direction > 0.1, ]
    boot_subgraph$strength <- as.character(round(boot_subgraph$strength, 2))
    g1 <- graph_from_data_frame(boot_subgraph)
  } else {
    g1 <- igraph.from.graphNEL(as.graphNEL(finalBN))
  }

  g1 <- set_vertex_attr(g1, "visit", value = as.numeric(gsub("[a-zA-Z0-9_]{1,}_VIS", "", V(g1)$name)))
  V(g1)$name <- gsub("SA_|VIS", "", V(g1)$name)
  #write_graph(g1, paste0(name,type,'.gml'), format = "gml")

  E(g1)$strength <- as.character(E(g1)$strength)
  E(g1)$direction <- as.character(E(g1)$direction)

  #netcont <- createNetworkFromIgraph(g1, name, collection = "Graphs")
  #layoutNetwork("hierarchical")
  #setVisualStyle("default")
  netcont <- createNetworkFromIgraph(g1, paste0(name, "_noaux"), collection = "Graphs")
  layoutNetwork("hierarchical")
  setVisualStyle("default")
}

# originally from helper/plot_bn.R
save_graph <- function(g, name, boot) {
  type <- ifelse(boot, "_bootstrapBN", "_finalBN")

  if (boot) {
    boot_subgraph <- g[g$strength > 0.1 & g$direction > 0.1, ]
    boot_subgraph$strength <- round(boot_subgraph$strength, 4)
    g1 <- graph_from_data_frame(boot_subgraph)
    g2 <- graph_from_data_frame(boot_subgraph[!grepl("AUX_|visitmiss_", boot_subgraph$from), ])
  } else {
    g1 <- igraph.from.graphNEL(as.graphNEL(finalBN))
    g2 <- g1-names(finalBN$nodes)[grepl("AUX_|visitmiss_", names(finalBN$nodes))]
  }

  g1 <- set_vertex_attr(g1, "visit", value = as.numeric(gsub("[a-zA-Z0-9_]{1,}_VIS", "", V(g1)$name)))
  V(g1)$name <- gsub("SA_|VIS", "", V(g1)$name)
  write_graph(g1, paste0(name, type, ".gml"), format = "gml")

  g2 <- set_vertex_attr(g2, "visit", value = as.numeric(gsub("[a-zA-Z0-9_]{1,}_VIS", "", V(g2)$name)))
  V(g2)$name <- gsub("SA_|VIS", "", V(g2)$name)
  write_graph(g2, paste0(name, type, "noaux.gml"), format = "gml")
}

# originally from helper/plot_bn.R
# Plot and save bnet (graphviz)
plot_gr <- function(g, name) {
  G <- as.graphNEL
  graphviz.plot(g)
  pdf(paste0(name, "_graph.pdf"))
  graphviz.plot(g)
  dev.off()
}

# originally from helper/plot_bn.R
# Compare algorithms
compare_algs <- function(data, name) {
  cores <- detectCores()
  cl <- makeCluster(cores)
  cvres1 <- bn.cv(data, "rsmax2", runs=10, fit="bayes", loss="logl",  algorithm.args = list( blacklist=bl, whitelist=wl), cluster=cl) 
  cvres2 <- bn.cv(data, "mmhc", runs=10, fit="bayes", loss="logl",  algorithm.args = list(blacklist=bl,  whitelist=wl), cluster=cl) 
  cvres3 <- bn.cv(data, "hc", runs=10, fit="bayes", loss="logl", algorithm.args = list(maxp=5, blacklist=bl,  whitelist=wl, restart=10, score="bic"), cluster=cl) 
  cvres4 <- bn.cv(data, "tabu", runs=10, fit="bayes", loss="logl", algorithm.args = list(maxp=5, blacklist=bl,  whitelist=wl, restart=10, score="bic"), cluster=cl)
  cvres5 <- bn.cv(data, "si.hiton.pc", runs=10, fit="bayes", loss="logl", algorithm.args = list(blacklist=bl,  whitelist=wl, undirected=FALSE), cluster=cl)
  cvres6 <- bn.cv(data, "mmpc", runs=10, fit="bayes", loss="logl", algorithm.args = list(blacklist=bl, whitelist=wl, undirected=FALSE), cluster=cl)

  plot(cvres1, cvres2, cvres3,cvres4, cvres5 ,cvres6,
        xlab = c("rsmax2", "mmhc","hc","tabu", "si.hiton.pc", "mmpc"))
  pdf(paste(name, "BNcv.pdf", sep=""))
  plot(cvres1, cvres2, cvres3,cvres4, cvres5 ,cvres6,
        xlab = c("rsmax2", "mmhc","hc","tabu", "si.hiton.pc", "mmpc"))
  dev.off()

  stopCluster(cl)
}

# originally from helper/clean_help.R
missingD <- function(dat) {
  # return column i's where more than 50% of data is present
  out <- apply(is.na(dat), 2, mean)
  unlist(which(out < 0.5))
}

# originally from helper/clean_help.R
# return column i's where variance is not 0
includeVar <- function(dat) {
  out <- lapply(dat, function(x) length(unique(x[!is.na(x)])))
  want <- which(out > 1)
  unlist(want)
}

# originally from helper/clean_help.R
# return column i's where missing data <50%
rmMiss <- function(dat) {
  out <- lapply(dat, function(x) mean(is.na(x)))
  want <- which(out < 0.5)
  unlist(want)
}

# originally from helper/clean_help.R
# rescale between 0 and 1
rescale <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# originally from helper/simulate_VP.R
simulate_VPs <- function(real, res, iterative=FALSE, scr, mth, wl, bl, n=NA) {
  n_VP <- ifelse(is.na(n), NROW(real), n)
  print("[*] Number of virtual patients:")
  print(n_VP)

  # estimate the structure and parameters of the Bayesian network.
  #res = tabu(real, maxp=5, blacklist=bl,whitelist=wl,  score=scr) 
  # assuming tabu was the best structure learning approach (currently it seems like for PPMI hc is better!)
  fitted <- bn.fit(res, real, method = mth)
  VP <- c()
  iter <- 1

  # loops until we have a full dataset of VPs (overshoots so data is not always < n_ppts)
  while (NROW(VP) < n_VP) {
    cat("iteration = ", iter, "\n")

    # generate data (until no NAs in any variables)
    generatedDF <- rbn(fitted, n = n_VP)
    comp <- F

    # using mixed data sometimes results in NAs in the generated VPs. These VPs are rejected.
    while (!comp) {
      generatedDF <- generatedDF[complete.cases(generatedDF), ]
      print("[*] Dimensions of generated df:")
      print(dim(generatedDF))
      gen <- n_VP - dim(generatedDF)[1]
      if (gen > 0) {
        generatedDF <- rbind(generatedDF, rbn(fitted, n = gen)) # draw virtual patients
      } else {
        comp <- T
      }
    }

    # VPs are iteratively rejected if they have less than 50% chance to be classified as "real" in a network focussing on correctly classifying real ppts.
    if (iterative) {
      y <- factor(c(rep("original", NROW(real)), rep("generated", n_VP)))
      df <- data.frame(y=y, x=rbind(real, generatedDF))
      fit <- rfsrc(y ~ ., data = df, case.wt = c(rep(1, sum(y == "original")), rep(0.2 * sum(y == "original") / sum(y == "generated"), sum(y == "generated"))))
      print("[*] fit:")
      print(fit)

      DGz <- predict(fit)$predicted[(NROW(df) - NROW(generatedDF) + 1):NROW(df), "original"]
      DGz <- (DGz > 0.5) * 1
      acceptedVPs <- generatedDF[DGz == 1, ]
    } else {
      acceptedVPs <- generatedDF
    }
    VP <- rbind.data.frame(VP, acceptedVPs)
    iter <- iter + 1
    print("[*] NROW(VP):")
    print(NROW(VP))
  }
  VP
}

# originally from helper/VP_misslist.R
VP_misslist <- function(virtual) {
  data <- virtual[, !grepl("AUX_|visitmiss|SA_|scode_", colnames(virtual))]
  missing <- virtual[, grepl("AUX_|visitmiss", colnames(virtual))]

  for (col in colnames(data)) {
    if (paste0("AUX_", sub("zcode_|scode_", "", col)) %in% colnames(missing)) {
      # if exists, replace with inverse of AUX
      data[, col] <- ifelse(missing[, paste0("AUX_", sub("zcode_|scode_", "",col))] == 1,0,1)
    } else {
      # else inverse of visitmiss
      data[, col] <- ifelse(missing[, sub("\\w*_VIS", "visitmiss_VIS", col)] == 1,0,1)
    }
  }
  colnames(data) <- gsub("zcode_|scode_", "", colnames(data))
  write.table(data, path_vp_misslist, sep = ",", row.names = F, col.names = T, quote = F, na = "NaN")
}

# originally from helper/merge_data.R
# merge all data into right directories
merge_data <- function() {

  #(standalone)
  data_all <- readRDS(file = path_data_all_imp)

  # grab all stalone visits columns by keyword indexing
  data_stalone <- data_all[grepl("stalone_VIS", names(data_all))]
  names(data_stalone) <- NULL
  data_stalone <- data_stalone %>% reduce(merge, by = "SUBJID")

  #(aux)
  data_aux <- readRDS(path_data_aux)
  data_aux <- data_aux %>% reduce(merge, by = "SUBJID")
  data_aux <- as.data.frame(lapply(data_aux, factor))

  #(meta)
  data_meta <- read.csv(path_metaenc)

  # merge all
  data <- list(data_meta, data_aux, data_stalone) %>% reduce(merge, by = "SUBJID")

  #flag 0 var cols
  print(colnames(data)[-includeVar(data)])
  data <- data[includeVar(data)]

  saveRDS(data, path_data_final)
  return(data)
}

# originally from addnoise.R
# Add small amount of noise to AUX=1 data (constant otherwise)
addnoise <- function(dat, noise) {
  rm <- c()
  for (col in colnames(dat)) {
    if(!is.factor(dat[, col]) & (any(sapply(colnames(dat),function(x) paste0("AUX_", gsub("zcode_|scode_", "", col)) == x)))) {
      daux <- dat[, sapply(colnames(dat), function(x) paste0("AUX_", gsub("zcode_|scode_", "", col)) == x)]
      if (is.na(sd(dat[daux == 1, col])) | sd(dat[daux == 1, col]) == 0) {
        if (length(dat[daux == 1, col]) == 1) {
          print(paste(col,"has only one missing data point! Removing AUX!"))
          rm <- c(rm, paste0("AUX_", gsub("zcode_|scode_", "", col)))
        } else {
          dat[,col] <- ifelse(daux == 1, dat[daux == 1, col] + rnorm(length(dat[daux == 1, col]), 0, sd(dat[, col], na.rm = T) * noise), dat[, col])
        }
      }
    }
  }
  if (length(rm) == 0) {
    return(dat)
  } else{
    return(dat[, -which(colnames(dat) %in% rm)])
  }
}

# originally from helper/save_VPmisslist.R
# save out VP misslist
save_VPmisslist <- function(virtual, inputfolder, hivaefolder) {
  # for every zcode variable (autoencoder var group) in the data
  for (code in colnames(virtual) [grepl("zcode", colnames(virtual))]) {
    group <- gsub("zcode_", "", code)
    visit <- gsub(paste0("zcode_", "|", gsub("zcode_|_VIS\\d+", "", code)), "", code)
    aux <- paste0("AUX_", group)
    vismiss <- paste0("visitmiss", visit)
    aux_exists <- aux %in% colnames(virtual)
    vismiss_exists <- vismiss %in% colnames(virtual)

    raw <- read.csv(paste0(inputfolder, "data_python/", group, ".csv"), header = F)

    if (aux_exists | vismiss_exists) {
      if (aux_exists) {
        matrix <- matrix(ifelse(virtual[, aux] == 1,0,1), dim(raw)[1], dim(raw)[2])
      } else {
        matrix <- matrix(ifelse(virtual[, vismiss] == 1,0,1), dim(raw)[1], dim(raw)[2])
      }
    } else {
      matrix <- matrix(1, dim(raw)[1], dim(raw)[2])
    }
    # write to file in the hivae folder
    write.table(matrix, paste0(hivaefolder, "VP_misslist/", group, "_vpmiss.csv"), sep=",", row.names = F, col.names = F)
  }
}
