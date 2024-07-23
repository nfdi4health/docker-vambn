############################## Dependencies

library(tidyverse)
library(arules)
library(mclust)
library(rpart)
library(parallel)
library(bnlearn)
library(stringr)
library(igraph)
library(purrr)
library(tidyverse)
library(readxl)
library(tidyr)
library(dplyr)
library(optparse)

############################## Get the CLI arguments

option_list <- list(
  make_option("--fitted_model", type="character", help="Fitted model"),
  make_option("--n_patients", type="integer", default=1000, help="Number of patients to generate"),
  make_option("--encoding", type="character", help="Output file path for data out"),
  make_option("--likelihood_out", type="character", help="Output file path for likelihood"),
  make_option("--seed", type="integer", help="Random seed for reproducibility", default=42)
)


parser <- OptionParser(option_list=option_list)
args <- parse_args(parser)
set.seed(args$seed)
##############################

cat("Loading model...\n")
mod <- readRDS(args$fitted_model)

cat("Generating synthetic data...\n")
synthetic <- rbn(mod, n=args$n_patients)

cat("Calculating likelihood of virtual data.\n")
virtual_likelihood <- logLik(mod, data=synthetic, by.sample=TRUE)
likelihood_df <- data.frame(likelihood=virtual_likelihood, subjid=1:args$n_patients)

write.csv(synthetic, args$encoding)
write.csv(likelihood_df, args$likelihood_out)
