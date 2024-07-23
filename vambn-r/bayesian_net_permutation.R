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

source("vambn-r/helpers.R") # Load helper functions

################################################################################
# CLI Parsing ##################################################################
################################################################################
options(warn = 1)
option_list <- list(
    make_option("--bn_data", type = "character", help = "Input file path for data used for fitting"),
    make_option("--grouping_file", type = "character", help = "Input file path for grouping data"),
    make_option("--blacklist", type = "character", help = "Input file path for blacklist data"),
    make_option("--fitted_bn", type = "character", help = "Input file path for fitted Bayesian network"),
    make_option("--maxp", type = "integer", help = "Maximum number of parents for each node", default = 2),
    make_option("--seed", type = "integer", help = "Random seed for reproducibility", default = 42),
    make_option("--cores", type = "integer", default = 1, help = "Number of cores to use")
)

args <- parse_args(OptionParser(option_list = option_list), args = c(
    "--bn_data", "reportsTesting0429/bn/traditional_adni_wogan_womtl/bn_data.rds",
    "--grouping_file", "data/raw/grouping_adni.csv",
    "--blacklist", "data/raw/blacklist_adni.csv",
    "--fitted_bn", "reportsTesting0429/bn/traditional_adni_wogan_womtl/bn.rds",
    "--maxp", "5",
    "--seed", "42",
    "--cores", "1"
))

################################################################################
# Functions ####################################################################
################################################################################

fit_permuted_graph <- function(bn, blacklist, maxp, data, seed) {
    set.seed(seed)
    node_names <- names(bn)
    random_graph <- random.graph(node_names, method = "melancon", max.in.degree = maxp)
    random_edges <- as.data.frame(random_graph$arcs)

    # cat("Random edges:\n")
    # print(random_edges)
    # filter out edges that are exact matches in the blacklist
    to_remove <- c()
    for (i in 1:nrow(random_edges)) {
        from_edge <- random_edges$from[i]
        to_edge <- random_edges$to[i]
        blacklist_subset <- blacklist %>% filter(from == from_edge & to == to_edge)
        if (nrow(blacklist_subset) > 0) {
            to_remove <- c(to_remove, i)
        }
    }

    random_edges <- random_edges[-to_remove, ]

    # # identify static features that only have one value
    static_features <- colnames(data)[sapply(data, function(x) length(unique(x)) <= 2)]
    random_edges <- random_edges %>%
        filter(!from %in% static_features) %>%
        filter(!to %in% static_features)

    # cat("Static features:\n")
    # print(static_features)

    # cat("Random edges after filtering:\n")
    # print(random_edges)

    # initialize an empty graph with the filtered edges
    random_bn <- empty.graph(node_names)
    arcs(random_bn) <- as.matrix(random_edges)

    # print(random_bn)
    # fit the parameters of the graph
    random_bn <- bn.fit(random_bn, data = data)

    return(random_bn)
}

compare_likelihoods <- function(bn, random_bn, data) {
    # overall
    # bn_likelihood_overall <- logLik(bn, data)
    random_bn_likelihood_overall <- logLik(random_bn, data)

    # by sample
    # bn_likelihood <- logLik(bn, data, by.sample=TRUE)
    random_bn_likelihood <- logLik(random_bn, data, by.sample = TRUE)

    # cat("bn_likelihood_overall:", bn_likelihood_overall, "\n")
    cat("random_bn_likelihood_overall:", random_bn_likelihood_overall, "\n")

    return(
        list(
            # real_likelihood = bn_likelihood_overall,
            random_likelihood = random_bn_likelihood_overall,
            random_likelihood_by_sample = random_bn_likelihood
            # by_sample_difference = bn_likelihood - random_bn_likelihood
        )
    )
}

run_permutation_test <- function(bn, blacklist, maxp, data, n_permutations) {
    random_bns <- lapply(1:n_permutations, function(i) {
        fit_permuted_graph(bn, blacklist, maxp, data, i)
    })
    likelihood_diffs <- lapply(random_bns, function(random_bn) {
        compare_likelihoods(bn, random_bn, data)
    })
    return(likelihood_diffs)
}

calculate_p_value <- function(likelihood_diffs, real_likelihood) {
    # likelihood_diffs is a list of lists
    # for the likelihood we check how often the real likelihood is higher than the permuted likelihood
    # then we calculate the p-value
    n_permutations <- length(likelihood_diffs)
    random_likelihoods <- as.numeric(sapply(likelihood_diffs, function(x) x$random_likelihood))
    count_real_greater <- sum(real_likelihood > random_likelihoods)
    cat("count_real_greater:", count_real_greater, "\n")
    cat("n_permutations:", n_permutations, "\n")

    p_value <- (n_permutations - count_real_greater) / n_permutations
    cat("p_value:", p_value, "\n")
    return(p_value)
}

################################################################################
# Main #########################################################################
################################################################################

bn_data <- readRDS(args$bn_data)
z_suffixes <- unique(str_extract(colnames(bn_data), "_z\\d*$"))
z_suffixes <- z_suffixes[!is.na(z_suffixes)]
cat("Existing z_suffixes:", z_suffixes, "\n")

s_columns <- colnames(bn_data)[grepl("_s$", colnames(bn_data))]
# set to factor
bn_data[s_columns] <- lapply(bn_data[s_columns], as.factor)

data <- addnoise(bn_data, 0.02) # Add noise to the data
bl <- if (!is.null(args$blacklist)) {
    bl_input <- ifelse(!is.null(z_suffixes) && length(z_suffixes) > 1, length(z_suffixes), 0)
    read.csv(args$blacklist) %>%
        fix_constraints(column_names = colnames(bn_data), z_suffixes = bl_input)
} else {
    NULL
}
bn <- readRDS(args$fitted_bn)
bl <- as.data.frame(bl)

real_likelihood <- logLik(bn, bn_data)
real_likelihood_by_sample <- logLik(bn, bn_data, by.sample = TRUE)

cat("Real likelihood overall:", real_likelihood, "\n")

cat("Starting permutation test...\n")
likelihood_diffs <- run_permutation_test(bn, bl, args$maxp, bn_data, 1000)
p_value <- calculate_p_value(likelihood_diffs, real_likelihood)
print(p_value)
