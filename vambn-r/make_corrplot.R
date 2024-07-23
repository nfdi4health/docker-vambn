library(argparse)
library(jsonlite)
library(gridExtra)
library(ggcorrplot)
library(corrplot)
library(patchwork)
library(dplyr)
library(tidyr)
library(Hmisc)

# Command-line argument parsing
parser <- ArgumentParser()

parser$add_argument("grouping", type = "character", help = "Path to the grouping CSV file.")
parser$add_argument("original_data", type = "character", help = "Path to the original data CSV file.")
parser$add_argument("decoded_data", type = "character", help = "Path to the decoded data CSV file.")
parser$add_argument("virtual_data", type = "character", help = "Path to the virtual data CSV file.")
parser$add_argument("all_heatmap_virtual", type = "character", help = "Path to save the heatmap for all virtual data.")
parser$add_argument("all_heatmap_decoded", type = "character", help = "Path to save the heatmap for all decoded data.")
parser$add_argument("cont_heatmap_virtual", type = "character", help = "Path to save the heatmap for continuous virtual data.")
parser$add_argument("cont_heatmap_decoded", type = "character", help = "Path to save the heatmap for continuous decoded data.")
parser$add_argument("result_file", type = "character", help = "Path to save the result metrics JSON file.")
parser$add_argument("dataset_name", type = "character", help = "Name of the dataset.")
parser$add_argument("experiment", type = "character", help = "Name of the experiment.")

args <- parser$parse_args()

# Function to calculate relative correlation error
relative_correlation_error <- function(real, synthetic) {
    real_norm <- norm(real, type = "F")
    diff_norm <- norm((real - synthetic), type = "F")
    return(list(
        error = diff_norm / real_norm,
        real_norm = real_norm,
        syn_norm = norm(synthetic, type = "F")
    ))
}

# Function to prepare data without splitting by visit
prepare_data <- function(data) {
    clean_data <- data %>%
        select(-VISIT) %>%
        filter(!is.na(SUBJID)) %>%
        select(-SUBJID) %>%
        select(sort(names(.)))
    return(clean_data)
}

# Function to generate relative correlation plots
rel_correlation_plot <- function(real_data, synthetic_data, file, method = "spearman", syn_string = "synthetic") {
    real_data <- real_data
    synthetic_data <- synthetic_data[, colnames(real_data)]

    zero_var_cols <- apply(real_data, 2, var) == 0
    zero_var_cols_synthetic <- apply(synthetic_data, 2, var) == 0
    zero_var_cols <- unique(c(which(zero_var_cols), which(zero_var_cols_synthetic)))
    if (length(zero_var_cols) > 0) {
        print("Columns with zero variance:")
        print(colnames(real_data)[zero_var_cols])
        real_data <- real_data[, -zero_var_cols]
        synthetic_data <- synthetic_data[, -zero_var_cols]
    }

    print(paste("Number of NAs in real data:", sum(is.na(real_data))))
    print(paste("Number of NAs in synthetic data:", sum(is.na(synthetic_data))))

    corr_real <- cor(real_data, method = method, use = "complete.obs")
    corr_synthetic <- cor(synthetic_data, method = method, use = "complete.obs")
    metrics <- relative_correlation_error(corr_real, corr_synthetic)

    sorted_corr_real <- corr_real[order(row.names(corr_real), decreasing = FALSE), order(colnames(corr_real), decreasing = FALSE)]
    sorted_corr_synthetic <- corr_synthetic[order(row.names(corr_synthetic), decreasing = FALSE), order(colnames(corr_synthetic), decreasing = FALSE)]

    pdf(file)
    corrplot(corr_real, type = "upper", tl.cex = 0.2, title = sprintf("Original (Norm %.2f)", norm(corr_real, type = "F")), mar = c(0, 0, 3, 0), method = "color")
    corrplot(corr_synthetic, type = "upper", tl.cex = 0.2, title = sprintf("%s (Norm %.2f); Error %.2f", syn_string, norm(corr_synthetic, type = "F"), metrics$error), mar = c(0, 0, 3, 0), method = "color")
    dev.off()

    num_rows <- nrow(corr_real)
    width <- (4000 / 198) * num_rows
    height <- (2000 / 198) * num_rows
    cex <- (0.5 / 198) * num_rows
    title_cex <- (1 / 198) * num_rows
    png_file <- gsub(".pdf", ".png", file)
    print(paste("Width:", width))
    print(paste("Height:", height))
    png(png_file, width = width, height = height, res = 155)
    par(mfrow = c(1, 2))
    corrplot(corr_real, type = "upper", tl.cex = cex, title = sprintf("Original (Norm %.2f)", norm(corr_real, type = "F")), method = "color", cex.main = title_cex)
    corrplot(corr_synthetic, type = "upper", tl.cex = cex, title = sprintf("%s (Norm %.2f); Error %.2f", syn_string, norm(corr_synthetic, type = "F"), metrics$error), method = "color", cex.main = title_cex)
    dev.off()
    print(paste("Saved PNG file:", png_file))

    return(list(
        metrics = metrics
    ))
}

# Read and process data
groups <- read.csv(args$grouping)
subset <- groups[!grepl("stalone_", groups$technical_group_name), ]
continuous_cols <- c(subset[subset$hivae_types %in% c("pos", "real", "truncate_norm", "count", "gamma"), "column_names"], "VISIT", "SUBJID")
module_cols <- c(subset$column_names, "VISIT", "SUBJID")

initial <- read.csv(args$original_data)
virtual <- read.csv(args$virtual_data)
decoded <- read.csv(args$decoded_data)

initial_cols <- colnames(initial)
decoded_cols <- colnames(decoded)
general_overlap <- intersect(initial_cols, decoded_cols)
all_vars_subset <- intersect(general_overlap, module_cols)
continuous_cols <- intersect(all_vars_subset, continuous_cols)

subset_initial <- prepare_data(initial[, all_vars_subset])
subset_virtual <- prepare_data(virtual[, all_vars_subset])
subset_decoded <- prepare_data(decoded[, all_vars_subset])

# Generate Spearman correlation plots for all variables
print("Generate spearman correlation plots for all variables")
print("Start with decoded data")
out_all_d <- rel_correlation_plot(subset_initial, subset_decoded, file = args$all_heatmap_decoded, syn_string = "Decoded")

print("Continue with virtual data")
out_all_v <- rel_correlation_plot(subset_initial, subset_virtual, file = args$all_heatmap_virtual, syn_string = "Virtual")

subset_initial <- prepare_data(initial[, continuous_cols])
subset_virtual <- prepare_data(virtual[, continuous_cols])
subset_decoded <- prepare_data(decoded[, continuous_cols])

# Generate Pearson correlation plots for continuous variables
print("Generate pearson correlation plots for continuous variables")
print("Start with decoded data")
out_cont_d <- rel_correlation_plot(subset_initial, subset_decoded, file = args$cont_heatmap_decoded, method = "pearson", syn_string = "Decoded")

print("Continue with virtual data")
out_cont_v <- rel_correlation_plot(subset_initial, subset_virtual, file = args$cont_heatmap_virtual, method = "pearson", syn_string = "Virtual")

# Save results
numeric_results <- list(
    spearman_relcorr_virtual = out_all_v$metrics$error,
    spearman_relcorr_decoded = out_all_d$metrics$error,
    pearson_relcorr_virtual = out_cont_v$metrics$error,
    pearson_relcorr_decoded = out_cont_d$metrics$error,
    dataset = args$dataset_name,
    experiment = args$experiment
)
write_json(numeric_results, args$result_file)
