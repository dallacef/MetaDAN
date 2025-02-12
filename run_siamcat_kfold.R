#!/usr/bin/env Rscript
options(repos = c(CRAN = "https://cloud.r-project.org"))
# Load necessary libraries
if (!requireNamespace("optparse", quietly = TRUE)) install.packages("optparse")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("SIAMCAT", quietly = TRUE)) install.packages("SIAMCAT")
if (!requireNamespace("ranger", quietly = TRUE)) install.packages("ranger")

library(optparse)
library(dplyr)
library(SIAMCAT)
library(ranger)

# Define command-line options
option_list <- list(
  make_option(c("-x", "--data"), type = "character", default = "SIAMCAT_results/dataset.csv",
              help = "Input feature data file", metavar = "character"),
  make_option(c("-m", "--meta"), type = "character", default = "SIAMCAT_results/meta.csv",
              help = "Metadata file", metavar = "character"),
  make_option(c("-o", "--output_folder"), type = "character", default = "SIAMCAT_results/",
              help = "Output folder path", metavar = "character"),
  make_option(c("-n", "--dataset_name"), type = "character", default = "",
              help = "Dataset name for kfold", metavar = "character"),
  make_option(c("-d", "--disease"), type = "character", default = "crc",
              help = "type of disease", metavar = "character")
)

# Parse options
opt <- parse_args(OptionParser(option_list = option_list))
x_file <- opt$data
meta_file <- opt$meta
output_folder <- opt$output_folder
dataset_name <- opt$dataset_name
disease <- opt$disease
split_type <- 'kfold'



# Read data
x <- read.table(x_file, sep = ',', header = TRUE, quote = '', row.names = 1, stringsAsFactors = FALSE, check.names = FALSE)
meta <- read.table(meta_file, sep = ',', header = TRUE, quote = '', row.names = 1, stringsAsFactors = FALSE, check.names = FALSE)

label <- create.label(meta=meta, label="Group", control=0, case=1)
sc.obj <- siamcat(feat=x, label=label, meta=meta)
sc.obj <- filter.features(sc.obj,
                          filter.method = 'abundance',
                          cutoff = 1e-6)

sc.obj <- normalize.features(sc.obj, norm.method = 'log.std',
                                   norm.param=list(log.n0=1e-06, sd.min.q=0.1),
                                   feature.type = 'original')


# Split data, train model, and make predictions
sc.obj <- create.data.split(sc.obj, num.folds = 5, num.resample = 5)
sc.obj <- train.model(sc.obj, method = "lasso")
sc.obj <- make.predictions(sc.obj)
pred_matrix <- pred_matrix(sc.obj)
output_file <- paste0(output_folder, split_type, "_", disease, "_", dataset_name, ".csv")

file_conn <- file(output_file, open = "w")
for (run in seq_along(sc.obj@data_split$test.folds)) {
  for (fold in seq_along(sc.obj@data_split$test.folds[[run]])){
    cat(sprintf("run/fold\t%d/%d\n", run-1, fold-1), file = file_conn, append = TRUE)
    # Extract true labels for test samples
    test_true_labels <- label$label[sc.obj@data_split$test.folds[[run]][[fold]]]
    test_true_labels <- unname(test_true_labels)
    cat("true labels\t", paste(test_true_labels, collapse = "\t"), "\n", file = file_conn, append = TRUE)
    
    # Extract estimated labels for test samples (assuming threshold 0.5 for classification)
    test_estimated_probabilities <- as.vector(pred_matrix[sc.obj@data_split$test.folds[[run]][[fold]], paste0("CV_rep", run), drop = FALSE])
    test_estimated_labels <- ifelse(test_estimated_probabilities > 0.5, 1, 0)
    cat("estimated labels\t", paste(test_estimated_labels, collapse = "\t"), "\n", file = file_conn, append = TRUE)
    
    # Extract estimated probabilities for test samples    
    cat("estimated probabilities\t", paste(round(test_estimated_probabilities, 3), collapse = "\t"), "\n", file = file_conn, append = TRUE)
    cat("sample index\t", paste(sc.obj@data_split[["test.folds"]][[run]][[fold]], collapse = "\t"), "\n", file = file_conn, append = TRUE)

  }
}
# Close file connection
close(file_conn)

