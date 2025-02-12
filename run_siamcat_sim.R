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
  make_option(c("-x", "--data"), type = "character", default = "simulation_results/dataset.csv",
              help = "Input feature data file", metavar = "character"),
  make_option(c("-m", "--meta"), type = "character", default = "simulation_results/siamcat_meta.csv",
              help = "Metadata file", metavar = "character"),
  make_option(c("-o", "--output_folder"), type = "character", default = "simulation_results/",
              help = "Output folder path", metavar = "character"),
  make_option(c("-d", "--disease"), type = "character", default = "crc",
              help = "type of disease", metavar = "character"),
  make_option(c("-a", "--alpha"), type = "character", default = "",
              help = "alpha", metavar = "character"),
  make_option(c("-p", "--prop_diff"), type = "character", default = "",
              help = "prop_diff", metavar = "character"),
  make_option(c("-e", "--effect_size"), type = "character", default = "",
              help = "effect_size", metavar = "character"),
  make_option(c("-f", "--train_study"), type = "character", default = "",
              help = "train_study", metavar = "character"),
  make_option(c("-g", "--test_study"), type = "character", default = "",
              help = "test_study", metavar = "character")
)

# Parse options
opt <- parse_args(OptionParser(option_list = option_list))
x_file <- opt$data
meta_file <- opt$meta
output_folder <- opt$output_folder
split_type <- 'toso'
train_dataset_name <- 'train'
test_dataset_name <- 'test'
disease <- opt$disease
alpha <- opt$alpha
prop_diff <- opt$prop_diff
effect_size <- opt$effect_size
train_study <- opt$train_study
test_study <- opt$test_study



# Read data
x <- read.table(x_file, sep = ',', header = TRUE, quote = '', row.names = 1, stringsAsFactors = FALSE, check.names = FALSE)
meta <- read.table(meta_file, sep = ',', header = TRUE, quote = '', row.names = 1, stringsAsFactors = FALSE, check.names = FALSE)

all_pred_matrices <- list()
for (run in 1:10) {
  meta.test <- meta %>% filter(Dataset == test_dataset_name)
  x.test <- x[, rownames(meta.test)]
  label.test <- create.label(meta = meta.test, label = "Group", case = 1, control=0)
  meta.train <- meta %>% filter(Dataset == train_dataset_name)
  x.train <- x[, rownames(meta.train)]
  label.train <- create.label(meta = meta.train, label = "Group", case = 1, control = 0)
  sc.obj.train <- siamcat(feat = x.train, label = label.train, meta = meta.train)
  
  sc.obj.train <- filter.features(sc.obj.train,
                                filter.method = 'abundance',
                                cutoff = 1e-6)
  sc.obj.train <- normalize.features(sc.obj.train, norm.method = 'log.std',
                                     norm.param=list(log.n0=1e-06, sd.min.q=0.1),
                                     feature.type = 'original')
  model_type <- 'lasso'
      
    # Create data split
  sc.obj.train <- create.data.split(sc.obj.train, num.resample = 1)
  
  min_nonzero <- 5  # Starting value for min.nonzero
  success <- FALSE  # Flag to track if the function runs successfully
  while (!success && min_nonzero > 0) {
    tryCatch({
      # Attempt to run the function
      sc.obj.train <- train.model(sc.obj.train, 
                                  method = 'lasso', 
                                  feature.type = "normalized", 
                                  min.nonzero = min_nonzero)
      success <- TRUE  # If no error occurs, mark success as TRUE
    }, error = function(e) {
      # If an error occurs, decrement min.nonzero
      message(sprintf("Failed with min.nonzero = %d. Retrying with %d...", min_nonzero, min_nonzero - 1))
      min_nonzero <<- min_nonzero - 1
    })
  }
    
  if (success) {
    message(sprintf("Model trained successfully with min.nonzero = %d", min_nonzero))
  } else {
    message("Failed to train model even with min.nonzero = 0")
  }
  sc.obj.train <- make.predictions(sc.obj.train)
  sc.obj.train <- evaluate.predictions(sc.obj.train)
  
  sc.obj.test <- siamcat(feat = x.test, label = label.test, meta = meta.test)
  # make holdout predictions
  sc.obj.test <- make.predictions(sc.obj.train, 
                                  siamcat.holdout = sc.obj.test,
                                  normalize.holdout = TRUE)
  sc.obj.test <- evaluate.predictions(sc.obj.test)
  pred_matrix <- pred_matrix(sc.obj.test)
  
  pred_matrix <- as.data.frame(sapply(seq(1, ncol(pred_matrix), by = 2), function(i) {
    rowMeans(pred_matrix[, i:(i + 2 - 1)], na.rm = TRUE)
  }))
  all_pred_matrices[[run]] <- pred_matrix
}
final_pred_matrix <- do.call(cbind, all_pred_matrices)

output_file <- paste0(output_folder, "SIAMCAT_", disease, "_alpha", alpha,
                      "_proportion_diff_", prop_diff, "_ed_",
                      effect_size, "_train_", train_study, "_test_", 
                      test_study, ".csv")

file_conn <- file(output_file, open = "w")
for (run in seq(ncol(final_pred_matrix))) {
  cat(sprintf("run\t%d\n", run - 1), file = file_conn, append = TRUE)
  # Extract true labels for test samples
  test_true_labels <- label.test$label
  test_true_labels <- unname(test_true_labels)
  cat("true labels\t", paste(test_true_labels, collapse = "\t"), "\n", file = file_conn, append = TRUE)
  
  # Extract estimated labels for test samples (assuming threshold 0.5 for classification)
  test_estimated_probabilities <- as.vector(final_pred_matrix[, run, drop = FALSE])
  test_estimated_probabilities <- unlist(test_estimated_probabilities)
  test_estimated_labels <- ifelse(test_estimated_probabilities > 0.5, 1, 0)
  cat("estimated labels\t", paste(test_estimated_labels, collapse = "\t"), "\n", file = file_conn, append = TRUE)
  
  # Extract estimated probabilities for test samples    
  cat("estimated probabilities\t", paste(round(test_estimated_probabilities, 3), collapse = "\t"), "\n", file = file_conn, append = TRUE)
  cat("sample index\t", paste(row.names(final_pred_matrix), collapse = "\t"), "\n", file = file_conn, append = TRUE)
  
}
# Close file connection
close(file_conn)
