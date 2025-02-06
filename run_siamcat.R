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
  make_option(c("-x", "--data"), type = "character", default = "siamcat_results/dataset.csv",
              help = "Input feature data file", metavar = "character"),
  make_option(c("-m", "--meta"), type = "character", default = "siamcat_results/meta.csv",
              help = "Metadata file", metavar = "character"),
  make_option(c("-o", "--output_folder"), type = "character", default = "siamcat_results/",
              help = "Output folder path", metavar = "character"),
  make_option(c("-s", "--split_type"), type = "character", default = "loso",
              help = "kfold/loso/toso", metavar = "character"),
  make_option(c("-d", "--disease"), type = "character", default = "crc",
              help = "type of disease", metavar = "character"),
  make_option(c("-e", "--d_type"), type = "character", default = "rel_abund",
              help = "rel_abund/embed", metavar = "character")
  
)

# Parse options
opt <- parse_args(OptionParser(option_list = option_list))
x_file <- opt$data
meta_file <- opt$meta
output_folder <- opt$output_folder
split_type <- opt$split_type
disease <- opt$disease
d_type <- opt$d_type


# Read data
x <- read.table(x_file, sep = ',', header = TRUE, quote = '', row.names = 1, stringsAsFactors = FALSE, check.names = FALSE)
meta <- read.table(meta_file, sep = ',', header = TRUE, quote = '', row.names = 1, stringsAsFactors = FALSE, check.names = FALSE)

datasets <- unique(meta[, c("Dataset")])


for (d in datasets) {
  
  if (split_type == 'loso') {
    all_pred_matrices <- list()
    for (run in 1:10) {
      meta.test <- meta %>% filter(Dataset == d)
      x.test <- x[, rownames(meta.test)]
      label.test <- create.label(meta = meta.test, label = "Group", control=0)
      meta.train <- meta %>% filter(Dataset != d)
      x.train <- x[, rownames(meta.train)]
      label.train <- create.label(meta = meta.train, label = "Group", case = 1, control = 0)
      sc.obj.train <- siamcat(feat = x.train, label = label.train, meta = meta.train)
      
      if (d_type == 'rel_abund') {
        sc.obj.train <- filter.features(sc.obj.train,
                                        filter.method = 'abundance',
                                        cutoff = 0.001)
        sc.obj.train <- normalize.features(sc.obj.train, norm.method = 'log.std',
                                           norm.param=list(log.n0=1e-06, sd.min.q=0.1),
                                           feature.type = 'original')
        model_type <- 'lasso'
        
      } else {
        sc.obj.train <- filter.features(sc.obj.train,
                                        filter.method = 'variance',
                                        cutoff = .001)
        sc.obj.train <- normalize.features(sc.obj.train, norm.method = 'pass')
        model_type <- 'randomForest'
      }
      # Create data split
      sc.obj.train <- create.data.split(sc.obj.train, num.resample = 1)
      
      if (model_type == 'randomForest'){
        sc.obj.train <- train.model(sc.obj.train, 
                                    method = 'randomForest', 
                                    feature.type = "normalized")
      } else {
        # train LASSO model
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
    
    if (d_type == 'rel_abund') {
      output_file <- paste0(output_folder, split_type, "_", disease, "_", d, ".csv")
    } else {
      output_file <- paste0(output_folder, split_type, "_", disease, "_", d, "_dan", ".csv")
    }
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
    
        
    
    
  } else if (split_type == 'toso') {
    for (d2 in datasets) {
      if (d == d2) {
        next  # Skip the current iteration
      }
      all_pred_matrices <- list()
      for (run in 1:10) {
        meta.test <- meta %>% filter(Dataset == d2)
        x.test <- x[, rownames(meta.test)]
        label.test <- create.label(meta = meta.test, label = "Group", control=0)
        meta.train <- meta %>% filter(Dataset == d)
        x.train <- x[, rownames(meta.train)]
        label.train <- create.label(meta = meta.train, label = "Group", case = 1, control = 0)
        sc.obj.train <- siamcat(feat = x.train, label = label.train, meta = meta.train)
        
        if (d_type == 'rel_abund') {
          sc.obj.train <- filter.features(sc.obj.train,
                                          filter.method = 'abundance',
                                          cutoff = 0.001)
          sc.obj.train <- normalize.features(sc.obj.train, norm.method = 'log.std',
                                             norm.param=list(log.n0=1e-06, sd.min.q=0.1),
                                             feature.type = 'original')
          model_type <- 'lasso'
          
        } else {
          sc.obj.train <- filter.features(sc.obj.train,
                                          filter.method = 'variance',
                                          cutoff = .001)
          sc.obj.train <- normalize.features(sc.obj.train, norm.method = 'pass')
          model_type <- 'randomForest'
        }
        # Create data split
        sc.obj.train <- create.data.split(sc.obj.train, num.resample = 1)
        
        if (model_type == 'randomForest'){
          sc.obj.train <- train.model(sc.obj.train, 
                                      method = 'randomForest', 
                                      feature.type = "normalized")
        } else {
          # train LASSO model
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
      
      if (d_type == 'rel_abund') {
        output_file <- paste0(output_folder, split_type, "_", disease, "_", d,
                              "_train_", d2, "_test.csv")
      } else {
        output_file <- paste0(output_folder, split_type, "_", disease, "_", d,
                              "_train_", d2, "_test_dan", ".csv")
      }
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
    }
    
    
    
    
    
  } else if (split_type == 'kfold') {
    subset_meta <- meta %>% filter(Dataset == d)
    subset_x <- x[, rownames(subset_meta)]
    label <- create.label(meta=subset_meta, label="Group", control=0, case=1)
    sc.obj <- siamcat(feat=subset_x, label=label, meta=subset_meta)
    sc.obj <- filter.features(sc.obj,
                              filter.method = 'abundance',
                              cutoff = 0.001)
    #sc.obj <- normalize.features(sc.obj, norm.method = "log.unit",
    #                            norm.param = list(log.n0 = 1e-06, n.p = 2,norm.margin = 1))
    

    sc.obj <- normalize.features(sc.obj, norm.method = 'log.std',
                                       norm.param=list(log.n0=1e-06, sd.min.q=0.1),
                                       feature.type = 'original')
    
    
    # Split data, train model, and make predictions
    sc.obj <- create.data.split(sc.obj, num.folds = 5, num.resample = 5)
    sc.obj <- train.model(sc.obj, method = "lasso")
    sc.obj <- make.predictions(sc.obj)
    pred_matrix <- pred_matrix(sc.obj)
    output_file <- paste0(output_folder, split_type, "_", disease, "_", d, ".csv")
    
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
    
    
  }
}

