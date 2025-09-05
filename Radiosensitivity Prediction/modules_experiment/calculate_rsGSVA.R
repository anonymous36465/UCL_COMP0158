# if (!requireNamespace("BiocManager", quietly = TRUE))
#     install.packages("BiocManager")

# BiocManager::install("GSVA")
# BiocManager::install("GSEABase")  # Needed for gene set formats

#### Specify your dataset name and module range ######

data_name <- "methylation_imputed" #"methylation_imputed"
number_of_modules <- "500" # options: "all", "most-enriched", int

module_grouping_method <- "rsgsva" #options: plage, gsva

library(GSVA)
library(rsGSVA)
source("C:/Users/mmarc/Documents/code/P-Net-Reproducibility-Paper-Fork/rsGSVA/R/rsGSVA.R")

data_dir <- "C:/Users/mmarc/Documents/code/P-Net-Reproducibility-Paper-Fork/Radiosensitivity Prediction/data/Cleveland" # nolint: line_length_linter.
mice_data_dir <- "C:/Users/mmarc/Documents/code/P-Net-Reproducibility-Paper-Fork/Radiosensitivity Prediction/model_evaluation_on_mice_data/data/prepare_mice_data_for_gsva" # nolint: line_length_linter.
experiment_dir <- "C:/Users/mmarc/Documents/code/P-Net-Reproducibility-Paper-Fork/Radiosensitivity Prediction/modules_experiment" # nolint: line_length_linter.

# Load expression matrix
filename <- sprintf("%s.csv", data_name)
mice_filename <- sprintf("mice_%s_%s.csv", data_name, number_of_modules)

### First dataset
expr <- read.csv(file.path(data_dir, filename), row.names = 1)
expr <- as.matrix(expr)
if (ncol(expr) > nrow(expr)) {
  expr <- t(expr)
}

if (data_name == "methylation_imputed") {
  expr <- pmax(pmin(expr, 1), 0)
}

### Second dataset
mice_expr <- read.csv(file.path(mice_data_dir, mice_filename), row.names = 1)
mice_expr <- as.matrix(mice_expr)
if (ncol(mice_expr) > nrow(mice_expr)) {
  mice_expr <- t(mice_expr)
}

# Load gene sets
filename <- sprintf("module_to_genes_%s_%s.csv", data_name, number_of_modules)
df_sets <- read.csv(file.path(experiment_dir, "module_to_genes", filename))
gene_sets <- split(df_sets$gene, df_sets$module)

# Filter the genes used in expr to only be the ones included in gene_sets
genes_in_sets <- unique(df_sets$gene)
expr_filtered <- expr[rownames(expr) %in% genes_in_sets, ]
mice_expr_filtered <- mice_expr[rownames(mice_expr) %in% genes_in_sets, ]

# Run rsGSVA
rs_gsva_result <- rsGSVA(mice_expr_filtered, gene_sets, train_expr = expr_filtered)

# Run GSVA with the param object
gsva_result <- gsva(gsva_par, verbose = TRUE)

# Save result

filename <- sprintf("mice_%s_scores_%s_%s.csv", module_grouping_method, data_name, number_of_modules) # nolint

write.csv(t(gsva_result), file.path(experiment_dir, "gsva_scores", filename))
