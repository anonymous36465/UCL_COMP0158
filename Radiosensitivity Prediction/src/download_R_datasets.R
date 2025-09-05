##############################################################################
#                                                                            #
# This script is for downloading the Cleveland and CCLE datasets from their  #
# respective R packages RadioGx and PharmacoGx.                              #
#                                                                            #
##############################################################################
options(repos = c(CRAN = "https://cran.r-project.org"))

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("RadioGx")
BiocManager::install("PharmacoGx")
library(RadioGx)
library(PharmacoGx)
library(tidyverse)

# Set this to the path you wish to save the dataset to
# setwd("Radiosensitivity Prediction/data")

# Download Cleveland data from RadioGx
cleveland <- downloadRSet("Cleveland")

# Download tissue and histology info for each sample
phenodata <- sampleInfo(cleveland)
colnames(phenodata) <- c("Tumor_Sample_Barcode", colnames(phenodata)[2:dim(phenodata)[2]])

# Download gene expression data for each sample
col_to_sampleid <- phenoInfo(cleveland, "rnaseq")[, c("sampleid", "rownames")]
row_to_genesymbol <- featureInfo(cleveland, "rnaseq")[, c("Symbol", "rownames")]
cleveland_gene_expr <- molecularProfiles(cleveland, "rnaseq")
colnames(cleveland_gene_expr) <- col_to_sampleid$sampleid
rownames(cleveland_gene_expr) <- row_to_genesymbol$Symbol
cleveland_gene_expr <- t(cleveland_gene_expr)
cleveland_gene_expr <- cbind(rownames(cleveland_gene_expr), cleveland_gene_expr)
colnames(cleveland_gene_expr) <- c("Tumor_Sample_Barcode", colnames(cleveland_gene_expr)[2:dim(cleveland_gene_expr)[2]])


# Download response data
#max_dose <- sapply(strsplit(names(unlist(sensitivityProfiles(cleveland)[,"alpha"])), ".", fixed=T), function(x){as.numeric(gsub("doses", "", x[length(x)]))})
#max_dose[is.na(max_dose)] <- 10
cleveland_response <- cbind(sensitivityInfo(cleveland)$sampleid, sensitivityProfiles(cleveland)[,"AUC_recomputed"],
                            unlist(sensitivityProfiles(cleveland)[,"alpha"]),
                            unlist(sensitivityProfiles(cleveland)[,"beta"]))
#                            max_dose)
colnames(cleveland_response) <- c("id", "auc", "alpha", "beta") #, "max_dose")

common_cells <- intersect(intersect(phenodata$Tumor_Sample_Barcode, cleveland_response[, 1]), cleveland_gene_expr[, 1])
cleveland_response <- cleveland_response[cleveland_response[, 1] %in% common_cells, ]
cleveland_response <- cleveland_response[!duplicated(cleveland_response[,1]),]
cleveland_gene_expr <- cleveland_gene_expr[cleveland_gene_expr[, 1] %in% common_cells, ]
cleveland_gene_expr <- cleveland_gene_expr[!duplicated(cleveland_gene_expr[,1]),]
phenodata <- phenodata[phenodata$Tumor_Sample_Barcode %in% common_cells, ]
phenodata <- phenodata[!duplicated(phenodata$Tumor_Sample_Barcode), ]

# Reorder everything to match
cleveland_gene_expr <- cleveland_gene_expr[match(cleveland_response[, 1], cleveland_gene_expr[, 1]), ]
cleveland_response[, 1] <- phenodata$CellLine
cleveland_gene_expr[, 1] <- phenodata$CellLine
phenodata <- phenodata[, 3:dim(phenodata)[2]]

dir.create("Cleveland", showWarnings = FALSE)
write.csv(phenodata, "Cleveland/cleveland_sampleInfo.csv", row.names=FALSE)
write.csv(cleveland_response, "Cleveland/cleveland_auc_and_model_params.csv", row.names=FALSE)
write.csv(cleveland_gene_expr, "Cleveland/cleveland_gene_expression.csv", row.names=FALSE)
write.csv(cleveland_response[,c(1,2)], "Cleveland/cleveland_auc_only.csv", row.names=FALSE)

# Save normalised auc as well
cleveland_response[, 2] <- as.numeric(cleveland_response[, 2]) / 10
write.csv(cleveland_response, "Cleveland/cleveland_auc_and_model_params_normalised.csv", row.names=FALSE)
write.csv(cleveland_response[,c(1,2)], "Cleveland/cleveland_auc_only_normalised.csv", row.names=FALSE)
