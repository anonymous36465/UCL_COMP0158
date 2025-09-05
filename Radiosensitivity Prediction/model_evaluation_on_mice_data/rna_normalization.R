# BiocManager::install("edgeR")

library(edgeR)
library(limma)

# ########################################################
# # Version 1 ############################################
# pdx_counts <- read.csv('Radiosensitivity Prediction/data/Mice/data_mrna_seq_rpkm.csv', row.names = 1)
# trained_gene_list <- scan("Radiosensitivity Prediction/model_evaluation_on_mice_data/data/rna_imputed_list_of_genes.txt", 
#                           what = character(), sep = "\n")

# # Subset to genes used in training
# genes_to_keep <- intersect(rownames(pdx_counts), trained_gene_list)
# pdx_counts_subset <- pdx_counts[genes_to_keep, ]

# pdx_counts_subset <- pdx_counts

# # Create DGEList object
# dge_pdx <- DGEList(counts = pdx_counts_subset)

# # TMM normalization
# dge_pdx <- calcNormFactors(dge_pdx)

# # Voom transformation
# design <- model.matrix(~1, data = data.frame(sample = colnames(pdx_counts_subset)))
# voom_pdx <- voom(dge_pdx, design = design, plot = FALSE)

# # Extract normalized matrix (log2-CPM)
# X_pdx <- voom_pdx$E  # rows: genes, columns: samples

# X_pdx <- t(X_pdx)
# write.csv(X_pdx, 
#           file = "Radiosensitivity Prediction/model_evaluation_on_mice_data/data/mice_rna_normalized.csv", 
#           row.names = TRUE)

# ##############################################
# # Version 2 ##################################
# # pdx_counts <- read.csv('Radiosensitivity Prediction/data/Mice/data_mrna_seq_rpkm.csv', row.names = 1)
# # trained_gene_list <- scan("Radiosensitivity Prediction/model_evaluation_on_mice_data/data/rna_imputed_list_of_genes.txt", 
# #                           what = character(), sep = "\n")

# pdx_counts <- read.csv('Radiosensitivity Prediction/data/Mice/data_mrna_seq_rpkm.csv',
#                        row.names = 1, check.names = FALSE)

# # # Subset to genes used in training
# # genes_to_keep <- intersect(rownames(pdx_counts), trained_gene_list)
# # pdx_counts_subset <- pdx_counts[genes_to_keep, ]

# pdx_counts_subset <- pdx_counts

# # Create DGEList object
# dge_pdx <- DGEList(counts = pdx_counts_subset)

# # TMM normalization
# dge_pdx <- calcNormFactors(dge_pdx)

# # Voom transformation
# design <- model.matrix(~1, data = data.frame(sample = colnames(pdx_counts_subset)))
# voom_pdx <- voom(dge_pdx, design = design, plot = FALSE)

# # Extract normalized matrix (log2-CPM)
# X_pdx <- voom_pdx$E  # rows: genes, columns: samples

# X_pdx <- t(X_pdx)
# write.csv(X_pdx, 
#           file = "Radiosensitivity Prediction/model_evaluation_on_mice_data/data/mice_rna_normalized_v2.csv", 
#           row.names = TRUE)

########################################
# Version 3 ###########################

# pdx_rpkm <- read.csv('Radiosensitivity Prediction/data/Mice/data_mrna_seq_rpkm.csv',
#                      row.names = 1, check.names = FALSE)

# trained_gene_list <- scan("Radiosensitivity Prediction/model_evaluation_on_mice_data/data/rna_imputed_list_of_genes.txt", 
#                           what = character(), sep = "\n")

# # Subset to genes used in training
# genes_to_keep <- intersect(rownames(pdx_counts), trained_gene_list)
# pdx_counts_subset <- pdx_counts[genes_to_keep, ]

# # Log2-transform RPKM values, adding pseudocount to avoid -Inf
# X_pdx <- log2(pdx_counts_subset + 1e-3)

# # Now samples x genes
# X_pdx <- t(X_pdx)

# write.csv(X_pdx,
#           file = "Radiosensitivity Prediction/model_evaluation_on_mice_data/data/mice_rna_normalized_v4.csv",
#           row.names = TRUE)


# library(data.table)

# # 1) Load
# cbio <- fread("Radiosensitivity Prediction/data/Mice/data_mrna_seq_rpkm.csv",
#               data.table = FALSE, check.names = FALSE)
# rownames(cbio) <- cbio[[1]]; cbio[[1]] <- NULL
# cbio <- as.matrix(cbio)

# orig <- fread("Radiosensitivity Prediction/model_evaluation_on_mice_data/data/transcriptomics_used_by_MOSA.csv",
#               data.table = FALSE, check.names = FALSE)
# rownames(orig) <- orig[[1]]; orig[[1]] <- NULL
# orig <- as.matrix(orig)

# # 2) Keep common genes (and same order)
# genes <- intersect(rownames(orig), rownames(cbio))
# orig  <- orig[genes, , drop = FALSE]
# cbio  <- cbio[genes, , drop = FALSE]

# # 3) Log-transform the cBio RPKM
# cbio_log <- log2(cbio + 0.5)

# # 4) Per-gene mean/SD alignment: map cBio to the "orig" scale
# ref_mu <- rowMeans(orig, na.rm = TRUE)
# ref_sd <- apply(orig, 1, sd, na.rm = TRUE)

# new_mu <- rowMeans(cbio_log, na.rm = TRUE)
# new_sd <- apply(cbio_log, 1, sd, na.rm = TRUE)

# # avoid division by ~0 sd
# scale_factor <- ref_sd / pmax(new_sd, 1e-8)

# cbio_aligned <- sweep(cbio_log, 1, new_mu, "-")
# cbio_aligned <- sweep(cbio_aligned, 1, scale_factor, "*")
# cbio_aligned <- sweep(cbio_aligned, 1, ref_mu, "+")


# # cbio_aligned is now on (approximately) the same per-gene scale as "orig"
# write.csv(cbio_aligned,
#           file = "Radiosensitivity Prediction/model_evaluation_on_mice_data/data/mice_rna_normalized_v5.csv",
#           row.names = TRUE)


library(preprocessCore)   # install.packages("preprocessCore") if needed

# 1) Load
cbio  <- read.csv("Radiosensitivity Prediction/data/Mice/data_mrna_seq_rpkm.csv",
                  row.names = 1, check.names = FALSE)
orig  <- read.csv("Radiosensitivity Prediction/model_evaluation_on_mice_data/data/transcriptomics_used_by_MOSA.csv",
                  row.names = 1, check.names = FALSE)

# 2) Restrict to genes in 'orig' (common set, same order)
genes <- intersect(rownames(orig), rownames(cbio))
orig  <- as.matrix(orig[genes, , drop = FALSE])
cbio  <- as.matrix(cbio[genes, , drop = FALSE])

# 3) Log-transform cBio RPKM
cbio_log <- log2(cbio + 0.5)

# 4) Build target quantile distribution from 'orig'
#    (mean of sorted values across 'orig' samples; length = nGenes)
orig_sorted <- apply(orig, 2, sort)
target <- rowMeans(orig_sorted, na.rm = TRUE)

# 5) Quantile-normalize cBio to the 'orig' target distribution
cbio_qn <- normalize.quantiles.use.target(cbio_log, target = target)
rownames(cbio_qn) <- rownames(cbio_log); colnames(cbio_qn) <- colnames(cbio_log)

# 6) Clip to your range
# cbio_qn[cbio_qn < -3]  <- -3
# cbio_qn[cbio_qn > 17] <- 17

write.csv(cbio_qn,
          file = "Radiosensitivity Prediction/model_evaluation_on_mice_data/data/mice_rna_normalized_v6.csv",
          row.names = TRUE)