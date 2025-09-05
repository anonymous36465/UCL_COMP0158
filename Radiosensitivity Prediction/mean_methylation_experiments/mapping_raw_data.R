# Step 1: Load beta matrix (adjust path if needed)
beta_data <- read.delim("Radiosensitivity Prediction/mean_methylation_experiments/GSE240704_Matrix_normalized.txt.gz")

# Step 2: Load required packages and annotation
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

# Install the annotation package if not already
if (!require("IlluminaHumanMethylationEPICanno.ilm10b4.hg19", character.only = TRUE)) {
    BiocManager::install("IlluminaHumanMethylationEPICanno.ilm10b4.hg19")
}

library(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)
library(minfi)

# Step 3: Load annotation
anno <- getAnnotation(IlluminaHumanMethylationEPICanno.ilm10b4.hg19)

# Step 4: Merge beta matrix with annotation
merged <- merge(beta_data, anno, by.x = "ID_REF", by.y = "Name")

# Step 5: Count total number of unique genes
all_genes <- unique(unlist(strsplit(merged$UCSC_RefGene_Name, ";")))
cat("Total unique genes in data:", length(all_genes), "\n")

# Step 6: Filter to promoter probes only (TSS200 or TSS1500)
promoter_probes <- merged[grepl("TSS200|TSS1500", merged$UCSC_RefGene_Group), ]
promoter_genes <- unique(unlist(strsplit(promoter_probes$UCSC_RefGene_Name, ";")))
cat("Unique promoter genes (TSS200 or TSS1500):", length(promoter_genes), "\n")

writeLines(sort(unique(all_genes)), "all_genes_from_beta_data.txt")
