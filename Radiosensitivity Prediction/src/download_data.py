import os, subprocess, requests

if not os.path.exists("Radiosensitivity Prediction/data"):
    os.mkdir("Radiosensitivity Prediction/data")

subprocess.call('"C:/Program Files/R/R-4.5.0/bin/x64/Rscript" "Radiosensitivity Prediction/src/download_R_datasets.R', 
                shell=True)

with open("Radiosensitivity Prediction/data/hugo_genes.txt", "wb") as f:
    text = requests.get("https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/locus_types/gene_with_protein_product.txt").content
    f.write(text)
    