import pandas as pd
import requests
import os
import gzip
import subprocess

wd = "Radiosensitivity Prediction/data"

# Download list of coding genes
temp = requests.get("https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/locus_types/gene_with_protein_product.txt")
with open(f"{wd}/hugo_genes.txt", "wb") as f:
    f.write(temp.content)

# Download radiosensitivity data from RadioGx R package
subprocess.run('"C:\\Program Files\\R\\R-4.5.0\\bin\\Rscript" download_R_datasets.R', shell=True)

# (Data can be downloaded manually from https://depmap.org/portal/data_page/?tab=allData )
# Download methylation data for CCLE to use with gene expression data from RadioGx
# temp = requests.get("https://storage.googleapis.com/depmap-external-downloads/ccle/ccle_2019/CCLE_RRBS_TSS1kb_20181022.txt.gz?GoogleAccessId=depmap-external-downloads%40broad-achilles.iam.gserviceaccount.com&Expires=1742467134&Signature=hDW7ixGAiGBz%252FoSJoATcDK%252BnHp5NjsSO4%252FFowKpDSzyzbfttkgG9c6xMVSrcg%252FSPdb5%252FHDv8Gck6quq%252BpncS8ipan0A%252BPuPFcD4UvKRtodFHXYm0%252FdXZkWkB34Ug5gJVWkrUREI2gOqHyjHE9SAiVkD2rfTWJ02KoNrPOlpcrObp32X%252Bv4zRSqwUvzW2oGXpY6akmr%252BWhEDGIPoxS2YTgudjeJrdOFHr%252BdI1gIW%252B4oXpn%252FuygIq9Fq%252F%252B9AEXZF39etB6KkJm1sAgbEAHEPE1CLwZgUgiARa13X%252ByCA4c37uOs1UJwcl9Zjt1uLHF0NsVPT%252BKX8X2HjgkbPeYBG%252Bp%252Bg%3D%3D&userProject=broad-achilles")
# with open(f"{wd}/Cleveland/CCLE_RRBS_TSS1kb_20181022.txt.gz", "wb") as f:
#     f.write(temp.content)
# with gzip.open(f"{wd}/Cleveland/CCLE_RRBS_TSS1kb_20181022.txt.gz", "rb") as f_in:
#     with open(f"{wd}/Cleveland/CCLE_RRBS_TSS1kb_20181022.txt", "wb") as f_out:
#         f_out.write(f_in.read())

df = pd.read_csv(f"{wd}/Cleveland/CCLE_RRBS_TSS1kb_20181022.txt", sep="\t").dropna()
df["locus_id"] = df["locus_id"].apply(lambda x : x.split("_")[0])
df = df.drop(df.columns[1:3], axis=1)
df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda x : pd.to_numeric(x, errors="coerce")).fillna(0.0)
df = df.groupby("locus_id").mean().T
df.index = [x.split("_")[0] for x in df.index]
df = df.loc[~df.index.duplicated(keep="first")]
df.to_csv(f"{wd}/Cleveland/CCLE_Methylation_TSS1kb_20181022.csv", index=True)
# os.remove(f"{wd}/Cleveland/CCLE_RRBS_TSS1kb_20181022.txt.gz")
# os.remove(f"{wd}/Cleveland/CCLE_RRBS_TSS1kb_20181022.txt")

df = pd.read_csv(f"{wd}/Cleveland/cleveland_sampleinfo.csv")
unique_primary_sites = df["Primarysite"].unique()
primary_sites = {site: (i+1) for i, site in enumerate(unique_primary_sites)}

unique_histology = df["Histology"].unique()
histology = {site: (i+1) for i, site in enumerate(unique_histology)}

unique_subhistology = df["Subhistology"].unique()
subhistology = {site: (i+1) for i, site in enumerate(unique_subhistology)}

df["Primarysite"] = df["Primarysite"].map(primary_sites)
df["Histology"] = df["Histology"].map(histology)
df["Subhistology"] = df["Subhistology"].map(subhistology)

df.set_index("CellLine", inplace=True)
print(df.head(30))

df.to_csv(f"{wd}/Cleveland/cleveland_sampleinfo_numerical.csv")