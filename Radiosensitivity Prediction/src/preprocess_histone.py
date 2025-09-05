import pandas as pd
import requests
import os
import gzip
import subprocess
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression


wd = Path("Radiosensitivity Prediction/data/Cleveland")

histone_data_path = wd / "CCLE_GlobalChromatinProfiling_20181130.csv"
histone_data_fillna =  wd / "histone_modification_data_fillna.csv"
histone_data_dropna =  wd / "histone_modification_data_dropna.csv"

# if not histone_data_path.exists():
#     url = "https://depmap.org/portal/data_page/?tab=allData&releasename=CCLE%202019&filename=CCLE_GlobalChromatinProfiling_20181130.csv"
#     response = requests.get(url)
#     with open(histone_data_path, "wb") as f:
#         f.write(response.content)
#     print("Download complete!")
# else:
#     df = pd.read_csv(histone_data_path)

# Two ways of handling NaN values
# df_drop = pd.read_csv(histone_data_path, sep=",", low_memory=False).dropna()

# df_fill = pd.read_csv(histone_data_path)
# df_fill = df_fill.fillna(0)

# for file_path, df in [(histone_data_dropna, df_drop), (histone_data_fillna, df_fill)]:

#     df["CellLineName"] = df["CellLineName"].apply(lambda x : x.split("_")[0])

#     # drop the 'BroadID' column
#     df = df.drop(df.columns[1], axis=1)

#     # keep only the cell lines that we have the sensitivity data for
#     unique_count = df.iloc[:, 0].nunique()
#     print(f"{unique_count} unique cell lines in histone data")
#     cleveland_cell_lines = list(set(pd.read_csv(f"{wd}/cleveland_auc_only.csv")['id']))
#     df = df[df.iloc[:, 0].isin(cleveland_cell_lines)]
#     print(f"{df.iloc[:, 0].nunique()} unique cell lines, after keeping only lines in Cleveland AUC set.")
#     # cell_lines_to_keep = df.iloc[:, 0]

#     df.to_csv(file_path, index=False, header=True)


# # """ combined gene expression and hisotne dataset"""

# df = pd.read_csv( wd / 'rna_imputed.csv')

# rna_reduced_df = df.set_index('id')

# # combine with histone dataset
# histone = pd.read_csv( wd / 'histone_modification_data_process_na.csv')
# rna = rna_reduced_df.reset_index()  # ensures 'Tumor_Sample_Barcode' becomes a column
# rna = rna.rename(columns={'id': 'CellLineName'})

# rna = rna.add_prefix('gexpr_') # Add prefixes to column names (except for key)
# histone = histone.add_prefix('histone_')

# rna = rna.rename(columns={'gexpr_CellLineName': 'CellLineName'})
# histone = histone.rename(columns={'histone_CellLineName': 'CellLineName'})

# merged_df = pd.merge(rna, histone, on='CellLineName', how='inner')
# file_path = wd / 'rna_imputed_and_histone_processed_na.csv'
# merged_df.to_csv(file_path, index=False, header=True)


# """combined methylation (imputed) and hisotne data"
df = pd.read_csv( wd / 'methylation_imputed.csv')

rna_reduced_df = df.set_index('model_id')

# combine with histone dataset
histone = pd.read_csv( wd / 'histone_modification_data_process_na.csv')
rna = rna_reduced_df.reset_index()  # ensures 'Tumor_Sample_Barcode' becomes a column
rna = rna.rename(columns={'model_id': 'CellLineName'})

rna = rna.add_prefix('meth_') # Add prefixes to column names (except for key)
histone = histone.add_prefix('histone_')

rna = rna.rename(columns={'meth_CellLineName': 'CellLineName'})
histone = histone.rename(columns={'histone_CellLineName': 'CellLineName'})

merged_df = pd.merge(rna, histone, on='CellLineName', how='inner')
file_path = wd / 'meth_imputed_and_histone_processed_na.csv'
merged_df.to_csv(file_path, index=False, header=True)


# # Also use unsupervized SelectKBest

# selector = SelectKBest(score_func=f_regression, k=100)
# rna_reduced = selector.fit_transform(df_numeric, y=None)

# selected_cols = df_numeric.columns[selector.get_support()]
# rna_reduced_df = pd.DataFrame(rna_reduced, columns=selected_cols, index=row_names)

# # combine with histone dataset
# df_h = pd.read_csv( wd / 'histone_modification_data_dropna.csv')
# rna = rna_reduced_df.reset_index()  # ensures 'Tumor_Sample_Barcode' becomes a column
# rna = rna.rename(columns={'Tumor_Sample_Barcode': 'CellLineName'})

# rna = rna.add_prefix('gexpr_') # Add prefixes to column names (except for key)
# histone = histone.add_prefix('histone_')

# rna = rna.rename(columns={'gexpr_CellLineName': 'CellLineName'})
# histone = histone.rename(columns={'histone_CellLineName': 'CellLineName'})

# merged_df = pd.merge(rna, histone, on='CellLineName', how='inner')
# file_path = wd / 'gene_expression_and_histone_reduced.csv'
# merged_df.to_csv(file_path, index=False, header=True)


#### Process the imputed datasets
my_path = "Radiosensitivity Prediction/data/Imputed"
for filename in ["_raw", "_fillna", "_fillsamples"]:
    df = pd.read_csv(f"{my_path}/histone_imputed_2025{filename}.csv.gz")
    df["CellLineName"] = df["CellLineName"].apply(lambda x : x.split("_")[0])
    cleveland_cell_lines = list(set(pd.read_csv(f"{wd}/cleveland_auc_only.csv")['id']))
    df = df[df.iloc[:, 0].isin(cleveland_cell_lines)]
    df.set_index("CellLineName", inplace=True)
    df.to_csv(f"{my_path}/histone_imputed_2025{filename}.csv", index=True)
