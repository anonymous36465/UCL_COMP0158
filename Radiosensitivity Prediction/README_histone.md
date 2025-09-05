## Data Download
The histone modification data should be downloaded manually from https://depmap.org/portal/data_page/?tab=allData and saved into the data folder.

In `preprocess.py`, I comment out automatic download and get the data `CCLE_RRBS_TSS1kb_20181022.txt.gz` manually from https://depmap.org/portal/data_page/?tab=allData, unzip it on my computer and add to data/Cleveland.

## Preprocessing

We make two files `histone_modification_data_dropna.csv` and `histone_modification_data_fillna.csv` for the two different ways of handling NaN vlaues. Only cell lines exsisting in the Cleveland AUC response data are included in these files.

I also make the `gene_expression_and_histone.csv` that comncatenates the cleveland_gene_expression.csv with histone_modification_data_dropna.csv. The column names are prefixed with 'histone' and 'gexpr' respectivelly. The gene expression data features are first filtered dropping columns with many NaN values or small variance.

## Feature Selection

Can be added to config["feature_selection"] by providing a sklearn feature selector. The `feature_selection.py` file has a wrapper of such fuction enabling to do the SelectKBest selection of only features matching a prefix. - In our case it only does feature selction of 'gexpr' features leaving all of 'histone' ones.


