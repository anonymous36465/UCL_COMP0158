
0. Download the experiment data and organize it in the folders:
* imputed data:
    * data/Clevelandmethyaltion_imputed.csv
    * data/Cleveland/rna_imputed.csv
* Mice PDX models data
    * "/data/Mice/data_methylation_hm450.txt"
    * /data/Mice/data_mrna_seq_rpkm.txt
    * run `rna_normalization.R` on the Mice RNA data

1. `preprocess.py` Preprocess the Mice PDX data
* (delete duplicated rows) and save back to the same directory
* preprocess data (impute missing genes) and save it to "data/mice" folder

2. `calculate_GSVA_on_mice_data.ipynb` Compute the GSVA scores on the mice data, using original as reference
* get both the original and mice df
* impute missing genes with mean original value
* remove random rows and put mice in there
-> save to data/prepare_mice_data_for_gsva
* (run GSVA using the `/modules_experiments/calculate_GSVA.R`)
* load the results, extract the results for mice
-> save to: data/mice_gsva/mice_gsva_scores_{data_name}_{number_of_modules}.csv


3. `train_model.py`
* using the original dataset train a model on all data
-> save model to: /runs/runs_modules/MICE/{data_type}_{data_name}/{model_name}.sav
where data_type is either nothing, gsva_scores or mice_gsva_scores


4. `eval model.py`
* specify: data_name, model_name, data_type
* load the processed mice data from /data/
* plot predictions and report other metrics

