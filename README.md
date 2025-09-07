# COMP0158 - A thesis submitted for the degree of MSc Data Science and Machine Learning

_Note: The code was developed drawing on the configuration and data processing setup provided with P-NET (Haitham A Elmarakeby et al. in “Biologically informed deep neural network for prostate cancer discover" (paper link: https://www.nature.com/articles/s41586-021-03922-4)), which I adapted and extended for our own validation framework._


## Installation

```
conda create --name pnet-repro python=3.9.16
```
```
conda activate pnet-repro
```
```
pip install -r requirements.txt
```
To run with GPU support:
```
conda install -c conda-forge cudatoolkit=11.0 cudnn=8.0
```


## Structure of the repository

### Architecture - Added functionality
1. bayesian_regressor.py - Negative Log Likelihood loss for both per-sample and fixed noise. Simple Bayesian MLP with NLL loss and MC dropout.
2. evaluation.py (edited) - added evaluation specific to classification
4. feature_selection.py - Enables subset selection within the CV applied to either all datasets or only one of them.
5. mid_fusion_bayesian_mlp.py - The Bayesian version of mid_fusion_model that includes MC dropout and negative log likelihood loss.
6. mid_fusion_model.py - The multi-head MLP with modality-specific heads and a joining MLP. Option to turn off a head or set it to identity for ablation studies. Learning rate scheduler included.
7. pipeline.py (edited) - To incorporate sample weighting through config["weight_samples"] and self.config["weight_kwargs"] - strategies for fixed weights, within CV calculation and both regression and classification variants. Also addition of return_model argument so that the model is returned, can be saved and then externally evaluated.
8. samples_weights_utilis.py - The MAD-based and Perc-based sample weighting strategies


### Radiosensitivity Prediction/src - data download and preprocessing
* data should be downloaded and preprocessed using these files:
   * download_data.py, download_R_datasets.R
   * preprocess.py, preprocess_histone.py

### Radiosensitivity Prediction/src/run_me_....py
The 'run_me...' files contain the main CV experiments on ML models.
  
### Radiosensitivity Prediction/Modules Experiment
* The main.ipynb file walks through how to generate GSVA Module-based scores

### Radiosensitivity Prediction/Mean Methylation Experiments
Contains experiments related to the mean methylation experiments on both CCLE and mice data (mean_methylation_on_CCLE.ipynb, mean_methylation_on_PDX.ipynb).

mean_methylation_analys.ipynb, data_differences_exploration.ipynb, preprocess_data.ipynb and mapping_raw_data.R do the necessary data preprocessing

special_genes_on_mie_data explores the TT53 and PTEN gene mutation relation to survival in PDX models.

### MOSA Imputation
The adapted code can be run from my colab notebook: https://drive.google.com/file/d/1nxggheuj4Rld-xn79Y41VqMd13K0QYkS/view?usp=sharing . The histone imputation run can be reproduced using these hyperparameters setup: https://drive.google.com/file/d/1yDzSIbekudaIY8d0zK17kI0CgbpRJ1S5/view?usp=sharing .

### Radiosensitivity Prediction/Model Evaluation on Mice Data

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

