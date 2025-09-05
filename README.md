# 

Note: The code was developed drawing on the configuration and data processing setup provided with P-NET (Haitham A Elmarakeby et al. in “Biologically informed deep neural network for prostate cancer discover" (paper link: https://www.nature.com/articles/s41586-021-03922-4)), which I adapted and extended for our own validation framework.


## Installation
Recommended to install with some kind of environment managing software like conda

```
conda create --name pnet-repro python=3.9.16
```
Then activate the environment
```
conda activate pnet-repro
```
Install requirements file from the root directory of the repository
```
pip install -r requirements.txt
```
To run with GPU support
```
conda install -c conda-forge cudatoolkit=11.0 cudnn=8.0
```


## Structure of the repository
All code pertaining to P-Net and supporting pipeline can be found in the architecture folder.

### Architecture
1. pnet_model.py - contains the code for constructing TensorFlow implementation of P-Net from Reactome.
2. pipeline.py - contains code for the pipeline object that is used to configure and run experiments with P-Net and other models. MLPipeline is used for any sklearn type model and TFPipeline is used for P-Net but in general the Pipeline class is designed such that you can subclass it and override _train to do any platform / framework specific changes before fitting the model.
3. data_utils.py - contains code for the classes used for loading multiple views of a dataset and integrates them for use with P-Net and other models. Features from multiple views are aligned according to alignment_ids and they features of the same alignment_id are kept contiguously when concatenated together.
4. layers_custom.py - contains Diagonal and SparseTF which are mostly unchanged from the original P-Net repository
5. callbacks_custom.py - callback functions to be used with tensorflow models, mostly unchanged from the original P-Net repository
6. coef_weights_utils.py - functions to help with extracting coefficients and outputs from the tensorflow model layers for the purpose of deeplift / explainability. Mostly unchanged from original P-Net repository
7. deepexplain - folder containing deepexplain / deeplift code. Mostly unchanged from original P-Net repository
8. Reactome - folder containing the reactome data used by P-Net. Unchanged from the original P-Net repository
9. evaluation.py - contains functions that are attached to the results_processors variable in the configuration file to allow flexibility in what kinds of evaluations to perform on each run e.g AUC, accuracy, F1, train history, deeplift etc.
10. pnet_config.py - template configuration file with values set to what was specified in the P-Net paper (which is not necessarily the same as in the original P-Net github repository). Gives an idea of what is available for configuration and what are expected inputs

### Usage
The code is meant to be used by importing / copying the pnet_config.py file and editing it to suit your experiment needs. There are a few config options that are experiment specific and listed below
1. run_id - specifies the tag for the current experiment run
2. data_dir - path to the folder containing all the data for the experiments
3. run_dir - path to the folder you wish to store all the outputs of your experiments
4. views - list of tuples containing paths to the datasets you wish to load in, as well as an identifying tag for what kind of data view it is and functions to preprocess the data and extract alignment_ids from the headers
5. view_alignment_method - a string to specify how to deal with NAs when aligning different views
6. labels - the response variables you wish to make a prediction for
   
The pipeline has 2 run methods. The first method is run_single_split. This is used when you do not want to do full crossvalidation, and lets you split the data into train, validation, and test sets either based on a random seed or a lists of sample ids for each split. To specify these splits you need to set the following config variables
1. train_samples - either a list of sample ids or a float between 0 and 1 specifying the size of train set
2. val_samples - either a list of sample ids or a float between 0 and 1 specifying the size of validation set
3. test_samples - either a list of sample ids or a float between 0 and 1 specifying the size of train samples
   
The second method is run_crossvalidation. For this you will need to specify a few extra config variables
1. tv_split_seed - random seed to make train-validation split reproducible
2. inner_kfolds - number of train-validation splits to compute per test split
3. outer_kfolds - number of development-test splits for the crossvalidation
4. validation_prop - a float specifying proportion of the development set to be used for validation. Only used if inner_kfolds is set to 1
#### Customisable entry points
The pipeline was designed to let users customise different steps in the model development pipeline beyond specifying parameters and hyperparameters of the model.
1. feature_selector - a class that follows the same format as a sklearn model e.g with fit, fit_transform, transform methods. The purpose of this entry point is to let users define a feature selection method that can be applied during each crossvalidation training run data independently and then apply the feature selection to the validation and test sets when evaluating
2. data_augmentor - a function that takes in the training dataset and outputs an augmented training dataset e.g with artificial new data points
3. results_processors - a list of functions that are run after a model for a training run has been completed. This can be various metrics, plotting training history, saving model weights etc.
   
#### Grid search
Grid searching can be done by specifying the desired parameters to gridsearch over in a dictionary where each config item that you want to gridsearch over is a key in the dictionary and the value is a dictionary of the parameters to be searched. The keys of the inner dictionary are just identifiers for that particular parameter setting and the value is the actual value you want to gridsearch over. You then use construct_gs_params on this dictionary and assign the output to grid_search variable in the config
