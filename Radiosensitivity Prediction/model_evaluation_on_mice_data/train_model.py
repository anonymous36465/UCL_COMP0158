import os, sys, json, time
import pandas as pd
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from keras.activations import linear, relu, tanh, leaky_relu
from keras.losses import MeanSquaredError
from functools import partial
from sklearn.kernel_ridge import KernelRidge

from sklearn.svm import SVR
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.dummy import DummyRegressor

# pip install pygam lightgbm xgboost
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from pygam import LinearGAM


sys.path.insert(0, os.getcwd())
from architecture.data_utils import *
from architecture.pnet_config import *
from architecture.pipeline import *
from architecture.evaluation import *
from architecture.callbacks_custom import step_decay, FixedEarlyStopping
from architecture.feature_selection import SubsetSelectorWrapper
from architecture.feature_scaling import StandardScalerProcessor
from architecture.samples_weights_utils import *

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.decomposition import TruncatedSVD

# data_name = "methylation_imputed_500"
# data_type = "gsva_scores"
# model_name = "krr"
y_all_labels = pd.read_csv("Radiosensitivity Prediction/data/Cleveland/cleveland_auc_full.csv")["auc"].values.flatten()

############# Run Details
data_name = "methylation_imputed"
data_type = None #"gsva_scores"
model_name = "lasso"

weight_samples = "ByLabelFn" #Options: None, Continuoous, Discrete, PercentileDistance, Precomputed
weight_fn = make_weight_fn_mad(y_all_labels, alpha=1.5, eps=1e-3, normalize="global", clip=(None, 25)); 
weight_fn_name = "mad_alpha_1.5_eps_1e3"
# weight_fn_name = None
###############


def aggregate_results(tag):
    results_path = os.path.join(run_dir, tag, "results.csv")
    output_path = os.path.join(run_dir, tag, "aggregated_test_results.csv")

    if os.path.exists(results_path):
        df = pd.read_csv(results_path, index_col=0)
    else:
        print(f"results.csv not found at: {results_path}")

    test_df = df[df['index'] == "test"]
    test_df = test_df.select_dtypes(include='number')
    mean_scores = test_df.mean()
    std_scores = test_df.std()

    summary_df = pd.DataFrame({
        "mean": mean_scores,
        "std": std_scores
    })
    summary_df.index.name = 'metric'
    summary_df.to_csv(output_path)

# Download data if not done so already and set up run directory
wd = "Radiosensitivity Prediction"
download_dir = f"{wd}/data"

data_dir = f"{download_dir}/Cleveland"
if data_type == "gsva_scores":
    data_dir = f"{wd}/modules_experiment/gsva_scores"
elif data_type == "mice_gsva_scores":
    data_dir = f"data/mice_gsva"

# run_dir = f"{wd}/runs/runs_modules/MICE/{data_name}"
run_dir = f"{wd}/model_evaluation_on_mice_data/runs/{data_name}"
if data_type is not None:
    # run_dir = f"{wd}/runs/runs_modules/MICE/{data_type}_{data_name}"
    run_dir = f"{wd}/model_evaluation_on_mice_data/runs/{data_type}_{data_name}"


if not os.path.exists(download_dir):
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

# # selected_genes = list(set(pd.read_csv(f"{download_dir}/hugo_genes.txt", sep="\t")["symbol"]))
# with open(f"{download_dir}/feature_subset/best1200.txt", "r") as f:
#     selected_groups = [line.strip() for line in f if line.strip()]

# prepare config
config["data_dir"] = data_dir
config["run_dir"] = run_dir

# config["views"] = [("rna_imputed", f"{data_name}.csv", feat_subset, 0, lambda x: x, lambda x: x)]
filename = data_name
if data_type is not None:
    filename = f"{data_type}_{data_name}"
    
config["views"] = [("meth_imp", f"{filename}.csv", None, 0, lambda x: x, lambda x: x)]

config["view_alignment_method"] = "drop samples"
config["labels"] = [("cleveland_auc_full.csv", 0)]
config["tv_split_seed"] = 42
config["inner_kfolds"] = 0
config["outer_kfolds"] = 0
# config["test_samples"] = 0.1
config["use_validation_on_test"] = False
config["val_metric"] = lambda x : r2_score(x["val_df"].ys, x["val_preds"])
config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"r2" : r2_score,
                                                                                    "explained_variance" : explained_variance_score,
                                                                                    "mse" : mean_squared_error,
                                                                                    "mae" : mean_absolute_error}, 
                                                                          "individual")]
config["train_samples"] =  1.0        
config['val_samples'] = 0.0
config['test_samples'] = 0.0       

if weight_fn_name is not None:
    config["weight_samples"] = weight_samples
    config["weight_kwargs"] = {"func": weight_fn}
                                             

if model_name == "krr":
    gs_params = {"model_params" : {f"degree_{d}_alpha_{a}" : {"kernel" : "poly", "degree" : d, "alpha" : a}
                             for d in [3] for a in [10]}}
    config["model"] = KernelRidge
elif model_name == "enet":
    gs_params = {"model_params" : {f"alpha_{a}_ratio_{r}" : {"alpha": a, "l1_ratio": r}
                            for a in [0.3] for r in [0.01]}}
    config["model"] = ElasticNet
elif model_name == "lasso":
    gs_params = {"model_params": {f"alpha_{a}" : {"alpha": a}
                            for a in [0.1]}}
    config["model"] = Lasso


if weight_fn_name is not None:
    model_name = model_name + "_" + weight_fn_name
config["run_id"] = model_name
config["task"] = "regression"
config["results_processors"] = config["results_processors"]
config["grid_search"] = construct_gs_params(gs_params)
pipeline = MLPipeline(config)
model = pipeline.run_single_split(return_model=True)

aggregate_results(model_name)


import pickle

# # save the model to disk
# filename = f"{run_dir}/{model_name}.sav"
# # os.makedirs(os.path.dirname(filename), exist_ok=True)
# pickle.dump(model, open(filename, 'wb')) 

# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))

# Strip the function that causes pickling to fail

model.weight_kwargs = {} #No need to store the weighting function

filename = f"{run_dir}/{model_name}.sav"
with open(filename, "wb") as f:
    pickle.dump(model, f)

