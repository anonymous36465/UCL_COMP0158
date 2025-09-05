import os, sys, json, time
import pandas as pd
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from keras.activations import linear, relu, tanh, leaky_relu
from keras.losses import MeanSquaredError
from functools import partial
from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer

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
from architecture.mid_fusion_model import MLP_mid_fusion
from architecture.samples_weights_utils import *

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.decomposition import TruncatedSVD

# credit: jamesLJ100
def aggregate_results(tag, print_summary=False):
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

    if print_summary:
        print(summary_df)

y_all_labels = pd.read_csv("Radiosensitivity Prediction/data/Cleveland/cleveland_auc_full.csv")["auc"].values.flatten()

#################### SETUP
test_on_bulk = False
labels_dataset = "cleveland_auc_only.csv"

use_modules = False
if use_modules:
    data_name = "rna_imputed"
    number_of_modules = "500"

use_imputed = True
if use_imputed:
    data_name = "methylation_imputed"

use_cleveland = False
if use_cleveland:
    data_name = "histone_modification_data_fillna"
    data_alias = "histone"

weight_samples = "ByLabelFn" #Options: None, Continuoous, Discrete, PercentileDistance, Precomputed
weight_fn = make_weight_fn_mad(y_all_labels, alpha=1.5, eps=1e-3, normalize="global", clip=(None, 25)); 
weight_fn_name = "NLL_mad_alpha_1.5_eps_1e-3"
# weight_samples = "None"
# weight_fn = None
                                                                                                                                                                                                                    
################################################

wd = "Radiosensitivity Prediction"
download_dir = f"{wd}/data"

# if labels_dataset == "cleveland_auc_only.csv":
#     run_dir = f'{wd}/runs/runs_separated'
# else:
run_dir = f'{wd}/runs'


if use_imputed:
    data_dir = f"{download_dir}/Imputed"
    config["views"] = [
        ("imp", f"{data_name}.csv", selected_genes, 0, lambda x: x, lambda x: x),
    ]
    run_dir = f"{run_dir}/{data_alias}"

if use_cleveland:
    data_dir = f"{download_dir}/Cleveland"
    config["views"] = [
        ("imp", f"{data_name}.csv", selected_genes, 0, lambda x: x, lambda x: x),
    ]
    un_dir = f"{run_dir}/{data_alias}"


if use_modules:
    data_dir = f"{wd}/modules_experiment/gsva_scores"
    config["views"] = [("gsva_meth_imp", f"gsva_scores_{data_name}_{number_of_modules}.csv", None, 0 , lambda x:x, lambda x:x)]
    
    run_dir = f"{run_dir}/runs_modules/{data_name}/{number_of_modules}"

if weight_samples != "None":
    run_dir = f"{run_dir}/weighted_{weight_samples}"

    if weight_samples == "ByLabelFn":
        run_dir = f"{run_dir}/{weight_fn_name}"

if test_on_bulk: 
    run_dir = f"{run_dir}/test_on_bulk"

# config["feature_selector"] = SubsetSelectorWrapper(selector=VarianceThreshold, params={"threshold": 0.1})
# config["feature_selector"] = SubsetSelectorWrapper(selector=TruncatedSVD, params={"n_components": 50, "random_state": 42})
# config["feature_selector"] = SubsetSelectorWrapper(selector=SelectKBest, params={"score_func": f_regression, "k": 200}, requires_y = True)
# config["feature_selector"] = SubsetSelectorWrapper(selector=VarianceThreshold, params={"threshold": 0.00001})


if not os.path.exists(download_dir):
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

## Test on bulk after training on the tails
if test_on_bulk:
    df = pd.read_csv("Radiosensitivity Prediction/data/Cleveland/cleveland_auc_full.csv", index_col=0)
    auc = pd.to_numeric(df["auc"], errors="coerce")
    mask = (auc >= 2) & (auc <= 3)
    config["test_samples"] = df.index[mask].tolist()


config["data_dir"] = data_dir
config["run_dir"] = run_dir
config["run_id"] = "pnet"
config["labels"] = [(labels_dataset, 0)]
config["view_alignment_method"] = "drop samples"
config["tv_split_seed"] = 42
config["inner_kfolds"] = 5
config["outer_kfolds"] = 5
# config["test_samples"] = 0.1
config["use_validation_on_test"] = False
# config["val_metric"] = lambda x : r2_score(x["val_df"].ys, x["val_preds"])
# config["val_metric"] = lambda x: weighted_mse(x["val_df"].ys, x["val_preds"])
# config["val_metric"] = lambda x : r2_score(x["val_df"].ys, x["val_preds"], sample_weight=weight_fn(x["val_df"].ys))
config["val_metric"] = lambda x: nll(x["val_df"].ys, x["val_preds"], 0.3)


eval_weight_fn = make_weight_fn_mad(y_all_labels, alpha=2.0, eps=1e-1, normalize="global", clip=(None, 25))
eval_weight_fn_2 = make_weight_fn_percentile(y_all_labels, alpha=2.0, eps=1e-1, normalize="global", clip=(None, 25))

# Negative Log Likelihood
def nll(y_true, y_pred, sigma=0.1):
    """
    Gaussian Negative Log Likelihood with fixed sigma
    """
    n = len(y_true)
    const_term = 0.5 * np.log(2 * np.pi * sigma**2)
    error_term = 0.5 * ((y_true - y_pred) ** 2) / (sigma**2)
    return np.mean(const_term + error_term)

def weighted_mse(y_true, y_pred):
    squared_errors = (y_true - y_pred) ** 2
    return np.average(squared_errors, weights=eval_weight_fn(y_true))

def weighted_mse2(y_true, y_pred):
    squared_errors = (y_true - y_pred) ** 2
    return np.average(squared_errors, weights=eval_weight_fn_2(y_true))

config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"r2" : r2_score,
                                                                                    "explained_variance" : explained_variance_score,
                                                                                    "mse" : mean_squared_error,
                                                                                    "mae" : mean_absolute_error,
                                                                                    "w_mse": weighted_mse,
                                                                                    "w2_mse": weighted_mse2,
                                                                                    "nll_0.1": lambda x,y: nll(x, y, 0.1),
                                                                                    "nll_0.3": lambda x,y: nll(x, y, 0.3),
                                                                                    }, 
                                                                          "individual")]
#                                                                           plot_history]

config["weight_samples"] = weight_samples

config["weight_kwargs"] = {"func": weight_fn}




# # Run Kernel Regression
# gs_params = {"model_params" : {f"degree_{d}_alpha_{a}" : {"kernel" : "poly", "degree" : d, "alpha" : a}
#                             for d in [1, 2, 3] for a in [0.01, 0.1, 0.5, 1, 3, 5, 10, 50]}}
# config["model"] = KernelRidge
# config["run_id"] = "krr"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("krr")



# gs_params = {
#     "model_params": {
#         f"degree_{d}_alpha_{a}": {
#             # target transformer (spreads out the dense 2–3 region)
#             "transformer": PowerTransformer(method="yeo-johnson", standardize=True),
#             # QuantileTransformer(output_distribution="normal")

#             # inner regressor for TTR (your KRR with current grid)
#             "regressor": KernelRidge(kernel="poly", degree=d, alpha=a),
#         }
#         for d in [1, 2, 3]
#         for a in [0.01, 0.1, 0.5, 1, 3, 5, 10, 50]
#     }
# }

# config["model"] = TransformedTargetRegressor
# config["run_id"] = "krr_ttr"
# config["task"] = "regression"
# config["grid_search"] = construct_gs_params(gs_params)

# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("krr_ttr")

# # KRR with weighted samples
# # Run Kernel Regression (RBF and Sigmoid kernels)
# gs_params = {"model_params" : {f"{k}_gamma_{g}_alpha_{a}" : {"kernel" : k, "gamma" : g, "alpha": a}
#                             for k in ["rbf", "sigmoid"] for g in [0.001, 0.005, 0.01, 0.03, 0.05]
#                             for a in [0.1, 1, 10]}}
# config["model"] = KernelRidge
# config["run_id"] = "krr2_weighted"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("krr2_weighted")


# Run Lasso Regression
gs_params = {"model_params" : {f"alpha_{a}": {"alpha" : a}
                                for a in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]}}
config["model"] = Lasso
config["run_id"] = "lasso_weighted"
config["task"] = "regression"
config["results_processors"] = config["results_processors"]
config["grid_search"] = construct_gs_params(gs_params)
pipeline = MLPipeline(config)
pipeline.run_crossvalidation()
aggregate_results("lasso_weighted")


# # SVR with epsilon parameter.
# C_values = [1 / a for a in [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]]
# degrees = [1, 2, 3]
# gammas = [0.001, 0.01, 0.1]

# gs_params = {
#     "model_params": {
#         # Poly kernel with degree
#         **{ f"poly_C_{c}_deg_{d}": {"kernel": "poly","C": c,"degree": d,"gamma": "scale"        }
#             for c in C_values
#             for d in degrees
#         },
#     }
# }

# config["model"] = SVR
# config["run_id"] = "svr"
# config["task"] = "regression"
# config["grid_search"] = construct_gs_params(gs_params)

# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("svr")


# gs_params = {
#     "model_params": {
#         **{f"{k}_C_{c}_gamma_{g}": {"kernel": k,"C": c,"gamma": g
#             }
#             for k in ["rbf", "sigmoid"]
#             for c in C_values
#             for g in gammas
#         },
#     }
# }

# config["model"] = SVR
# config["run_id"] = "svr2"
# config["task"] = "regression"
# config["grid_search"] = construct_gs_params(gs_params)

# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("svr2")

