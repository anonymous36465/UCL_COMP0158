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

# Download data if not done so already and set up run directory
wd = "Radiosensitivity Prediction"
download_dir = f"{wd}/data"
data_dir = f"{download_dir}/Cleveland"

run_dir = f"{wd}/runs/runs_methylation/hugo_and_VarianceThreshold_0.1"

# select100best_withincv" - the one that used more data to select features.

# if not os.path.exists(download_dir):
#     with open(f"{wd}/src/download_data.py") as file:
#         exec(file.read())


# with open(f"{download_dir}/qrcp10.txt", "r") as f:
#     selected_features_qr = [line.strip() for line in f]

selected_genes = list(set(pd.read_csv(f"{download_dir}/hugo_genes.txt", sep="\t")["symbol"]))

# selected_features = None

# prepare config


# Define views with placeholder preprocess/alignment funcs
# config["views"] = [
#     ("hist_imp_2025", f"methylation_imputed.csv", None, 0, lambda x: x, lambda x: x),
# ]
config["views"] = [("methylation", f"CCLE_Methylation_TSS1kb_20181022.csv", selected_genes, 0, lambda x : x, lambda x : x)]
# config["data_dir"] = f"{wd}/modules_experiment/gsva_scores"

# config["views"] = [
#     ("gexpr", "cleveland_gene_expression.csv", selected_genes, 0, lambda x:x, lambda x:x)
# ]

# config["views"] = [
#     ("hist_h3k4", f"histone_modification_data_fillna.csv", None, 0, lambda x:x, lambda x:x)
# ]

# data_dir = f"{wd}/modules_experiment/gsva_scores"
# config["views"] = [("gsva_rna_500", f"gsva_scores_rna_imputed_500.csv", None, 0, lambda x : x, lambda x : x),
#                     ("gsva_meth_500", f"gsva_scores_methylation_imputed_500.csv", None, 0, lambda x : x, lambda x : x),
#                     ("gsva_prot_500", f"gsva_scores_proteomics_imputed_500.csv", None, 0, lambda x : x, lambda x : x)]

# data_name = "rna_imputed"
# number_of_modules = "most-enriched"
# data_dir = f"{wd}/modules_experiment/gsva_scores"
# config["views"] = [("gsva_meth_imp", f"gsva_scores_{data_name}_{number_of_modules}.csv", None, 0 , lambda x:x, lambda x:x)]

# # run_dir = f"{wd}/runs/runs_separated/{data_name}"
# run_dir = f"{wd}/runs/runs_modules/{data_name}/{number_of_modules}"

# config["feature_selector"] = SubsetSelectorWrapper(selector=VarianceThreshold, params={"threshold": 0.1})
# config["feature_selector"] = SubsetSelectorWrapper(selector=TruncatedSVD, params={"n_components": 50, "random_state": 42})

# config["feature_selector"] = SubsetSelectorWrapper(selector=SelectKBest, params={"score_func": f_regression, "k": 100}, requires_y = True)

# config["feature_selector"] = SubsetSelectorWrapper(selector=VarianceThreshold, params={"threshold": 0.1})



# select100best_withincv" - the one that used more data to select features.


if not os.path.exists(download_dir):
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

config["data_dir"] = data_dir
config["run_dir"] = run_dir
config["run_id"] = "pnet"

config["view_alignment_method"] = "drop samples"
config["labels"] = [("cleveland_auc_full.csv", 0)]
config["tv_split_seed"] = 42
config["inner_kfolds"] = 5
config["outer_kfolds"] = 5
# config["test_samples"] = 0.1
config["use_validation_on_test"] = False
config["val_metric"] = lambda x : r2_score(x["val_df"].ys, x["val_preds"])
config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"r2" : r2_score,
                                                                                    "explained_variance" : explained_variance_score,
                                                                                    "mse" : mean_squared_error,
                                                                                    "mae" : mean_absolute_error}, 
                                                                          "individual")]
#                                                                           plot_history]


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



# # Run Kernel Regression (RBF and Sigmoid kernels)
# gs_params = {"model_params" : {f"{k}_gamma_{g}_alpha_{a}" : {"kernel" : k, "gamma" : g, "alpha": a}
#                             for k in ["rbf", "sigmoid"] for g in [0.001, 0.005, 0.01, 0.03, 0.05]
#                             for a in [0.1, 1, 10]}}
# config["model"] = KernelRidge
# config["run_id"] = "krr2"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("krr2")

# # Run Lasso Regression
# gs_params = {"model_params" : {f"alpha_{a}": {"alpha" : a}
#                                 for a in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]}}
# config["model"] = Lasso
# config["run_id"] = "lasso"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("lasso")

# Run ElasticNet (L1 and L2 combined)
gs_params = {"model_params" : {f"alpha_{a}_ratio_{r}" : {"alpha": a, "l1_ratio": r}
                                # for a in [0.05, 0.1, 0.3, 0.5] for r in [0.1, 0.15, 0.2, 0.5, 0.7, 0.8, 0.9]}}
                                # for a in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65] for r in [0.01, 0.05, 0.075, 0.1, 0.15]}}
                                for a in [0.001, 0.05, 0.1, 0.3, 0.5, 1.0] for r in [0.0, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0]}}

config["model"] = ElasticNet
config["run_id"] = "enet"
config["task"] = "regression"
config["results_processors"] = config["results_processors"]
config["grid_search"] = construct_gs_params(gs_params)
pipeline = MLPipeline(config)
pipeline.run_crossvalidation()
aggregate_results("enet")

# # # Run simple MLP
# gs_params = {
#     "model_params": {
#         f"{h}_alpha_{a}": {
#             "hidden_layer_sizes": h,
#             "alpha": a,
#             "max_iter": 500,
#             "early_stopping": True,
#         }
#         for h in [(8,4,2)] 
#         for a in [10.0, 50.0, 100.0]
#     }
# }
# config["model"] = MLPRegressor
# config["run_id"] = "mlp"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]  # optional
# config["grid_search"] = construct_gs_params(gs_params)

# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("mlp")

# # # Run simple 2layer MLP
# gs_params = {
#     "model_params": {
#         f"({hidden1},{hidden2},)_alpha_{a}": {
#             "hidden_layer_sizes": (hidden1, hidden2,),
#             "alpha": a,
#             "max_iter": 300,
#             "batch_size": 32,
#             # "early_stopping": True, #sets asside a validation set
#             "random_state": 42,
#             # "verbose": True,
#             "n_iter_no_change":5
#         }
#         for hidden1 in [64, 32, 16] 
#         for hidden2 in [32, 16, 8]
#         for a in [0.1, 1.0, 5.0]
#     }
# }
# config["model"] = MLPRegressor
# config["run_id"] = "mlp2"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]  # optional
# config["grid_search"] = construct_gs_params(gs_params)

# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("mlp2")

# # Run XGBoost
# gs_params = {
#     "model_params": {
#         f"depth_{d}_lr_{lr}_n_{n}": {
#             "max_depth": d,
#             "learning_rate": lr,
#             "n_estimators": n,
#             # "subsample": 0.8,
#             # "colsample_bytree": 0.8,
#             "objective": "reg:squarederror",
#             "random_state": 42,
#         }
#         for d in [3, 5]
#         for lr in [0.005, 0.01]
#         for n in [300, 500, 800]
#     }
# }

# config["model"] = XGBRegressor
# config["run_id"] = "xgb"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)

# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("xgb")

# Run XGBoost
# gs_params = {
#     "model_params": {
#         f"depth_{d}": {
#             "max_depth": d,
#             "n_estimators": 300,
#             "subsample": 0.8,
#             "colsample_bytree": 0.8,
#             "objective": "reg:squarederror",
#             "random_state": 42,
#         }
#         for d in [3]
#     }
# }

# config["model"] = XGBRegressor
# config["run_id"] = "xgb_small"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)

# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("xgb_small")

# # Run LightGBM
# gs_params = {
#     "model_params": {
#         f"lgb_d{d}_lr{lr}_alpha{a}": {
#             "max_depth": d,
#             "learning_rate": lr,
#             "n_estimators": 1000,
#             "subsample": 0.7,
#             "colsample_bytree": 0.7,
#             "reg_alpha": a,
#             "reg_lambda": 10.0,
#             "min_child_weight": 5,
#             "random_state": 42,
#         }
#         for d in [2, 3]
#         for lr in [0.001, 0.01, 0.05]
#         for a in [1.0, 2.0, 3.0]
#     }
# }

# config["model"] = LGBMRegressor
# config["run_id"] = "lgb"
# config["task"] = "regression"
# config["grid_search"] = construct_gs_params(gs_params)
# config["results_processors"] = config["results_processors"]
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("lgb")


# # # Run mid-fusion MLP
# gs_params = {
#     "model_params": {
#         f"default": {
#             "datasets": ["rna", "meth", "prot"],
#             "feature_layers": 3,
#             "feature_hidden_dim": 32,
#             "combining_layers": [128, 32, 1],
#             "num_epochs": 60,
#             "learning_rate": 1e-4,
#             "dropout_prob":0.1
#         }
#     }
# }
# config["model"] = MLP_mid_fusion
# config["run_id"] = "mlp_mid"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)

# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("mlp_mid", print_summary=True)

# C_values = [1 / a for a in [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100]]
# degrees = [1, 2, 3]
# gammas = [0.001, 0.01, 0.1]

# gs_params = {
#     "model_params": {
#         # Poly kernel with degree
#         **{ f"poly_C_{c}_deg_{d}": {"kernel": "poly","C": c,"degree": d,"gamma": "scale",
#         # "epsilon":0.8
#         }
#             for c in C_values
#             for d in degrees
#         },
#         # # RBF and sigmoid kernels with gamma
#         # **{f"{k}_C_{c}_gamma_{g}": {"kernel": k,"C": c,"gamma": g
#         #     }
#         #     for k in ["rbf", "sigmoid"]
#         #     for c in C_values
#         #     for g in gammas
#         # },
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
#         # # Poly kernel with degree
#         # **{ f"poly_C_{c}_deg_{d}": {"kernel": "poly","C": c,"degree": d,"gamma": "scale",
#         #     "epsilon":1.0
#         # }
#         #     for c in C_values
#         #     for d in degrees
#         # },
#         # RBF and sigmoid kernels with gamma
#         **{f"{k}_C_{c}_gamma_{g}": {"kernel": k,"C": c,"gamma": g, 
#         "epsilon": 0.8
#             }
#             for k in ["rbf", "sigmoid"]
#             for c in C_values
#             for g in gammas
#         },
#     }
# }

# config["model"] = SVR
# config["run_id"] = "svr2_epsilon_0.8"
# config["task"] = "regression"
# config["grid_search"] = construct_gs_params(gs_params)

# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("svr2_epsilon_0.8")

