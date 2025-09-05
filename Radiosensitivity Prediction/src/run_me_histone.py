import os, sys, json, time
import pandas as pd
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from keras.activations import linear, relu, tanh, leaky_relu
from keras.losses import MeanSquaredError
from functools import partial
from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.feature_selection import VarianceThreshold 

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
from architecture.feature_selection import SelectKBestWrapper, SubsetSelectorWrapper

# credit: jamesLJ100
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
run_dir = f"{wd}/runs_histone/histone_and_gexpr/varianceThreshold_0.1"

# select100best_withincv" - the one that used more data to select features.

if not os.path.exists(download_dir):
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

# with open(f"{download_dir}/qrcp10.txt", "r") as f:
#     selected_features_qr = [line.strip() for line in f]

selected_genes = list(set(pd.read_csv(f"{download_dir}/hugo_genes.txt", sep="\t")["symbol"]))

selected_features = None

# prepare config
config["data_dir"] = data_dir
config["run_dir"] = run_dir
config["run_id"] = "pnet"

Define views with placeholder preprocess/alignment funcs
config["views"] = [
    ("gexpr_and_histone", f"gene_expression_and_histone.csv", None, 0, lambda x: x, lambda x: x),
]

# Will at each fold choose 100 features out of the ones that have name starting with "gexpr"
# config["feature_selector"] = SelectKBestWrapper(k=100, dataset_name="gexpr_and_histone", featurename_to_reduce="gexpr")
# config["feature_selector"] = SubsetSelectorWrapper(selector=VarianceThreshold, params={"threshold":0.1})
SubsetSelectorWrapper(selector=TruncatedSVD, params={"n_components": 50, "random_state": 42})

config["view_alignment_method"] = "drop samples"
config["labels"] = [("cleveland_auc_only.csv", 0)]
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

# Run Kernel Regression
gs_params = {"model_params" : {f"degree_{d}_alpha_{a}" : {"kernel" : "poly", "degree" : d, "alpha" : a}
                            for d in [1, 2, 3] for a in [0.1, 0.3, 0.5, 0.6, 1, 1.5, 2, 2.5]}} #3, 3.5, 4, 4.5, 5]}}
config["model"] = KernelRidge
config["run_id"] = "krr"
config["task"] = "regression"
config["results_processors"] = config["results_processors"]
config["grid_search"] = construct_gs_params(gs_params)
pipeline = MLPipeline(config)
pipeline.run_crossvalidation()
aggregate_results("krr")


# # Run Kernel Regression (RBF and Sigmoid kernels)
# gs_params = {"model_params" : {f"{k}_gamma_{g}" : {"kernel" : k, "gamma" : g}
#                             #    for k in ["rbf", "sigmoid"] for g in [0.001, 0.005, 0.01, 0.03, 0.05]}}
#                                for k in ["rbf", "sigmoid"] for g in [0.0008, 0.001, 0.0015, 0.002, 0.003]}}
# config["model"] = KernelRidge
# config["run_id"] = "krr2"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("krr2")

# Run Lasso Regression
gs_params = {"model_params" : {f"alpha_{a}": {"alpha" : a}
                                for a in [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.3, 0.5]}}
config["model"] = Lasso
config["run_id"] = "lasso"
config["task"] = "regression"
config["results_processors"] = config["results_processors"]
config["grid_search"] = construct_gs_params(gs_params)
pipeline = MLPipeline(config)
pipeline.run_crossvalidation()
aggregate_results("lasso")

# Run ElasticNet (L1 and L2 combined)
gs_params = {"model_params" : {f"alpha_{a}_ratio_{r}" : {"alpha": a, "l1_ratio": r}
                                # for a in [0.05, 0.1, 0.3, 0.5] for r in [0.1, 0.15, 0.2, 0.5, 0.7, 0.8, 0.9]}}
                                for a in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65] for r in [0.01, 0.05, 0.075, 0.1, 0.15]}}

config["model"] = ElasticNet
config["run_id"] = "enet"
config["task"] = "regression"
config["results_processors"] = config["results_processors"]
config["grid_search"] = construct_gs_params(gs_params)
pipeline = MLPipeline(config)
pipeline.run_crossvalidation()
aggregate_results("enet")

# # Run simple MLP
# gs_params = {
#     "model_params": {
#         f"{h}_alpha_{a}_fillna": {
#             "hidden_layer_sizes": h,
#             "alpha": a,
#             "max_iter": 500,
#             "early_stopping": True,
#         }
#         for h in [(8,), (8, 4), (8,4,2)] 
#         for a in [1.0, 8.0, 10.0, 15.0, 100.0]
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

# Run XGBoost
gs_params = {
    "model_params": {
        f"depth_{d}_lr_{lr}_n_{n}": {
            "max_depth": d,
            "learning_rate": lr,
            "n_estimators": n,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        for d in [3, 4]
        for lr in [0.001, 0.005, 0.01]
        # for n in [500, 700, 800, 900]
        for n in [600]
    }
}

# config["model"] = XGBRegressor
# config["run_id"] = "xgb"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)

# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("xgb")

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




"""
# Compile results
def compile_results(tag, gridsearch):
    results = []
    for i, fold in enumerate([x for x in os.listdir(f"{run_dir}/{tag}") if os.path.isdir(f"{run_dir}/{tag}/{x}")]):
        for cv in [x for x in os.listdir(f"{run_dir}/{tag}/{fold}") if os.path.isdir(f"{run_dir}/{tag}/{fold}/{x}")]:
            with open(f"{run_dir}/{tag}/{fold}/{cv}/config.txt") as f:
                run_config = json.loads(f.read())
            result = pd.read_csv(f"{run_dir}/{tag}/{fold}/{cv}/summary_results.csv")
            result.columns = ["split"] + list(result.columns[1:])
            result["test_fold"] = i
            for k,v in gridsearch.items():
                result[k] = run_config[v]
            results.append(result)
    results = pd.concat(results)
    val_means = results.loc[results["split"] == "val"].groupby(list(gridsearch.keys()) + ["test_fold"])["auc_r2"].mean().reset_index()
    top = val_means.iloc[val_means.groupby("test_fold")["auc_r2"].idxmax(), ]
    filt = None
    for k,v in gridsearch.items():
        if filt is None:
            filt = results[k] == top[k].iat[0]
        else:
            filt = (results[k] == top[k].iat[0]) & filt
    results = results.loc[filt]
    aggresults = results.groupby("split")[["auc_r2", "auc_explained_variance", "auc_mse", "auc_mae"]].agg(["mean", "std"])
    
    hyperparams = "_".join([top[k].iat[0] for k in gridsearch.keys()])
    aggresults.to_csv(f"{wd}/{tag}_{hyperparams}_results.csv")
    return results.loc[results["split"] == "test"]

# dense_results = compile_results("dense", {"reg" : "model_params_choice", "es" : "fitting_params_choice"})
# pnet_results = compile_results("pnet", {"reg" : "model_params_choice", "es" : "fitting_params_choice"})
krr_results = compile_results("krr", {"hyper" : "model_params_choice"})

# metrics = ["auc_r2", "auc_explained_variance", "auc_mse", "auc_mae"]
# pvd = [ttest_ind(pnet_results[x], dense_results[x]).pvalue for x in metrics]
# pvk = [ttest_ind(pnet_results[x], krr_results[x]).pvalue for x in metrics]
# svd = [ttest_ind(krr_results[x], dense_results[x]).pvalue for x in metrics]
# sigresults = pd.DataFrame((pvd, pvk, svd), columns=metrics, index=["pnet_v_dense", "pnet_v_krr", "krr_v_dense"])
# sigresults.to_csv(f"{wd}/significance_tests.csv")
"""

