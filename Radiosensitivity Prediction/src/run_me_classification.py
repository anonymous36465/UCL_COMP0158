import os, sys, json, time
import pandas as pd
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# pip install pygam lightgbm xgboost
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

sys.path.insert(0, os.getcwd())
from architecture.data_utils import *
from architecture.pnet_config import *
from architecture.pipeline import *
from architecture.evaluation import *
from architecture.callbacks_custom import step_decay, FixedEarlyStopping
from architecture.feature_selection import SubsetSelectorWrapper

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
run_dir = f"{wd}/runs_classification/histone_dropna/"

if not os.path.exists(download_dir):
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)


# data_name = "rna_imputed"
# number_of_modules = "500"
# data_dir = f"{wd}/modules_experiment/gsva_scores"
# config["views"] = [("gsva_meth_imp", f"gsva_scores_{data_name}_{number_of_modules}.csv", None, 0 , lambda x:x, lambda x:x)]

# run_dir = f"{wd}/runs_classification/runs_modules/{data_name}/{number_of_modules}"

config["views"] = [("rna_imp", f"rna_imputed.csv", None, 0 , lambda x:x, lambda x:x)]

if not os.path.exists(download_dir):
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

config["data_dir"] = data_dir
config["run_dir"] = run_dir

config["view_alignment_method"] = "drop samples"
config["labels"] = [("cleveland_auc_binary.csv", 0)]
config["tv_split_seed"] = 42
config["inner_kfolds"] = 5
config["outer_kfolds"] = 5
config["use_validation_on_test"] = False

config["val_metric"] = lambda x: accuracy_score(x["val_df"].ys, (x["val_preds"] >= 0.5).astype(int))

config["results_processors"] = [
    lambda x: save_results(
        x,
        save_supervised_result,
        {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score
        },
        "binary"
    )
]


# # Run SVM
# svm_params = {
#     "model_params": {
#         f"poly_deg{d}_C{c}": {"kernel": "poly", "degree": d, "C": c, "probability": True}
#         for d in [2, 3] for c in [0.1, 0.5, 1, 5, 10]
#     } | {
#         f"linear_C{c}": {"kernel": "linear", "C": c, "probability": True}
#         for c in [0.01, 0.1, 0.5, 1, 5, 10]
#     }
# }
# config["model"] = SVC
# config["run_id"] = "svm"
# config["task"] = "binary classification"
# config["grid_search"] = construct_gs_params(svm_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("svm")


# # Run SVM with multiple kernels
# svm_params = {
#     "model_params": {
#         # RBF kernel with varying gamma and C
#         **{
#             f"rbf_gamma{g}_C{c}": {"kernel": "rbf", "gamma": g, "C": c, "probability": True}
#             for g in [0.001, 0.01, 0.1, 1] for c in [0.1, 1, 10]
#         },
#         # Sigmoid kernel with varying C
#         **{
#             f"sigmoid_C{c}": {"kernel": "sigmoid", "C": c, "probability": True}
#             for c in [0.1, 1, 10]
#         },
#     }
# }

# # Set up config
# config["model"] = SVC
# config["run_id"] = "svm2"
# config["task"] = "binary classification"
# config["grid_search"] = construct_gs_params(svm_params)

# # Run
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("svm2")



# Run Logistic Regression
logreg_params = {
    "model_params": {
        f"l1_C{c}": {
            "penalty": "l1", "solver": "saga", "C": c, "max_iter": 5000
        }
        for c in [0.01, 0.1, 1, 10]
    }
}

config["model"] = LogisticRegression
config["run_id"] = "logreg"
config["task"] = "binary classification"
config["grid_search"] = construct_gs_params(logreg_params)
pipeline = MLPipeline(config)
pipeline.run_crossvalidation()
aggregate_results("logreg")

# # Run XGBoost
# xgb_params = {
#     "model_params": {
#         f"xgb_eta{eta}_depth{depth}": {
#             "learning_rate": eta,
#             "max_depth": depth,
#             "use_label_encoder": False,
#             "eval_metric": "logloss"
#         }
#         for eta in [0.01, 0.1]
#         for depth in [3, 5]
#     }
# }

# config["model"] = XGBClassifier
# config["run_id"] = "xgb"
# config["task"] = "bianry classification"
# config["grid_search"] = construct_gs_params(xgb_params)

# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("xgb")

