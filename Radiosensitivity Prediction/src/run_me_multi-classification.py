import os, sys, json, time
import pandas as pd
import json

from sklearn.metrics import r2_score, mean_squared_error
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
from sklearn.linear_model import LogisticRegression

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
run_dir = f"{wd}/runs_multiclass/rna_500/"

if not os.path.exists(download_dir):
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())


data_name = "rna_imputed"
number_of_modules = "all"
data_dir = f"{wd}/modules_experiment/gsva_scores"
config["views"] = [("gsva_meth_imp", f"gsva_scores_{data_name}_{number_of_modules}.csv", None, 0 , lambda x:x, lambda x:x)]

run_dir = f"{wd}/runs_multiclass/runs_modules/{data_name}/{number_of_modules}/separated"

# config["views"] = [("rna_imp", f"rna_imputed.csv", None, 0 , lambda x:x, lambda x:x)]

# class_weights = {0.0:0.9, 1.0:0.49270073, 2.0:1.35, 3.0:8.4375}
# class_weights2 = {0.0:1.0, 1.0:0.1, 2.0:10.0, 3.0:50.0}

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

if not os.path.exists(download_dir):
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

config["data_dir"] = data_dir
config["run_dir"] = run_dir

config["view_alignment_method"] = "drop samples"
config["labels"] = [("cleveland_auc_separated_2classes.csv", 0)]
config["tv_split_seed"] = 42
config["inner_kfolds"] = 5
config["outer_kfolds"] = 5
config["use_validation_on_test"] = False

config["val_metric"] = lambda x: accuracy_score(x["val_df"].ys, x["val_preds"])

config["results_processors"] = [
    lambda x: save_results(
        x,
        save_supervised_result,
        {
            "accuracy": accuracy_score,
            # "precision": precision_score,
            # "recall": recall_score,
            # "f1": f1_score,
            **{
                f"acc_class_{cls}": (
                    lambda y_true, y_pred, cls=cls:
                        ((y_true == cls) & (y_pred == cls)).sum()
                        / max((y_true == cls).sum(), 1)
                )
                for cls in [0.0, 1.0, 2.0, 3.0]
            }
        },
        "individual" #because there is no need for binning
    )
]

# # Run Logistic Regression
# logreg_params = {
#     "model_params": {
#         f"l1_C{c}": {
#             "multi_class":"multinomial",
#             "penalty": "l1", 
#             "solver": "saga", 
#             "C": c, 
#             "max_iter": 5000,
#             "class_weight": "balanced"
#         }
#         # for c in [0.01, 0.1, 1, 10]
#         for c in [0.1, 1]
#     }
# }

# config["model"] = LogisticRegression
# config["run_id"] = "logreg_balanced"
# config["task"] = "multiclass"
# config["weight_samples"] = False

# config["grid_search"] = construct_gs_params(logreg_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("logreg_balanced")
# evaluate_predictions_like_regression("logreg_balanced", run_dir, data_dir)



# Run SVC
for balanced in [False, True]:
    if balanced:
        class_weight = "balanced"
        suffix = "_balanced"
    else:
        class_weight = None
        suffix = ""

    # Run Logistic Regression
    logreg_params = {
        "model_params": {
            f"l1_C{c}": {
                "multi_class":"multinomial",
                "penalty": "l1", 
                "solver": "saga", 
                "C": c, 
                "max_iter": 5000,
                "class_weight": class_weight
            }
            # for c in [0.01, 0.1, 1, 10]
            for c in [0.1, 1]
        }
    }

    config["model"] = LogisticRegression
    config["run_id"] = "logreg"+suffix
    config["task"] = "multiclass"
    config["weight_samples"] = "None"

    config["grid_search"] = construct_gs_params(logreg_params)
    pipeline = MLPipeline(config)
    pipeline.run_crossvalidation()
    aggregate_results("logreg"+suffix)
    evaluate_predictions_like_regression("logreg"+suffix, run_dir, data_dir)

    # Run SVC
    svm_params = {
        "model_params": {
            f"poly_deg{d}_C{c}": {"kernel": "poly", "degree": d, "C": c, "probability": True, 
            "class_weight":class_weight
            }
            for d in [2, 3] for c in [0.1, 0.5, 1, 5, 10]
        } | {
            f"linear_C{c}": {"kernel": "linear", "C": c, "probability": True, 
            "class_weight":class_weight
            }
            for c in [0.01, 0.1, 0.5, 1, 5, 10]
        }
    }
    config["model"] = SVC
    config["run_id"] = "svm"+suffix
    config["task"] = "multiclass"
    config["weight_samples"] = "None"

    config["grid_search"] = construct_gs_params(svm_params)
    pipeline = MLPipeline(config)
    pipeline.run_crossvalidation()
    aggregate_results("svm"+suffix)
    evaluate_predictions_like_regression("svm"+suffix, run_dir, data_dir)


    # # # Run XGBoost
    # xgb_params = {
    #     "model_params": {
    #         f"xgb_eta{eta}_depth{depth}": {
    #             "learning_rate": eta,
    #             "max_depth": depth,
    #             "n_estimators": 500,
    #             "subsample": 0.8,
    #             "colsample_bytree": 0.8,
    #             # "objective": "multi:softprob",  # probs over 4 classes
    #             "num_class": 2,
    #             "eval_metric": "mlogloss",      # (optionally add "merror")
    #             "tree_method": "hist"
    #         }
    #         for eta in [0.05]
    #         for depth in [4]
    #     }
    # }

    # config["model"] = XGBClassifier
    # config["run_id"] = "xgb"+suffix
    # config["task"] = "multiclass"
    # config["weight_samples"] = "Discrete" if balanced else "None"

    # config["grid_search"] = construct_gs_params(xgb_params)

    # pipeline = MLPipeline(config)
    # pipeline.run_crossvalidation()
    # aggregate_results("xgb"+suffix)

    # evaluate_predictions_like_regression("xgb"+suffix, run_dir, data_dir)




# # Run Logistic Regression
# logreg_params = {
#     "model_params": {
#         f"l1_C{c}": {
#             "penalty": "l1", "solver": "saga", "C": c, "max_iter": 5000
#         }
#         for c in [0.01, 0.1, 1, 10]
#     }
# }

# config["model"] = LogisticRegression
# config["run_id"] = "logreg"
# config["task"] = "binary classification"
# config["grid_search"] = construct_gs_params(logreg_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("logreg")






# # Random Forest - classical
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=500)


# support vector classifier
