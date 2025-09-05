import os, sys, json, time
import pandas as pd
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.preprocessing import MinMaxScaler

from keras.activations import linear, relu, tanh, leaky_relu
from keras.losses import MeanSquaredError
from functools import partial
from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Lasso, ElasticNet
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
from architecture.mid_fusion_model import MLP_mid_fusion, MLP_mid_fusion_scheduled
from architecture.mid_fusion_bayesian_mlp import BayesianMLP_MidFusion


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
run_dir = f"{wd}/runs/runs_modules/RnaImp_Hist/"

if not os.path.exists(download_dir):
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

# prepare config
minmax_features = lambda df, cols=None: (lambda X,c:(X.__setitem__(c, MinMaxScaler().fit_transform(X[c])) or X))(df.copy(), cols or df.select_dtypes(include='number').columns)
h3k4_features = ['H3K4me0', 'H3K4me1', 'H3K4me2', 'H3K4ac1']


data_dir = f"{wd}/modules_experiment/gsva_scores"
config["views"] = [
                    ("gsva_rna_500", f"gsva_scores_rna_imputed_500.csv", None, 0, lambda x : x, lambda x : x),
                    ("gsva_meth_500", f"gsva_scores_methylation_imputed_500.csv", None, 0, lambda x : x, lambda x : x),
                    ("gsva_prot_500", f"gsva_scores_proteomics_imputed_500.csv", None, 0, lambda x : x, lambda x : x),
                    # ("histone_imp", f"histone_imputed_2025_fillna.csv", None, 0, lambda x : x, lambda x:x)
    ]
dataset_names = ["Rna", "Meth", "Prot"]
dataset_sizes = [500, 500, 500]
architecture_search = True
use_scheduler = True
# identity_head_indices = None if "HistH3k4" not in dataset_names else [(len(dataset_names) - 1)]

run_dir = f"{wd}/runs/runs_modules/{''.join(dataset_names)}/"
if architecture_search:
    run_dir += "architecture_search/"

if not os.path.exists(download_dir):
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

# config["feature_selector"] = SubsetSelectorWrapper(selector=VarianceThreshold, params={"threshold": 0.1})
# config["feature_selector"] = SubsetSelectorWrapper(selector=TruncatedSVD, params={"n_components": 50, "random_state": 42})

# config["feature_selector"] = SubsetSelectorWrapper(selector=SelectKBest, params={"score_func": f_regression, "k": 200}, requires_y = True)

config["feature_selector"] = SubsetSelectorWrapper(selector=VarianceThreshold, params={"threshold": 0.00001})

config["data_dir"] = data_dir
config["run_dir"] = run_dir
config["run_id"] = "pnet"

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
#                                                                           plot_history]

if not use_scheduler:
    # # Run mid-fusion MLP
    if architecture_search:
        gs_params = {
            "model_params": {
                f"featLayers_{feat_layers}_{feat_hidden_dim}_combLayer_{combining_layers}_epochs_{num_epochs}_lr_{lr}_dropout_{drop}": {
                    "datasets": dataset_names,
                    "head_input_dims":dataset_sizes,
                    "feature_layers": feat_layers,
                    "feature_hidden_dim": feat_hidden_dim,
                    "combining_layers": combining_layers,
                    "num_epochs": num_epochs,
                    "learning_rate": lr,
                    "dropout_prob":drop,
                    "identity_head_indices":identity_head_indices
                } for feat_layers in [2, 3] for feat_hidden_dim in [16, 32, 50]
                for combining_layers in [[32, 8, 1], [64, 16, 1], [128, 32, 1]]
                for num_epochs in [10]
                for lr in [1e-3]
                for drop in [0.0]
            }
        }
    else:
        gs_params = {
            "model_params": {
                f"featLayers_{feat_layers}_{feat_hidden_dim}_combLayer_{combining_layers}_epochs_{num_epochs}_lr_{lr}_dropout_{drop}": {
                    "datasets": dataset_names,
                    "head_input_dims":dataset_sizes,
                    "feature_layers": feat_layers,
                    "feature_hidden_dim": feat_hidden_dim,
                    "combining_layers": combining_layers,
                    "num_epochs": num_epochs,
                    "learning_rate": lr,
                    "dropout_prob":drop
                } for feat_layers in [3] for feat_hidden_dim in [16]
                for combining_layers in [[128, 32, 1]]
                for num_epochs in [10, 25, 50, 100]
                for lr in [1e-4, 5e-4, 1e-3]
                for drop in [0.0]
            }
        }

    config["model"] = MLP_mid_fusion
    config["run_id"] = "mlp_mid"
    config["task"] = "regression"
    config["results_processors"] = config["results_processors"]
    config["grid_search"] = construct_gs_params(gs_params)

    pipeline = MLPipeline(config)
    pipeline.run_crossvalidation()
    aggregate_results("mlp_mid", print_summary=True)

else:
    n_heads = len(dataset_names)
    all_heads = list(range(n_heads))

    active_head_indices = all_heads if len(dataset_names)>1 else [0]

    # gs_params = {
    #     "model_params": {
    #         f"SINGLE-featL{feat_layers}-featH{feat_hidden}-comb{comb}-lr{lr}-drop{drop}-wd{wd}": {
    #             "datasets": dataset_names,
    #             "head_input_dims": dataset_sizes,
    #             "feature_layers": feat_layers,
    #             "feature_hidden_dim": feat_hidden,
    #             "combining_layers": comb,
    #             "num_epochs": 100,
    #             "learning_rate": lr,
    #             "dropout_prob": drop,
    #             "identity_head_indices": [],
    #             "active_head_indices": active_head_indices,
    #             "weight_decay": wd,
    #             "scheduler": "plateau",
    #             "sched_factor": 0.5,
    #             "sched_patience": 2,
    #             "early_stopping_patience": 5,
    #         }
    #         for feat_layers in [2, 3]
    #         for feat_hidden in [16, 32, 50]
    #         for comb in [[32, 8, 1], [64, 16, 1], [128, 32, 1]]
    #         # for comb in [[1]]
    #         for lr in [1e-3]
    #         for drop in [0.0]
    #         for wd in [0.2]
    #     }
    # }

    gs_params2 = {
        "model_params": {
            f"SINGLE-featL{feat_layers}-featH{feat_hidden}-comb{comb}-lr{lr}-drop{drop}-wd{wd}": {
                "datasets": dataset_names,
                "head_input_dims": dataset_sizes,
                "feature_layers": feat_layers,
                "feature_hidden_dim": feat_hidden,
                "combining_layers": comb,
                "num_epochs": 50,
                "learning_rate": lr,
                "dropout_prob": drop,
                "identity_head_indices": [],
                "active_head_indices": active_head_indices,
                "weight_decay": wd,
                "scheduler": "plateau",
                "sched_factor": 0.5,
                "sched_patience": 3,
                "early_stopping_patience": 5,
            }
            for feat_layers in [3]
            # for feat_hidden in [16, 32]
            for feat_hidden in [16]
            for comb in [[64, 16, 1], [128, 32, 1]]
            # for comb in [[1]]
            for lr in [1e-3]
            for drop in [0.1]
            # for wd in [0.01, 0.2]
            for wd in [0.2]
        }
    }

    config["model"] = MLP_mid_fusion_scheduled
    config["run_id"] = "mlp_mid_scheduled2"
    config["task"] = "regression"
    config["grid_search"] = construct_gs_params(gs_params2)

    pipeline = MLPipeline(config)
    pipeline.run_crossvalidation()
    aggregate_results("mlp_mid_scheduled2", print_summary=True)






# ### MID FUSION MLP:
# gs_params = {"model_params" : {f"." : 

#     {"datasets": dataset_names,
#     "head_input_dims": dataset_sizes,
#     "feature_layers": 3,
#     "feature_hidden_dim": 32,
#     "combining_layers": (128, 16, 1),
#     "dropout_prob": 0.1,
#     "predict_variance": True,
#     "learning_rate": 1e-3,
#     "num_epochs": 15,
#     "batch_size": 32,
#     "mc_samples":30,
#     "default_y_sem": 0.1

#    }}}
# config["model"] = BayesianMLP_MidFusion
# config["run_id"] = "mlp_mid_bayesian"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("mlp_mid_bayesian", print_summary=True)



# # Run simple MLP

# gs_params = {
#     "model_params": {
#         f"{h}_alpha": {
#             "hidden_layer_sizes": h,
#             "max_iter": 10,
#             "learning_rate_init": 0.001,
#             "random_state": 42,
#         }
#         for h in [(8,), (8, 4,), (8,4,2,),(16, 16, ), (32, 32,), (64, 16,), (128, 16,)]
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