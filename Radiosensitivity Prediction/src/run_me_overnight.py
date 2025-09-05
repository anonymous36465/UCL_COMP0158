import os, sys, json, time
import pandas as pd
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from keras.activations import linear, relu, tanh, leaky_relu
from keras.losses import MeanSquaredError
from functools import partial
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

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
from architecture.feature_selection import SubsetSelectorWrapper, SelectiveReducerWrapper
from architecture.mid_fusion_model import MLP_mid_fusion
from architecture.feature_scaling import StandardScalerProcessor
from architecture.pipeline import IdentityProcessor
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

def nll(y_true, y_pred, sigma=0.1):
    """
    Gaussian Negative Log Likelihood with fixed sigma
    """
    n = len(y_true)
    const_term = 0.5 * np.log(2 * np.pi * sigma**2)
    error_term = 0.5 * ((y_true - y_pred) ** 2) / (sigma**2)
    return np.mean(const_term + error_term)

### MLP Bayesian -- One modality only!
list_of_experiments = [
    ["methylation", "proteomics"],
    ["methylation","rna"],
    ["rna", "proteomics"],
    ["methylation", "rna", "proteomics"],
]

dataset_alias = {"methylation": "Meth", "rna": "Rna", "proteomics" : "Prot"}

for list_of_data in list_of_experiments:
    # Download data if not done so already and set up run directory
    wd = "Radiosensitivity Prediction"
    download_dir = f"{wd}/data"
    data_dir = f"{download_dir}/Cleveland"
    run_dir = f"{wd}/runs/runs_modules/"

    views = []
    run_dir_name = ""
    for dataset in list_of_data:
        views.append(((f"gsva_{dataset}_500", f"gsva_scores_{dataset}_imputed_500.csv", None, 0, lambda x : x, lambda x : x)))
        run_dir_name += dataset_alias[dataset]

    config["views"] = views
    run_dir += run_dir_name

    data_dir = f"{wd}/modules_experiment/gsva_scores"
    dataset_names = [dataset_alias[dataset] for dataset in list_of_data]
    dataset_sizes = [500]*len(list_of_data)
    architecture_search = True
    use_scheduler = True

    if architecture_search:
        run_dir += "/architecture_search/"

    if not os.path.exists(download_dir):
        with open(f"{wd}/src/download_data.py") as file:
            exec(file.read())

    if not os.path.exists(run_dir):
        os.mkdir(run_dir)


    config["data_dir"] = data_dir
    config["run_dir"] = run_dir

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
                                                                                        "mae" : mean_absolute_error, 
                                                                                        "nll_0.2": lambda x,y: nll(x, y, 0.2)},
                                                                            "individual")]
    config["feature_selector"] = SubsetSelectorWrapper(selector=VarianceThreshold, params={"threshold": 0.00001})
    # EXTRA:
    # config["val_metric"] = lambda x: nll(x["val_df"].ys, x["val_preds"], 0.2)

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
    #             "predict_variance": False,
    #             "weight_decay": wd,
    #             "scheduler": "plateau",
    #             "sched_factor": 0.5,
    #             "sched_patience": 2,
    #             "early_stopping_patience": 5,
    #             "default_y_sem": 0.2
    #         }
    #         for feat_layers in [2, 3]
    #         for feat_hidden in [16, 32, 50]
    #         for comb in [[1]]
    #         for lr in [1e-3]
    #         for drop in [0.1]
    #         for wd in [0.2]
    #     }
    # }

    # config["model"] = BayesianMLP_MidFusion
    # config["run_id"] = "bayesian_mlp_mid_scheduled"
    # config["task"] = "regression"
    # config["grid_search"] = construct_gs_params(gs_params)

    # pipeline = MLPipeline(config)
    # pipeline.run_crossvalidation()
    # aggregate_results("bayesian_mlp_mid_scheduled", print_summary=True)


    gs_params = {
        "model_params": {
            f"SINGLE-featL{feat_layers}-featH{feat_hidden}-comb{comb}-lr{lr}-drop{drop}-wd{wd}": {
                "datasets": dataset_names,
                "head_input_dims": dataset_sizes,
                "feature_layers": feat_layers,
                "feature_hidden_dim": feat_hidden,
                "combining_layers": comb,
                "num_epochs": 15,
                "learning_rate": lr,
                "dropout_prob": drop,
                "predict_variance": False,
                "weight_decay": wd,
                "scheduler": "plateau",
                "sched_factor": 0.5,
                "sched_patience": 2,
                "early_stopping_patience": 5,
                "default_y_sem": 0.2
            }
            for feat_layers in [2, 3]
            for feat_hidden in [16, 32, 50]
            # for comb in [[1]]
            # for comb in [[32, 8, 1], [64, 16, 1], [128, 32, 1]]
            for comb in [[128, 32, 1]]
            for lr in [1e-3]
            for drop in [0.1]
            for wd in [0.5]
        }
    }

    config["model"] = BayesianMLP_MidFusion
    config["run_id"] = "bayesian_mlp_mid_scheduled4"
    config["task"] = "regression"
    config["grid_search"] = construct_gs_params(gs_params)

    pipeline = MLPipeline(config)
    pipeline.run_crossvalidation()
    aggregate_results("bayesian_mlp_mid_scheduled4", print_summary=True)

    # gs_params = {
    #     "model_params": {
    #         f"SINGLE-featL{feat_layers}-featH{feat_hidden}-comb{comb}-lr{lr}-drop{drop}-wd{wd}": {
    #             "datasets": dataset_names,1e-3
    #             "head_input_dims": dataset_sizes,
    #             "feature_layers": feat_layers,
    #             "feature_hidden_dim": feat_hidden,
    #             "combining_layers": comb,
    #             "num_epochs": 100,
    #             "learning_rate": lr,
    #             "dropout_prob": drop,
    #             "predict_variance": False,
    #             "weight_decay": wd,
    #             "scheduler": "plateau",
    #             "sched_factor": 0.5,
    #             "sched_patience": 3,
    #             "early_stopping_patience": 7,
    #             "default_y_sem": 0.2
    #         }
    #         for feat_layers in [2, 3]
    #         for feat_hidden in [16, 32, 50]
    #         for comb in [[1]]
    #         for lr in [1e-3]
    #         for drop in [0.1]
    #         for wd in [0.2]
    #     }
    # }

    # config["model"] = BayesianMLP_MidFusion
    # config["run_id"] = "bayesian_mlp_mid_scheduled2"
    # config["task"] = "regression"
    # config["grid_search"] = construct_gs_params(gs_params)

    # pipeline = MLPipeline(config)
    # pipeline.run_crossvalidation()
    # aggregate_results("bayesian_mlp_mid_scheduled2", print_summary=True)

# list_of_experiments = [
#     ("rna_imputed", "all"),
#     ("methylation_imputed", "500"),
#     ("methylation_imputed", "most-enriched"),
#     ("methylation_imputed", "all"),
#     ("proteomics_imputed", "500"),
#     ("proteomics_imputed", "most-enriched"),
#     ("proteomics_imputed", "all")
# ]

# for data_name, number_of_modules in list_of_experiments:
#     wd = "Radiosensitivity Prediction"
#     download_dir = f"{wd}/data"
#     data_dir = f"{download_dir}/Imputed"

#     data_dir = f"{wd}/modules_experiment/gsva_scores"
#     config["views"] = [("gsva_meth_imp", f"gsva_scores_{data_name}_{number_of_modules}.csv", None, 0 , lambda x:x, lambda x:x)]

#     # run_dir = f"{wd}/runs/runs_separated/{data_name}"
#     run_dir = f"{wd}/runs/runs_modules/{data_name}/{number_of_modules}"

#     if not os.path.exists(download_dir):
#         with open(f"{wd}/src/download_data.py") as file:
#             exec(file.read())

#     if not os.path.exists(run_dir):
#         os.mkdir(run_dir)

#     config["data_dir"] = data_dir
#     config["run_dir"] = run_dir
#     config["run_id"] = "pnet"

#     config["view_alignment_method"] = "drop samples"
#     config["labels"] = [("cleveland_auc_full.csv", 0)]
#     config["tv_split_seed"] = 42
#     config["inner_kfolds"] = 5
#     config["outer_kfolds"] = 5
#     # config["test_samples"] = 0.1
#     config["use_validation_on_test"] = False
#     config["val_metric"] = lambda x : r2_score(x["val_df"].ys, x["val_preds"])
#     config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"r2" : r2_score,
#                                                                                         "explained_variance" : explained_variance_score,
#                                                                                         "mse" : mean_squared_error,
#                                                                                         "mae" : mean_absolute_error}, 
#                                                                             "individual")]
#     #                                                                           plot_history]


#     # # Run simple 2layer MLP
#     gs_params = {
#         "model_params": {
#             f"({hidden1},{hidden2},)_alpha_{a}": {
#                 "hidden_layer_sizes": (hidden1, hidden2,),
#                 "alpha": a,
#                 "max_iter": 300,
#                 "batch_size": 32,
#                 # "early_stopping": True, #sets asside a validation set
#                 "random_state": 42,
#                 # "verbose": True,
#                 "n_iter_no_change":5
#             }
#             for hidden1 in [64, 32, 16] 
#             for hidden2 in [32, 16, 8]
#             for a in [0.1, 1.0, 5.0, 10.0]
#         }
#     }
#     config["model"] = MLPRegressor
#     config["run_id"] = "mlp2"
#     config["task"] = "regression"
#     config["results_processors"] = config["results_processors"]  # optional
#     config["grid_search"] = construct_gs_params(gs_params)

#     pipeline = MLPipeline(config)
#     pipeline.run_crossvalidation()
#     aggregate_results("mlp2")


# list_of_experiments = [
#     #  ("histone_imputed_2025_fillna", "runs_histone_imputed_2025"),
#     ("proteomics_imputed", "runs_proteomics"),
#     ("rna_imputed", "runs_rna_imputed"),
#     ("methylation_imputed", "runs_methylation/imputed"),
   # ]

# for (data_name, run_subfolder) in list_of_experiments:
#     wd = "Radiosensitivity Prediction"
#     download_dir = f"{wd}/data"
#     data_dir = f"{download_dir}/Imputed"

#     config["views"] = [(data_name, f"{data_name}.csv", None, 0 , lambda x:x, lambda x:x)]

#     # run_dir = f"{wd}/runs/runs_separated/{data_name}"
#     run_dir = f"{wd}/runs/{run_subfolder}"

#     if not os.path.exists(download_dir):
#         with open(f"{wd}/src/download_data.py") as file:
#             exec(file.read())

#     if not os.path.exists(run_dir):
#         os.mkdir(run_dir)

#     config["data_dir"] = data_dir
#     config["run_dir"] = run_dir
#     config["run_id"] = "pnet"

#     config["view_alignment_method"] = "drop samples"
#     config["labels"] = [("cleveland_auc_full.csv", 0)]
#     config["tv_split_seed"] = 42
#     config["inner_kfolds"] = 5
#     config["outer_kfolds"] = 5
#     # config["test_samples"] = 0.1
#     config["use_validation_on_test"] = False
#     config["val_metric"] = lambda x : r2_score(x["val_df"].ys, x["val_preds"])
#     config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"r2" : r2_score,
#                                                                                         "explained_variance" : explained_variance_score,
#                                                                                         "mse" : mean_squared_error,
#                                                                                         "mae" : mean_absolute_error}, 
#                                                                             "individual")]
#     #                                                                           plot_history]

#     # # Run simple 2layer MLP
#     gs_params = {
#         "model_params": {
#             f"({hidden1},{hidden2},)_alpha_{a}": {
#                 "hidden_layer_sizes": (hidden1, hidden2,),
#                 "alpha": a,
#                 "max_iter": 300,
#                 "batch_size": 32,
#                 # "early_stopping": True, #sets asside a validation set
#                 "random_state": 42,
#                 # "verbose": True,
#                 "n_iter_no_change":5
#             }
#             for hidden1 in [64, 32, 16] 
#             for hidden2 in [32, 16, 8]
#             for a in [0.1, 1.0, 5.0, 10.0]
#         }
#     }
#     config["model"] = MLPRegressor
#     config["run_id"] = "mlp2"
#     config["task"] = "regression"
#     config["results_processors"] = config["results_processors"]  # optional
#     config["grid_search"] = construct_gs_params(gs_params)

#     pipeline = MLPipeline(config)
#     pipeline.run_crossvalidation()
#     aggregate_results("mlp2")




# datasets: rna477 (done), rna477(modified), rna+histone concat (done), rna(modified)+histone 
# modifications of rna: none, truncated50, variancethreshold0.1, selectkbest

list_of_experiments = [
    # ("histone_modification_data_process_na", "Cleveland", "runs_histone/histone_and_gexpr/concat"),
    # ("histone_imputed_2025_fillna", "Imputed", "runs_histone_imputed_2025/fillna/concat_gexpr")
    # ("rna_imputed_477samples_histone_processed_na", "Cleveland", "runs_histone/histone_and_gexpr/only_gexpr"),
    

    # ("setup1", "rna_imputed_477samples_histone_processed_na", "Cleveland", "runs_histone/histone_and_gexpr/only_gexpr/TruncatedSVD_50"),
    # ("setup2", "rna_imputed_477samples_histone_processed_na", "Cleveland", "runs_histone/histone_and_gexpr/only_gexpr/VarianceThreshold_0.1"),

    # ("setup1", "rna_imputed_432samples_histone_processed_na", "Cleveland", "runs_histone/histone_and_gexpr/only_gexpr/TruncatedSVD_50"),
    # ("setup2", "rna_imputed_432samples_histone_processed_na", "Cleveland", "runs_histone/histone_and_gexpr/only_gexpr/VarianceThreshold_0.1"),
    
    # ("setup3", "rna_imputed_and_histone_processed_na", "Cleveland", "runs_histone/histone_and_gexpr/TruncatedSVD_50_new"),
    # ("setup4", "rna_imputed_and_histone_processed_na", "Cleveland", "runs_histone/histone_and_gexpr/VarianceThreshold_0.1_new"),

    # ("setup5", "meth_imputed_and_histone_processed_na", "Cleveland", "runs_histone/histone_and_meth/TruncatedSVD_50_new"),
    # ("setup6", "meth_imputed_and_histone_processed_na", "Cleveland", "runs_histone/histone_and_meth/VarianceThreshold_0.1_new"),

    # ("setup1", "meth_imputed_432samples_histone_processed_na", "Cleveland", "runs_histone/histone_and_meth/only_meth/TruncatedSVD_50"),
    # ("no_setup", "meth_imputed_432samples_histone_processed_na", "Cleveland", "runs_histone/histone_and_meth/only_meth"),

    # ("concat", "histone_modification_data_process_na", "Cleveland", "runs_histone/histone_and_meth/concat"),

    # ("no_setup", "rna_imputed_432samples_histone_processed_na","Cleveland", "runs_histone/histone_and_gexpr/only_gexpr")

]

# for setup, data_name, data_folder, run_subdir in list_of_experiments:
#     # histone concat with RNA
#     wd = "Radiosensitivity Prediction"
#     download_dir = f"{wd}/data"
#     data_dir = f"{download_dir}/{data_folder}"

#     run_dir = f"{wd}/runs/{run_subdir}"

#     config["views"] = [
#         ("gexpr_and_histone", f"{data_name}.csv", None, 0, lambda x:x, lambda x:x),
#         # ("rna", f"rna_imputed.csv", None, 0,  lambda x:x, lambda x:x)
#         # ("meth", f"methylation_imputed.csv", None, 0,  lambda x:x, lambda x:x)
#     ]

#     if setup == "setup1":
#         config["feature_selector"] = SubsetSelectorWrapper(selector=TruncatedSVD, params={"n_components": 50, "random_state": 42})
#     elif setup == "setup2":
#         config["feature_selector"] = SubsetSelectorWrapper(selector=VarianceThreshold, params={"threshold": 0.1})
#     elif setup == "setup3":
#         config["feature_selector"] = SelectiveReducerWrapper(
#             reducer = TruncatedSVD, params={"n_components": 50, "random_state": 42},
#             dataset_name="gexpr_and_histone", featurename_to_reduce="gexpr")
#     elif setup == "setup4":
#         config["feature_selector"] = SelectiveReducerWrapper(
#             reducer = VarianceThreshold, params={"threshold": 0.1}, 
#             dataset_name="gexpr_and_histone", featurename_to_reduce="gexpr")
#     elif setup == "setup5":
#         config["feature_selector"] = SelectiveReducerWrapper(
#             reducer = TruncatedSVD, params={"n_components": 50, "random_state": 42},
#             dataset_name="gexpr_and_histone", featurename_to_reduce="meth")
#     elif setup == "setup6":
#         config["feature_selector"] = SelectiveReducerWrapper(
#             reducer = VarianceThreshold, params={"threshold": 0.1}, 
#             dataset_name="gexpr_and_histone", featurename_to_reduce="meth")
#     elif setup == "concat":
#         config["feature_selector"] = SubsetSelectorWrapper(selector=VarianceThreshold, params={"threshold": 0.0001})
#     else:
#         config["feature_selector"] = IdentityProcessor()



#     if not os.path.exists(download_dir):
#         with open(f"{wd}/src/download_data.py") as file:
#             exec(file.read())

#     if not os.path.exists(run_dir):
#         os.mkdir(run_dir)

#     config["data_dir"] = data_dir
#     config["run_dir"] = run_dir

#     config["view_alignment_method"] = "drop samples"
#     config["labels"] = [("cleveland_auc_full.csv", 0)]
#     config["tv_split_seed"] = 42
#     config["inner_kfolds"] = 5
#     config["outer_kfolds"] = 5
#     # config["test_samples"] = 0.1
#     config["use_validation_on_test"] = False
#     config["val_metric"] = lambda x : r2_score(x["val_df"].ys, x["val_preds"])
#     config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"r2" : r2_score,
#                                                                                         "explained_variance" : explained_variance_score,
#                                                                                         "mse" : mean_squared_error,
#                                                                                         "mae" : mean_absolute_error}, 
#                                                                             "individual")]
#     #                                                                           plot_history]

#     # Run Kernel Regression
#     gs_params = {"model_params" : {f"degree_{d}_alpha_{a}" : {"kernel" : "poly", "degree" : d, "alpha" : a}
#                                 for d in [1, 2, 3] for a in [0.01, 0.1, 0.5, 1, 3, 5, 10, 50]}}
#     config["model"] = KernelRidge
#     config["run_id"] = "krr"
#     config["task"] = "regression"
#     config["results_processors"] = config["results_processors"]
#     config["grid_search"] = construct_gs_params(gs_params)
#     pipeline = MLPipeline(config)
#     pipeline.run_crossvalidation()
#     aggregate_results("krr")


#     # Run ElasticNet (L1 and L2 combined)
#     gs_params = {"model_params" : {f"alpha_{a}_ratio_{r}" : {"alpha": a, "l1_ratio": r}
#                                     # for a in [0.05, 0.1, 0.3, 0.5] for r in [0.1, 0.15, 0.2, 0.5, 0.7, 0.8, 0.9]}}
#                                     # for a in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65] for r in [0.01, 0.05, 0.075, 0.1, 0.15]}}
#                                     for a in [0.001, 0.05, 0.1, 0.3, 0.5, 1.0] for r in [0.0, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0]}}

#     config["model"] = ElasticNet
#     config["run_id"] = "enet"
#     config["task"] = "regression"
#     config["results_processors"] = config["results_processors"]
#     config["grid_search"] = construct_gs_params(gs_params)
#     pipeline = MLPipeline(config)
#     pipeline.run_crossvalidation()
#     aggregate_results("enet")

#     # Run Lasso Regression
#     gs_params = {"model_params" : {f"alpha_{a}": {"alpha" : a}
#                                     for a in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]}}
#     config["model"] = Lasso
#     config["run_id"] = "lasso"
#     config["task"] = "regression"
#     config["results_processors"] = config["results_processors"]
#     config["grid_search"] = construct_gs_params(gs_params)
#     pipeline = MLPipeline(config)
#     pipeline.run_crossvalidation()
#     aggregate_results("lasso")

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


# OTHER THINGS i MIGHT WANT TO RUN
# * early fusion of methylation and histone (done)
# * (early fusion of modules)

list_of_experiments = [
    # ['proteomics', 'histone'],
    # ['methylation', 'histone'],
    # ['rna', 'histone'],

    # ["methylation", "rna"],
    ["methylation", "proteomics"],
    ["rna", "proteomics"],
    ["methylation", "rna", "proteomics"],

    # ["methylation", "rna", "histone"],
    # ["methylation", "proteomics", "histone"],
    # ["rna", "proteomics", "histone"],               #needs: xgboost
    # ["rna", "methylation", "proteomics", "histone"], #needs: xgboost
]

dataset_alias = {"methylation": "Meth", "rna": "Rna", "proteomics" : "Prot", "histone": "Hist"}

# for list_of_data in list_of_experiments:
#     # histone concat with RNA
#     wd = "Radiosensitivity Prediction"
#     download_dir = f"{wd}/data"
#     data_dir = f"{wd}/modules_experiment/gsva_scores"

#     run_dir = f"{wd}/runs/runs_modules/"

#     views = []
#     run_dir_name = ""
#     for dataset in list_of_data:
#         if dataset == "histone":
#             views.append(("hisotne", "histone_imputed_2025_fillna.csv", None, 0, lambda x:x, lambda x:x))
#             run_dir_name += "Hist"
#         else:
#             views.append(((f"gsva_{dataset}_500", f"gsva_scores_{dataset}_imputed_500.csv", None, 0, lambda x : x, lambda x : x)))
#             run_dir_name += dataset_alias[dataset]

#     config["views"] = views
#     run_dir += run_dir_name
#     run_dir += "/standardScaler"

#     if not os.path.exists(download_dir):
#         with open(f"{wd}/src/download_data.py") as file:
#             exec(file.read())

#     if not os.path.exists(run_dir):
#         os.mkdir(run_dir)

#     config["feature_selector"] = SubsetSelectorWrapper(selector=VarianceThreshold, params={"threshold": 0.0001})
#     config["feature_preprocessor"] = StandardScalerProcessor()

#     config["data_dir"] = data_dir
#     config["run_dir"] = run_dir

#     config["view_alignment_method"] = "drop samples"
#     config["labels"] = [("cleveland_auc_full.csv", 0)]
#     config["tv_split_seed"] = 42
#     config["inner_kfolds"] = 5
#     config["outer_kfolds"] = 5
#     # config["test_samples"] = 0.1
#     config["use_validation_on_test"] = False
#     config["val_metric"] = lambda x : r2_score(x["val_df"].ys, x["val_preds"])
#     config["results_processors"] = [lambda x : save_results(x, save_supervised_result, {"r2" : r2_score,
#                                                                                         "explained_variance" : explained_variance_score,
#                                                                                         "mse" : mean_squared_error,
#                                                                                         "mae" : mean_absolute_error}, 
#                                                                             "individual")]
#     #                                                                           plot_history]

#     # Run Kernel Regression
#     gs_params = {"model_params" : {f"degree_{d}_alpha_{a}" : {"kernel" : "poly", "degree" : d, "alpha" : a}
#                                 for d in [1, 2, 3] for a in [0.01, 0.1, 0.5, 1, 3, 5, 10, 50]}}
#     config["model"] = KernelRidge
#     config["run_id"] = "krr"
#     config["task"] = "regression"
#     config["results_processors"] = config["results_processors"]
#     config["grid_search"] = construct_gs_params(gs_params)
#     pipeline = MLPipeline(config)
#     pipeline.run_crossvalidation()
#     aggregate_results("krr")


#     # Run ElasticNet (L1 and L2 combined)
#     gs_params = {"model_params" : {f"alpha_{a}_ratio_{r}" : {"alpha": a, "l1_ratio": r}
#                                     # for a in [0.05, 0.1, 0.3, 0.5] for r in [0.1, 0.15, 0.2, 0.5, 0.7, 0.8, 0.9]}}
#                                     # for a in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65] for r in [0.01, 0.05, 0.075, 0.1, 0.15]}}
#                                     for a in [0.001, 0.05, 0.1, 0.3, 0.5, 1.0] for r in [0.0, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0]}}

#     config["model"] = ElasticNet
#     config["run_id"] = "enet"
#     config["task"] = "regression"
#     config["results_processors"] = config["results_processors"]
#     config["grid_search"] = construct_gs_params(gs_params)
#     pipeline = MLPipeline(config)
#     pipeline.run_crossvalidation()
#     aggregate_results("enet")

#     # # Run Lasso Regression
#     # gs_params = {"model_params" : {f"alpha_{a}": {"alpha" : a}
#     #                                 for a in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]}}
#     # config["model"] = Lasso
#     # config["run_id"] = "lasso"
#     # config["task"] = "regression"
#     # config["results_processors"] = config["results_processors"]
#     # config["grid_search"] = construct_gs_params(gs_params)
#     # pipeline = MLPipeline(config)
#     # pipeline.run_crossvalidation()
#     # aggregate_results("lasso")

#     # # Run XGBoost
#     # gs_params = {
#     #     "model_params": {
#     #         f"depth_{d}_lr_{lr}_n_{n}": {
#     #             "max_depth": d,
#     #             "learning_rate": lr,
#     #             "n_estimators": n,
#     #             "subsample": 0.8,
#     #             "colsample_bytree": 0.8,
#     #             "objective": "reg:squarederror",
#     #             "random_state": 42,
#     #         }
#     #         for d in [3, 5]
#     #         for lr in [0.005, 0.01]
#     #         for n in [300, 500, 800]
#     #     }
#     # }

#     # config["model"] = XGBRegressor
#     # config["run_id"] = "xgb"
#     # config["task"] = "regression"
#     # config["results_processors"] = config["results_processors"]
#     # config["grid_search"] = construct_gs_params(gs_params)

#     # pipeline = MLPipeline(config)
#     # pipeline.run_crossvalidation()
#     # aggregate_results("xgb")

#     #     # # Run simple 2layer MLP
#     # gs_params = {
#     #     "model_params": {
#     #         f"({hidden1},{hidden2},)_alpha_{a}": {
#     #             "hidden_layer_sizes": (hidden1, hidden2,),
#     #             "alpha": a,
#     #             "max_iter": 300,
#     #             "batch_size": 32,
#     #             # "early_stopping": True, #sets asside a validation set
#     #             "random_state": 42,
#     #             # "verbose": True,
#     #             "n_iter_no_change":5
#     #         }
#     #         for hidden1 in [64, 32, 16] 
#     #         for hidden2 in [32, 16, 8]
#     #         for a in [0.1, 1.0, 5.0, 10.0]
#     #     }
#     # }
#     # config["model"] = MLPRegressor
#     # config["run_id"] = "mlp2"
#     # config["task"] = "regression"
#     # config["results_processors"] = config["results_processors"]  # optional
#     # config["grid_search"] = construct_gs_params(gs_params)

#     # pipeline = MLPipeline(config)
#     # pipeline.run_crossvalidation()
#     # aggregate_results("mlp2")