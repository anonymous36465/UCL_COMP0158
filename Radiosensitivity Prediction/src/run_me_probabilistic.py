import os, sys, json, time
import pandas as pd
import json
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, explained_variance_score
from keras.activations import linear, relu, tanh, leaky_relu
from keras.losses import MeanSquaredError
from functools import partial

sys.path.insert(0, os.getcwd())
from architecture.data_utils import *
from architecture.pnet_config import *
from architecture.pipeline import *
from architecture.evaluation import *
from architecture.callbacks_custom import step_decay, FixedEarlyStopping
from architecture.bayesian_regressor import BayesianRegressor
from architecture.mid_fusion_bayesian_mlp import BayesianMLP_MidFusion
from architecture.feature_selection import SubsetSelectorWrapper

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
# from sklearn.decomposition import TruncatedSVD

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RationalQuadratic
from sklearn.gaussian_process.kernels import RBF, ConstantKernel 



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
data_dir = f"{download_dir}/Imputed"
run_dir = f"{wd}/runs/runs_rna_imputed"

# select100best_withincv" - the one that used more data to select features.

if not os.path.exists(download_dir):
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

# with open(f"{download_dir}/qrcp10.txt", "r") as f:
#     selected_features_qr = [line.strip() for line in f]

selected_genes = list(set(pd.read_csv(f"{download_dir}/hugo_genes.txt", sep="\t")["symbol"]))

# selected_features = None

# prepare config


# Define views with placeholder preprocess/alignment funcs
config["views"] = [
    ("hist_imp_2025", f"rna_imputed.csv", None, 0, lambda x: x, lambda x: x),
]
# data_name = "rna_imputed"
# number_of_modules = "500"
# data_dir = f"{wd}/modules_experiment/gsva_scores"
# run_dir = f"{wd}/runs/runs_modules/{data_name}/{number_of_modules}"

# config["views"] = [("gsva_rna_imp", f"gsva_scores_{data_name}_{number_of_modules}.csv", None, 0 , lambda x:x, lambda x:x)]


if not os.path.exists(download_dir):
    with open(f"{wd}/src/download_data.py") as file:
        exec(file.read())

if not os.path.exists(run_dir):
    os.mkdir(run_dir)

config["data_dir"] = data_dir
config["run_dir"] = run_dir

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


# Run Gaussian Process Regressor
# n_restarts_optimizer because diff initializations give different minima

# gs_params = {"model_params" : {f"kernel_none_noise_{a}" : {"kernel" : None, "alpha" : a, "random_state": 0, "normalize_y": True, "n_restarts_optimizer": 5}
#                             for a in[0.01, 0.1, 0.5, 1]}}
# # gs_params = {"model_params" : {f"kernel_none_noise_" : {"kernel" : None, "random_state": 0, "normalize_y": True, "n_restarts_optimizer": 5}
# #                             }}
# config["model"] = GaussianProcessRegressor
# config["run_id"] = "gpr"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("gpr")

# # RBF
# kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
# gs_params = {"model_params" : {f"kernel_RBF_noise_{a}" : {"kernel" : kernel, "alpha" : a, "random_state": 0, "normalize_y": True, "n_restarts_optimizer": 5}
#                             for a in[0.01, 0.1, 0.5, 1]}}
# # gs_params = {"model_params" : {f"kernel_RBF_noise_" : {"kernel" : kernel, "random_state": 0, "normalize_y": True, "n_restarts_optimizer": 5}}}
# config["model"] = GaussianProcessRegressor
# config["run_id"] = "gpr_rbf"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("gpr_rbf")


# # DotProduct() + WhiteKernel(noise_level=0.5)
# kernel = DotProduct() + WhiteKernel(noise_level=0.5)
# gs_params = {"model_params" : {f"kernel_White_noise_" : {"kernel" : kernel,"random_state": 0, "normalize_y": True, "n_restarts_optimizer": 5}}}
# config["model"] = GaussianProcessRegressor
# config["run_id"] = "gpr_whitekernel"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("gpr_whitekernel")

# gs_params = {"model_params" : {f"kernel_Const_{a}_RBF_{l}_White_" : {
#     "kernel" : ConstantKernel(a) * RBF(length_scale=l) + WhiteKernel(),
#     "random_state": 0, "normalize_y": True, "n_restarts_optimizer": 2}
# for a in [0.1, 1.0, 3, 5, 6, 10] for l in [0.1,0.5,0.8,1,2,5,6]}}

# config["model"] = GaussianProcessRegressor
# config["run_id"] = "gpr_const_rbf_white"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("gpr_const_rbf_white")

# gs_params = {"model_params" : {f"kernel_RBF_{l}_White_" : {
#     "kernel" : RBF(length_scale=l) + WhiteKernel(),
#     "random_state": 0, "normalize_y": True, "n_restarts_optimizer": 2}
# for l in [0.1,0.5,0.8,1,2,5,6]}}

# config["model"] = GaussianProcessRegressor
# config["run_id"] = "gpr_rbf_white"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("gpr_rbf_white")

# gs_params = {"model_params" : {f"kernel_Const_{a}_RBF_{l}_White_{n}" : {
#     "kernel" : ConstantKernel(a) * RBF(length_scale=l) + WhiteKernel(n),
#     "random_state": 0, "normalize_y": True, "n_restarts_optimizer": 5}
# for a in [1.0] for l in [5] for n in [0.01, 0.1, 0.5, 1.0]}}

# config["model"] = GaussianProcessRegressor
# config["run_id"] = "gpr_const_rbf_white_2"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("gpr_const_rbf_white_2")


# # ARD RBF (rbf has feature-dependent parameters)
# kernel = RBF(length_scale=np.ones(42), length_scale_bounds=(1e-2, 1e3))
# gs_params = {"model_params" : {f"kernel_White_noise_" : {"kernel" : kernel,"random_state": 0, "normalize_y": True, "n_restarts_optimizer": 5}}}
# config["model"] = GaussianProcessRegressor
# config["run_id"] = "gpr_ard_rbf"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("gpr_ard_rbf")


# kernel = RBF() + RationalQuadratic() + WhiteKernel()
# gs_params = {"model_params" : {f"kernel_rbf_quadratic_white_" : 
#     {"kernel" : kernel,"random_state": 0, "normalize_y": True, "n_restarts_optimizer": 0}}}
# config["model"] = GaussianProcessRegressor
# config["run_id"] = "rbf_quad_white"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)Gaussian
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("rbf_quad_white")

# kernel = RBF() + RationalQuadratic()
# gs_params = {"model_params" : {f"kernel_rbf_quadratic_noise_{a}" : 
#     {"kernel" : kernel, 
#     "alpha" : a,
#     "random_state": 0, "normalize_y": True, "n_restarts_optimizer": 2}
#     for a in [0.01, 0.04, 0.08, 0.1]}}
# config["model"] = GaussianProcessRegressor
# config["run_id"] = "rbf_quad_noise"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("rbf_quad_noise")

# kernel = RBF() + RationalQuadratic()
# gs_params = {"model_params" : {f"kernel_rbf_quadratic_noise_0.1" : 
#     {"kernel" : kernel, 
#     "alpha" : a,
#     "random_state": 0, "normalize_y": True, "n_restarts_optimizer": 0}
#     for a in [0.1]}}
# config["model"] = GaussianProcessRegressor
# config["run_id"] = "rbf_quad_noise_0.1"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("rbf_quad_noise_0.1")

# kernel = RBF() + RationalQuadratic()
# gs_params = {"model_params" : {f"kernel_rbf_quadratic_noise_{a}" : 
#     {"kernel" : kernel, 
#     "alpha" : a,
#     "random_state": 0, "normalize_y": True, "n_restarts_optimizer": 2}
#     for a in [0.1]}}
# config["model"] = GaussianProcessRegressor
# config["run_id"] = "rbf_quad_noise_0.1"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("rbf_quad_noise_0.1")


# gs_params = {"model_params" : {f"kernel_rbf_{l}_quadratic_noise_{a}" : 
#     {"kernel" : RBF(l) + RationalQuadratic(), 
#     "alpha" : a,

#     "random_state": 0, "normalize_y": True, "n_restarts_optimizer": 2}
#     for a in [0.1]
#     for l in [0.5,1,2,5]}}
# config["model"] = GaussianProcessRegressor
# config["run_id"] = "rbf(param)_quad_noise_0.1"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("rbf(param)_quad_noise_0.1")


# ######### Bayesian Regressor
# gs_params = {"model_params" : {f"dims_(128, 32)" : 
#     {"input_dim": 500,
#     "dropout_prob": 0.1,
#     "hidden_dims": (128, 32),
#     "predict_variance": True,
#     "num_epochs": 20,
#     "default_y_sem": 0.1}}}
# config["model"] = BayesianRegressor
# config["run_id"] = "bayesian_mlp"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("bayesian_mlp")

######## Bayesian Regressor
# gs_params = {"model_params" : {f"dims_(32, 16, 4)" : 
#     {"input_dim": 500,
#     "dropout_prob": 0.1,
#     "hidden_dims": (32, 16, 4),
#     "predict_variance": True,
#     "num_epochs": 40,
#     "default_y_sem": 0.1}}}
# config["model"] = BayesianRegressor
# config["run_id"] = "bayesian_mlp_(32, 16, 4)"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("bayesian_mlp_(32, 16, 4)")

# gs_params = {"model_params" : {f"dims_(32, 4)" : 
#     {"input_dim": 500,
#     "dropout_prob": 0.1,
#     "hidden_dims": (32, 4),
#     "predict_variance": True,
#     "num_epochs": 30,
#     "default_y_sem": 0.1}}}
# config["model"] = BayesianRegressor
# config["run_id"] = "bayesian_mlp_(32, 4)"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("bayesian_mlp_(32, 4)")


# ### MID FUSION MLP:
# gs_params = {"model_params" : {f"dims_(32, 4)" : 

#     {"datasets": ["rna", "meth"]
#     "head_input_dims": [500, 500],
#     "feature_layers": 3,
#     "feature_hidden_dim": 32,
#     "combining_layers": (128, 32, 1),
#     "dropout_prob": 0.1,
#     "predict_variance": True,
#     "learning_rate": 1e-3,
#     "epochs": 20,
#     "batch_size": 32,
#     "mc_samples":30,
#     "default_y_sem": 0.1

#    }}}
# config["model"] = BayesianMLP_MidFusion
# config["run_id"] = "bayesian_mlp_mi"
# config["task"] = "regression"
# config["results_processors"] = config["results_processors"]
# config["grid_search"] = construct_gs_params(gs_params)
# pipeline = MLPipeline(config)
# pipeline.run_crossvalidation()
# aggregate_results("bayesian_mlp_(32, 4)")



# [datasets, dimentions, feat layers, feat hidden dim, combining layers, ]
list_of_experiments = [
    [["rna", "methylation"], 2, 32, [32, 8, 1]],
    [["rna", "proteomics"], 3, 16, [128, 32, 1]],
    [["methylation", "proteomics"], 3, 16, [128, 32, 1]]
]

dataset_alias = {"methylation": "Meth", "rna": "Rna", "proteomics" : "Prot", "histone": "Hist"}


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
# for datasets, fl, fhd, cl in list_of_experiments:
#     # histone concat with RNA
#     wd = "Radiosensitivity Prediction"
#     download_dir = f"{wd}/data"
#     data_dir = f"{wd}/modules_experiment/gsva_scores"

#     run_dir = f"{wd}/runs/runs_modules/"

#     views = []
#     run_dir_name = ""
#     for dataset in datasets:
#         views.append(((f"gsva_{dataset}_500", f"gsva_scores_{dataset}_imputed_500.csv", None, 0, lambda x : x, lambda x : x)))
#         run_dir_name += dataset_alias[dataset]

#     config["views"] = views
#     run_dir += run_dir_name

#     if not os.path.exists(download_dir):
#         with open(f"{wd}/src/download_data.py") as file:
#             exec(file.read())

#     if not os.path.exists(run_dir):
#         os.mkdir(run_dir)

#     config["feature_selector"] = SubsetSelectorWrapper(selector=VarianceThreshold, params={"threshold": 0.0001})

#     config["data_dir"] = data_dir
#     config["run_dir"] = run_dir

#     gs_params = {"model_params" : {f"dims_{fl}_{fhd}_{cl}_lr_{lr}_ep_{epochs}_noise_{noise}" : 

#         {"datasets": datasets,
#         "head_input_dims": [500]*len(datasets),
#         "feature_layers": fl,
#         "feature_hidden_dim": fhd,
#         "combining_layers": cl,
#         "dropout_prob": 0.1,
#         "predict_variance": False,
#         "learning_rate": 1e-3,
#         "num_epochs": epochs,
#         "batch_size": 32,
#         "mc_samples":30,
#         "default_y_sem": noise,
#         "weight_decay":0.001

#     } for lr in [1e-3] for epochs in [10, 15, 20] for noise in [0.1, 0.2]}}
#     config["model"] = BayesianMLP_MidFusion
#     config["run_id"] = "bayesian_mlp"
#     config["task"] = "regression"
#     config["results_processors"] = config["results_processors"]
#     config["grid_search"] = construct_gs_params(gs_params)
#     pipeline = MLPipeline(config)
#     pipeline.run_crossvalidation()
#     aggregate_results("bayesian_mlp")

# for datasets, fl, fhd, cl in list_of_experiments:
#     # histone concat with RNA
#     wd = "Radiosensitivity Prediction"
#     download_dir = f"{wd}/data"
#     data_dir = f"{wd}/modules_experiment/gsva_scores"

#     run_dir = f"{wd}/runs/runs_modules/"

#     views = []
#     run_dir_name = ""
#     for dataset in datasets:
#         views.append(((f"gsva_{dataset}_500", f"gsva_scores_{dataset}_imputed_500.csv", None, 0, lambda x : x, lambda x : x)))
#         run_dir_name += dataset_alias[dataset]

#     config["views"] = views
#     run_dir += run_dir_name

#     if not os.path.exists(download_dir):
#         with open(f"{wd}/src/download_data.py") as file:
#             exec(file.read())

#     if not os.path.exists(run_dir):
#         os.mkdir(run_dir)

#     config["feature_selector"] = SubsetSelectorWrapper(selector=VarianceThreshold, params={"threshold": 0.0001})

#     config["data_dir"] = data_dir
#     config["run_dir"] = run_dir

#     gs_params = {"model_params" : {f"dims_{fl}_{fhd}_{cl}_lr_{lr}_ep_{epochs}_noise_{noise}_wd_{weight_decay}" : 

#         {"datasets": datasets,
#         "head_input_dims": [500]*len(datasets),
#         "feature_layers": fl,
#         "feature_hidden_dim": fhd,
#         "combining_layers": cl,
#         "dropout_prob": 0.1,
#         "predict_variance": False,
#         "learning_rate": lr,
#         "num_epochs": epochs,
#         "batch_size": 32,
#         "mc_samples":30,
#         "default_y_sem": noise,
#         "weight_decay":weight_decay

#     } for lr in [1e-4] for epochs in [20, 40, 80] for noise in [0.1, 0.2] for weight_decay in [0.0, 0.01, 0.1]}}
#     config["model"] = BayesianMLP_MidFusion
#     config["run_id"] = "bayesian_mlp_lr1e-4"
#     config["task"] = "regression"
#     config["results_processors"] = config["results_processors"]
#     config["grid_search"] = construct_gs_params(gs_params)
#     pipeline = MLPipeline(config)
#     pipeline.run_crossvalidation()
#     aggregate_results("bayesian_mlp_lr1e-4")




# for datasets, fl, fhd, cl in list_of_experiments:
#     # histone concat with RNA
#     wd = "Radiosensitivity Prediction"
#     download_dir = f"{wd}/data"
#     data_dir = f"{wd}/modules_experiment/gsva_scores"

#     run_dir = f"{wd}/runs/runs_modules/"

#     views = []
#     run_dir_name = ""
#     for dataset in datasets:
#         views.append(((f"gsva_{dataset}_500", f"gsva_scores_{dataset}_imputed_500.csv", None, 0, lambda x : x, lambda x : x)))
#         run_dir_name += dataset_alias[dataset]

#     config["views"] = views
#     run_dir += run_dir_name

#     if not os.path.exists(download_dir):
#         with open(f"{wd}/src/download_data.py") as file:
#             exec(file.read())

#     if not os.path.exists(run_dir):
#         os.mkdir(run_dir)

#     config["feature_selector"] = SubsetSelectorWrapper(selector=VarianceThreshold, params={"threshold": 0.0001})

#     config["data_dir"] = data_dir
#     config["run_dir"] = run_dir

#     gs_params = {"model_params" : {f"noise_{noise}_wd_{weight_decay}" : 

#         {"datasets": datasets,
#         "head_input_dims": [500]*len(datasets),
#         "dropout_prob": 0.1,
#         "predict_variance": False,
#         # "learning_rate": lr,
#         # "num_epochs": epochs,
#         # "batch_size": 32,
#         # "mc_samples":30,
#         "default_y_sem": noise,
#         "weight_decay":weight_decay

#     }  for noise in [0.1] for weight_decay in [0.1, 0.5]}}
#     config["model"] = BayesianMLP_MidFusion
#     config["run_id"] = "bayesian_mlp_default"
#     config["task"] = "regression"
#     config["results_processors"] = config["results_processors"]
#     config["grid_search"] = construct_gs_params(gs_params)
#     pipeline = MLPipeline(config)
#     pipeline.run_crossvalidation()
#     aggregate_results("bayesian_mlp_default")


#  All have 3x16, (128, 32, 1)
list_of_experiments = [
    # [["rna", "methylation"], 3, 16, [128, 32, 1]],
    [["rna", "proteomics"], 3, 16, [128, 32, 1]],
    [["methylation", "proteomics"], 3, 16, [128, 32, 1]]
]

for datasets, fl, fhd, cl in list_of_experiments:
    # histone concat with RNA
    wd = "Radiosensitivity Prediction"
    download_dir = f"{wd}/data"
    data_dir = f"{wd}/modules_experiment/gsva_scores"

    run_dir = f"{wd}/runs/runs_modules/"

    views = []
    run_dir_name = ""
    for dataset in datasets:
        views.append(((f"gsva_{dataset}_500", f"gsva_scores_{dataset}_imputed_500.csv", None, 0, lambda x : x, lambda x : x)))
        run_dir_name += dataset_alias[dataset]

    config["views"] = views
    run_dir += run_dir_name

    if not os.path.exists(download_dir):
        with open(f"{wd}/src/download_data.py") as file:
            exec(file.read())

    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    config["feature_selector"] = SubsetSelectorWrapper(selector=VarianceThreshold, params={"threshold": 0.0001})

    config["data_dir"] = data_dir
    config["run_dir"] = run_dir

    gs_params = {"model_params" : {f"dims_{fl}_{fhd}_{cl}_lr_{lr}_ep_{epochs}_noise_{noise}_wd_{weight_decay}" : 

        {"datasets": datasets,
        "head_input_dims": [500]*len(datasets),
        "feature_layers": fl,
        "feature_hidden_dim": fhd,
        "combining_layers": cl,
        "dropout_prob": 0.1,
        "predict_variance": False,
        "learning_rate": lr,
        "num_epochs": epochs,
        "batch_size": 32,
        "mc_samples":30,
        "default_y_sem": noise,
        "weight_decay":weight_decay

    } for lr in [1e-4] for epochs in [100] for noise in [0.1, 0.2] for weight_decay in [0.0, 0.01, 0.1]}}
    config["model"] = BayesianMLP_MidFusion
    config["run_id"] = "bayesian_mlp_lr1e-4_2"
    config["task"] = "regression"
    config["results_processors"] = config["results_processors"]
    config["grid_search"] = construct_gs_params(gs_params)
    pipeline = MLPipeline(config)
    pipeline.run_crossvalidation()
    aggregate_results("bayesian_mlp_lr1e-4_2")