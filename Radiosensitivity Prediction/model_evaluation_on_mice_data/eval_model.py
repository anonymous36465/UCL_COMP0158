import pandas as pd
import pickle, sys, os

sys.path.insert(0, os.getcwd())
from architecture.data_utils import *
from architecture.pnet_config import *
from architecture.pipeline import *

from sklearn.kernel_ridge import KernelRidge

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error


################# Run Details
data_type = None #"gsva_scores"
train_data_name = "methylation_imputed"
model_name = "lasso"+ "_" + "mad_alpha_1.5_eps_1e3"

#########################

if data_type is not None:
    train_data_name = f'{data_type}_{train_data_name}'

wd = "Radiosensitivity Prediction"
experiment_dir = "Radiosensitivity Prediction/model_evaluation_on_mice_data"
download_dir = f"{wd}/data"

# Set train data dir
train_data_dir = f"{download_dir}/Cleveland"
if data_type == "gsva_scores":
    train_data_dir = f"{wd}/modules_experiment/gsva_scores"

# Set test (mice) data dir
test_data_dir = f'{experiment_dir}/data/mice'
if data_type == "gsva_scores":
    test_data_dir = f'{experiment_dir}/data/mice_gsva'

# Set train run_dir
run_dir = f"{wd}/model_evaluation_on_mice_data/runs/{train_data_name}"

# Load the trained model
model = pickle.load(open(f"{run_dir}/{model_name}.sav", 'rb'))
print(f" Loaded model of type {type(model)}")

# Load train&test dataset to align columns/features
train_df = pd.read_csv(f'{train_data_dir}/{train_data_name}.csv', index_col = 0)
train_features = train_df.columns
print(train_df.columns)

test_df = pd.read_csv(f'{test_data_dir}/mice_{train_data_name}.csv', index_col=0)
test_df = test_df[train_features]

results = model.predict(test_df)
print(results)

results_df = pd.DataFrame(results, index=test_df.index, columns=["prediction"])
results_df.index = results_df.index.str.replace('.', '-', regex=False)
output_dir = f"{experiment_dir}/predictions"
os.makedirs(output_dir, exist_ok=True)  # create dir if it doesn't exist
output_file = f"{output_dir}/{train_data_name}_{model_name}_predictions.csv"

results_df.to_csv(output_file)
print(f"Predictions saved to: {output_file}")

df = pd.read_csv(f"{download_dir}/Mice/PDX_response_to_standard_therapy.csv")
df['PDX_Line'] = ['Mayo-PDX-Sarkaria-'+str(val) for val in df['PDX_Line'].values]

label = 'RT_Ratio'

merged = results_df.merge(df[['PDX_Line', label]], left_index=True, right_on='PDX_Line')

# Scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(merged[label], merged["prediction"])
plt.xlabel("RT_Ratio (True)")
plt.ylabel("Model Prediction")
plt.title("Predicted vs Actual RT_Ratio")
plt.plot([merged[label].min(), merged[label].max()],
         [merged["prediction"].min(), merged["prediction"].max()],
         linestyle='--', color='gray')  # identity line

plt.grid(True)
plt.tight_layout()
plt.show()

# report the evaluation metrics
y_true, y_pred = merged[label], merged["prediction"]
print("R²:", r2_score(y_true, y_pred))
print("Explained Var:", explained_variance_score(y_true, y_pred))
print("MSE:", mean_squared_error(y_true, y_pred))
print("MAE:", mean_absolute_error(y_true, y_pred))