import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import mean_squared_error, roc_auc_score

##### RNA
all_data_name = "rna_imputed"
all_model = "krr"

# your experiments
list_of_experiments = [
    (all_data_name, all_model, None, "raw values"),
    (all_data_name, all_model, "mad_alpha_2.0_eps_1e3", "weighted"),
    ("gsva_scores_"+all_data_name+"_500", all_model, None, "500 modules"),
]

#### METHYLATION
# all_data_name = "methylation_imputed"
# all_model = "lasso"

# # your experiments
# list_of_experiments = [
#     (all_data_name, all_model, None, "raw values"),
#     (all_data_name, all_model, "mad_alpha_2.0_eps_1e3", "weighted"),
#     ("gsva_scores_"+all_data_name+"_500", all_model, None, "500 modules"),
# ]

wd = 'Radiosensitivity Prediction/model_evaluation_on_mice_data'
download_dir = 'Radiosensitivity Prediction/data'
label = 'RT_Ratio'

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12
})

# Figure: 2 rows (scatter, metrics) x 3 columns (experiments)
fig = plt.figure(figsize=(15, 6.5))
gs = GridSpec(nrows=2, ncols=len(list_of_experiments), height_ratios=[4, 1], hspace=0.35, wspace=0.28)

def minmax01(x):
    x = np.asarray(x, dtype=float)
    rng = np.ptp(x)
    if not np.isfinite(rng) or rng == 0:
        return np.zeros_like(x)
    return (x - np.nanmin(x)) / rng

for i, (data_name, model, weights, title) in enumerate(list_of_experiments):
    # --- Load predictions ---
    if weights is not None:
        results_df = pd.read_csv(f'{wd}/predictions/{data_name}_{model}_{weights}_predictions.csv', index_col=0)
    else:
        results_df = pd.read_csv(f'{wd}/predictions/{data_name}_{model}_predictions.csv', index_col=0)

    # --- Load ground truth and merge ---
    df = pd.read_csv(f"{download_dir}/Mice/PDX_response_to_standard_therapy.csv")
    df['PDX_Line'] = ['Mayo-PDX-Sarkaria-' + str(val) for val in df['PDX_Line'].values]
    merged = results_df.merge(df[['PDX_Line', label]], left_index=True, right_on='PDX_Line')

    # --- Mask + align (invert AUC so higher = more sensitive, like RT_Ratio) ---
    mask = np.isfinite(merged[label].values) & np.isfinite(merged["prediction"].values)
    y_true = merged.loc[mask, label].values.astype(float)            # RT_Ratio (higher = sensitive)
    y_pred_auc = merged.loc[mask, "prediction"].values.astype(float) # AUC (lower = sensitive)
    y_pred_sens = -y_pred_auc                                        # align direction

    # Scaled for NMSE in [0,1]
    y_true_mm = minmax01(y_true)
    y_pred_mm = minmax01(y_pred_sens)

    # Metrics
    spearman_rho = pd.Series(y_true).corr(pd.Series(y_pred_sens), method="spearman")
    nmse01 = mean_squared_error(y_true_mm, y_pred_mm)
    median_thr = np.median(y_true)
    y_bin = (y_true >= median_thr).astype(int)
    if y_bin.min() != y_bin.max():
        auc_sep = roc_auc_score(y_bin, y_pred_sens)
        auc_text = f"AUROC (median split): {auc_sep:.3f}"
    else:
        auc_text = "AUROC: n/a (1 class)"

    # --- Axes for this column ---
    ax_scatter = fig.add_subplot(gs[0, i])
    ax_text = fig.add_subplot(gs[1, i])
    ax_text.axis("off")

    # --- Scatter: raw AUC vs RT_Ratio (to show inverse relationship) ---
    ax_scatter.scatter(merged.loc[mask, "prediction"], merged.loc[mask, label])
    ax_scatter.set_title(title)
    ax_scatter.set_xlabel("AUC (Prediction)")
    if i == 0:
        ax_scatter.set_ylabel("RT Ratio (True)")
    else:
        ax_scatter.set_ylabel("")

    # Optional inverse-trend guide line (top-left to bottom-right)
    x1, x2 = 1.19, max(5.0, merged.loc[mask, "prediction"].max())
    y1, y2 = merged.loc[mask, label].max(), merged.loc[mask, label].min()
    ax_scatter.plot([x1, x2], [y1, y2], linestyle='--', color='gray')
    ax_scatter.grid(True)

    # --- Metrics text (direction-aligned) ---
    metrics_text = (
        f"  Spearman ρ: {spearman_rho:.3f}\n"
        f"  normalized MSE [0,1]: {nmse01:.3f}\n"
        f"  {auc_text}"
    )
    ax_text.text(0, 1, metrics_text, ha="left", va="top", fontsize=12)

# plt.suptitle(f"{all_data_name}, {all_model} — comparison across settings", y=1.02, fontsize=16)
plt.tight_layout()
plt.show()
