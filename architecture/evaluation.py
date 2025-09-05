import os, sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

from keras.models import Sequential

sys.path.insert(0, os.getcwd())
import architecture.coef_weights_utils as mcw

def collate_grid_search(results : dict):
    """
    Function for collating results after all settings in grid search have been computed
    Goes through all the summary results in each folder and selects the best hyperparameters based on metric
    for each test fold or across the the whole dataset depending on if nested cv is on.
    """
    summaries = []
    run_dir = results["save_dir"]
    gs_params = results["params"]
    gs_dirs = results["gs_dirs"]
    test_dirs = results["test_dirs"]
    for i, d in enumerate(gs_dirs):
        df = pd.read_csv(f"{d}/summary_results.csv", index_col=0)
        df["test_fold"] = test_dirs[i]
        df["hyperparams"] = gs_params[i]
        summaries.append(df)
    summaries = pd.concat(summaries).reset_index()
    summaries.to_csv(f"{run_dir}/results.csv")

def collate_folds(results : dict):
    """
    Function to perform any post processing across folds after the runs have finished
    """
    summaries = []
    for i, result in enumerate(results["results"]):
        df = pd.read_csv("{}/summary_results.csv".format(result),
                         index_col=0)
        df = df.reset_index(names="split")
        df["fold"] = i
        summaries.append(df)
    summaries = pd.concat(summaries)
    summaries.to_csv("{}/fold_summaries.csv".format(results["save_dir"]))


def save_results(results : dict, processor, metrics : dict, task : str, pred_idx : int = 0):
    """
    Function for saving the results based on a given processor function and metrics provided

    args:
        results (dict) : Results dictionary passed in via pipeline
        processor (function) : A function accepting a results dictionary, the fold string,
                                run directory, and metrics dictionary - outputs a dictionary
                                of results with keys same as metrics
        metrics (dict) : Dictionary of metric functions to evalulate the results
    """
    run_dir = results["save_dir"]
    # Save train results
    result_summary = []
    idxs = []
    idxs.append("train")
    result_summary.append(processor(results, "train", run_dir, metrics, task, pred_idx))
    # Save val results
    if len(results["val_df"]) > 0:
        idxs.append("val")
        result_summary.append(processor(results, "val", run_dir, metrics, task, pred_idx))
    if len(results["test_df"]) > 0:
        idxs.append("test")
        result_summary.append(processor(results, "test", run_dir, metrics, task, pred_idx))
    result_summary = pd.DataFrame(result_summary, index=idxs)
    result_summary.to_csv(f"{run_dir}/summary_results.csv")
    return result_summary

def save_supervised_result(results : dict, split : str, run_dir : str, metrics : dict, task : str,
                           pred_idx : int = 0):
    """
    Saves each split of supervised results e.g train, validation, test

    args:
        results (dict) : results dictionary containing all the information from the run
        split (str) : one of train, val, test to specify which split to save over
        run_dir (str) : path to the directory of the current fold run
        metrics (dict) : dictionary containing the metrics that you want computed
        task (str) : Specifies whether to treat the input labels as a group or individually
        pred_idx (int) : Assumes that model predictions are returned as a tuple, identifies
                        the item in the tuple that corresponds to supervised predictions
    
    returns:
        dict : Dictionary consisting of the summary metrics computed on the data
    """
    # Check if tuple is being returned, if so only keep the pred_idx element as predictions
    preds = results[f"{split}_preds"][pred_idx] if type(results[f"{split}_preds"]) is tuple else results[f"{split}_preds"]
    # Reduce if predictions dimension is larger than labels
    label_dims = len(results[f"{split}_df"].ys.shape)
    if len(preds.shape) > label_dims:
        preds = preds.flatten(start_dim=label_dims).mean(axis=-1)
    preds = preds.reshape(results[f"{split}_df"].ys.shape)
    # Collate results
    cols = results[f"{split}_df"].get_labels() + [f"{x}_pred" for x in results[f"{split}_df"].get_labels()]
    df = pd.DataFrame(np.concatenate((results[f"{split}_df"].ys, preds), axis=1), 
                            columns=cols, index=results[f"{split}_df"].ids)
    df.to_csv(f"{run_dir}/{split}_results.csv")
    out = {}
    for metric_name, metric_fn in metrics.items():
        if task == "individual":
            for i, label in enumerate(results[f"{split}_df"].get_labels()):
                is_na = np.isnan(results[f"{split}_df"].ys[:, i])
                out[f"{label}_{metric_name}"] = metric_fn(results[f"{split}_df"].ys[~is_na, i], preds[~is_na, i])
        elif task == "binary":
            for i, label in enumerate(results[f"{split}_df"].get_labels()):
                is_na = np.isnan(results[f"{split}_df"].ys[:, i])
                y_true = results[f"{split}_df"].ys[~is_na, i].astype(int)
                y_pred = (preds[~is_na, i] >= 0.5).astype(int)
                out[f"{label}_{metric_name}"] = metric_fn(y_true, y_pred)
        elif task == "multiclass_4":
            for i, label in enumerate(results[f"{split}_df"].get_labels()):
                is_na = np.isnan(results[f"{split}_df"].ys[:, i])
                y_true = results[f"{split}_df"].ys[~is_na, i].astype(int)
                y_pred = np.digitize(preds[~is_na, i], bins=[2, 4, 6, 8]).astype(int)
                out[f"{label}_{metric_name}"] = metric_fn(y_true, y_pred)

        elif task == "group":
            out[f"{metric_name}"] = metric_fn(results[f"{split}_df"].ys, preds)
    return out

def evaluate_on_external(results : dict, external_df : pd.DataFrame, tag : str):
    """
    Evaluates the current best model on an external dataset. Expects the dataset to have the first
    column as the index and first row as the feature names. Will align the input feature names according
    to the training data alignment ids. If there are missing inputs this is zero filled.

    args:
        results (dict) : results dictionary containing all the information from the run
        external_df (str) : path to the desired dataset to be evaluated
        tag (str) : Tag to give the external dataset results
    """
    # Load external dataset
    sample_ids = external_df.index
    # Get the model object
    model = results["model"]
    # Compute the difference in the input features and re-arrange the external dataset
    # to match the model expected input. Zero fills missing values.
    # Saves the overlap for analysis later
    model_inputs = set(results["train_df"].alignment_ids)
    external_inputs = set(external_df.columns)
    missing_inputs = list(model_inputs - external_inputs)
    with open(results["save_dir"] + f"/{tag}_feature_info.csv", "w") as f:
        f.write("model_features,dataset_features,overlap_features,missing_features\n")
        f.write("{},{},{},{}".format(len(model_inputs), len(external_inputs),
                                     len(model_inputs.intersection(external_inputs)), len(missing_inputs)))
    missing_inputs = pd.DataFrame(np.zeros((external_df.shape[0], len(missing_inputs))), index=external_df.index, columns=missing_inputs)
    df = pd.concat((external_df, missing_inputs), axis=1)
    df = df[results["train_df"].alignment_ids]
    # Evaluate on external dataset
    preds = model.predict(df)
    preds = pd.DataFrame(preds, index=sample_ids, columns=["auc_pred"])
    preds.to_csv(results["save_dir"] + f"/{tag}_results.csv", index=True)

def plot_channels(
    history,
    channels,
    filename: str,
    folder_name: str,
    xlabel: str = "epochs",
    ylabel: str = " ",
):
    """
    Plot the training history
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    plt.figure()
    for k in channels:
        v = history[k]
        plt.plot(v)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(channels)
    filename = os.path.join(folder_name, filename)
    plt.savefig(filename + ".pdf")
    plt.close()

def plot_history(results):
    """
    Make some plots of the history of training the model
    """
    folder_name = results["save_dir"] + "/train_hx"
    history = results["train_hx"].history
    keys = list(history.keys())

    losses = [x for x in keys if ("_loss" in x) and (x != "val_loss")]
    val_losses = [x for x in losses if "val_" in x]
    train_losses = [x for x in losses if ("val_" not in x) and (x != "loss")]

    monitors = [x for x in keys if "loss" not in x]
    val_monitors = [x for x in monitors if "val_" in x]
    train_monitors = [
        x for x in monitors if ("val_" not in x) and (x != "loss") and (x != "lr")
    ]

    monitors.sort()
    val_monitors.sort()
    train_monitors.sort()

    train_losses.sort()
    val_losses.sort()

    plot_channels(
        history, val_monitors, "val_monitors", folder_name, ylabel="Score [arb]"
    )
    plot_channels(
        history, train_monitors, "train_monitors", folder_name, ylabel="Score [arb]"
    )

    for v, t in zip(val_monitors, train_monitors):
        plot_channels(history, [v, t], t, folder_name, ylabel="Score [arb]")

    plot_channels(history, val_losses, "validation_loss", folder_name, ylabel="Loss ")
    plot_channels(history, train_losses, "training_loss", folder_name, ylabel="Loss")

    if "val_loss" in keys:
        plot_channels(history, ["val_loss", "loss"], "loss", folder_name, ylabel="Loss")
    else:
        plot_channels(history, ["loss"], "loss", folder_name, ylabel="Score [arb]")

    for v, t in zip(val_losses, train_losses):
        plot_channels(history, [v, t], t, folder_name, ylabel="Score [arb]")
    pd.DataFrame(history).to_csv(f"{folder_name}/train_hx.csv")

def get_coef_importance(
    model, X_train, y_train, target, feature_importance, detailed=True, **kwargs
):

    print(feature_importance)

    if feature_importance.startswith("skf"):
        coef_ = mcw.get_skf_weights(model, X_train, y_train, feature_importance)
        # pass
    elif feature_importance == "loss_gradient":
        coef_ = mcw.get_gradient_weights(
            model, X_train, y_train, signed=False, detailed=detailed, normalize=True
        )  # use total loss
    elif feature_importance == "loss_gradient_signed":
        coef_ = mcw.get_gradient_weights(
            model, X_train, y_train, signed=True, detailed=detailed, normalize=True
        )  # use total loss
    elif feature_importance == "gradient_outcome":
        coef_ = mcw.get_weights_gradient_outcome(
            model, X_train, y_train, target, multiply_by_input=False, signed=False
        )
    elif feature_importance == "gradient_outcome_signed":
        coef_ = mcw.get_weights_gradient_outcome(
            model,
            X_train,
            y_train,
            target=target,
            detailed=detailed,
            multiply_by_input=False,
            signed=True,
        )
    elif feature_importance == "gradient_outcome*input":
        coef_ = mcw.get_weights_gradient_outcome(
            model, X_train, y_train, target, multiply_by_input=True, signed=False
        )
    elif feature_importance == "gradient_outcome*input_signed":
        coef_ = mcw.get_weights_gradient_outcome(
            model, X_train, y_train, target, multiply_by_input=True, signed=True
        )

    elif feature_importance.startswith("deepexplain"):
        method = feature_importance.split("_")[1]
        coef_ = mcw.get_deep_explain_scores(
            model,
            X_train,
            y_train,
            target,
            method_name=method,
            detailed=detailed,
            **kwargs
        )

    elif feature_importance.startswith("shap"):
        method = feature_importance.split("_")[1]
        coef_ = mcw.get_shap_scores(
            model, X_train, y_train, target, method_name=method, detailed=detailed
        )

    elif feature_importance == "gradient_with_repeated_outputs":
        coef_ = mcw.get_gradient_weights_with_repeated_output(
            model, X_train, y_train, target
        )
    elif feature_importance == "permutation":
        coef_ = mcw.get_permutation_weights(model, X_train, y_train)
    elif feature_importance == "linear":
        coef_ = mcw.get_weights_linear_model(model, X_train, y_train)
    elif feature_importance == "one_to_one":
        weights = model.layers[1].get_weights()
        switch_layer_weights = weights[0]
        coef_ = np.abs(switch_layer_weights)
    else:
        coef_ = None
    return coef_

def get_layers(model, level=1):
    layers = []
    for i, l in enumerate(model.layers):

        # indent = '  ' * level + '-'
        if type(l) == Sequential:
            layers.extend(get_layers(l, level + 1))
        else:
            layers.append(l)

    return layers


def get_deeplift_global(results):
    global_coefs, sample_coefs = get_coef_importance(results["model"].predictor, results["train_df"].xs, results["train_df"].ys,
                                -1, "deepexplain_deeplift")
    features = results["model"].feature_names
    features["inputs"] = [x[1] for x in features["inputs"]]
    for k, v in global_coefs.items():
        df = pd.DataFrame(v, index=features[k], columns=["feature_importance"])
        df.to_csv(results["save_dir"] + f"/feature_importance_{k}.csv")


def evaluate_predictions_like_regression(tag, run_dir, data_dir,
                                         label_csv_path='cleveland_auc_full.csv',
                                         label_name='auc',
                                         pred_name='auc_pred',
                                         transform=lambda p: 2*p + 1.0,
                                         n_folds=5,
                                         bin_edges=None):  # e.g. [2,4,6,8] for binning to classes
    """
    Extended evaluation:
      - R², MSE
      - Overall accuracy
      - Overall relaxed accuracy (correct if predicted class == true or adjacent)
      - Per-class exact accuracy
      - Per-class relaxed accuracy
    """
    results_path = os.path.join(run_dir, tag)

    labels_df = pd.read_csv(f"{data_dir}/{label_csv_path}", index_col=0)
    label_series = labels_df[label_name] if label_name in labels_df.columns else labels_df.squeeze("columns")

    r2s, mses = [], []
    acc_overall_list, acc_relaxed_list = [], []
    acc_class_lists = {cls: [] for cls in range(4)}
    acc_class_relaxed_lists = {cls: [] for cls in range(4)}

    # def to_classes(y_true_vals, y_pred_vals):
    #     """Return y_true_cls, y_pred_cls in {0,1,2,3}."""
    #     if bin_edges is not None:
    #         y_true_cls = np.digitize(y_true_vals, bins=bin_edges)
    #         y_pred_cls = np.digitize(y_pred_vals, bins=bin_edges)
    #     else:
    #         y_true_cls = np.asarray(y_true_vals, dtype=int)
    #         y_pred_cls = np.rint(y_pred_vals).astype(int)
    #     y_true_cls = np.clip(y_true_cls, 0, 3)
    #     y_pred_cls = np.clip(y_pred_cls, 0, 3)
    #     return y_true_cls, y_pred_cls

    for test_id in range(n_folds):
        test_file = os.path.join(results_path, f'test_{test_id}', 'best', 'test_results.csv')
        if not os.path.exists(test_file):
            print(f"[warn] Missing: {test_file}")
            continue

        df = pd.read_csv(test_file, index_col=0)

        # Align ground truth
        y_true = label_series.reindex(df.index)
        if y_true.isna().any():
            missing = int(y_true.isna().sum())
            print(f"[warn] {missing} IDs missing in label CSV for fold {test_id}. Dropping them.")
            mask = ~y_true.isna()
            df = df.loc[mask]
            y_true = y_true.loc[mask]

        df['class'] = df['auc']
        df['class_pred'] = df['auc_pred']
        df[label_name] = y_true.values

        # Transform predictions
        if pred_name not in df.columns:
            pred_cols = [c for c in df.columns if c.endswith('_pred')]
            raise ValueError(f"'{pred_name}' not found. Available: {pred_cols}")
        df[pred_name] = transform(df[pred_name].astype(float).values)

        # Metrics (numeric)
        y_t = df[label_name].values
        y_p = df[pred_name].values
        r2 = r2_score(y_t, y_p)
        mse = mean_squared_error(y_t, y_p)
        r2s.append(r2)
        mses.append(mse)

        # Class-based metrics
        y_true_cls, y_pred_cls = df['class'], df['class_pred']

        # Overall metrics
        acc_overall = np.mean(y_true_cls == y_pred_cls)
        acc_overall_list.append(acc_overall)

        acc_relaxed = np.mean(np.abs(y_pred_cls - y_true_cls) <= 1)
        acc_relaxed_list.append(acc_relaxed)

        # Per-class metrics
        for cls in range(4):
            mask_cls = (y_true_cls == cls)
            if mask_cls.any():
                # Exact accuracy per class
                acc_cls_val = np.mean(y_pred_cls[mask_cls] == cls)
                acc_class_lists[cls].append(acc_cls_val)

                # Relaxed accuracy per class
                acc_cls_relaxed_val = np.mean(np.abs(y_pred_cls[mask_cls] - cls) <= 1)
                acc_class_relaxed_lists[cls].append(acc_cls_relaxed_val)
            else:
                acc_class_lists[cls].append(np.nan)
                acc_class_relaxed_lists[cls].append(np.nan)

        # Save modified CSV
        out_file = os.path.join(results_path, f'test_{test_id}', 'best', 'test_results_modified.csv')
        df.to_csv(out_file)
        print(f"[fold {test_id}] R²={r2:.6f} | MSE={mse:.6f} | acc={acc_overall:.6f} | acc_relaxed={acc_relaxed:.6f}")

    # Summary CSV
    if r2s:
        summary_rows = [
            {"metric": f"{label_name}_r2",           "mean": float(np.nanmean(r2s)), "std": float(np.nanstd(r2s, ddof=0))},
            {"metric": f"{label_name}_mse",          "mean": float(np.nanmean(mses)), "std": float(np.nanstd(mses, ddof=0))},
            {"metric": f"{label_name}_acc",          "mean": float(np.nanmean(acc_overall_list)), "std": float(np.nanstd(acc_overall_list, ddof=0))},
            {"metric": f"{label_name}_acc_relaxed",  "mean": float(np.nanmean(acc_relaxed_list)), "std": float(np.nanstd(acc_relaxed_list, ddof=0))}
        ]
        for cls in range(4):
            # summary_rows.append({
            #     "metric": f"{label_name}_acc_class_{cls}",
            #     "mean": float(np.nanmean(acc_class_lists[cls])),
            #     "std": float(np.nanstd(acc_class_lists[cls], ddof=0))
            # })
            summary_rows.append({
                "metric": f"{label_name}_acc_relaxed_class_{cls}",
                "mean": float(np.nanmean(acc_class_relaxed_lists[cls])),
                "std": float(np.nanstd(acc_class_relaxed_lists[cls], ddof=0))
            })

        summary = pd.DataFrame(summary_rows, columns=["metric", "mean", "std"])
        summary_file = os.path.join(run_dir, tag, 'summary_results_regression.csv')
        summary.to_csv(summary_file, index=False)
        print(f"[summary] saved -> {summary_file}")
        return summary

    else:
        print("[summary] No folds processed.")
        return pd.DataFrame(columns=["metric", "mean", "std"])

