import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union

YErrType = Union[List[float], np.ndarray, Dict[str, List[float]]]

def plot_grouped_bars_with_uncertainty(
    data: Dict[str, List[float]],
    methods: List[str],
    datasets: Optional[List[str]] = None,
    errors: Optional[Dict[str, YErrType]] = None,
    title: Optional[str] = None,
    ylabel: str = "Performance",
    xlabel: str = "Method",
    bar_total_width: float = 0.8,
    capsize: float = 4.0,
    annotate: bool = False,
    colors: Optional[Union[List[str], Dict[str, str]]] = None,
    ax: Optional[plt.Axes] = None,
    dataset_names: List[str] = None,
    hatches: List[bool] = None,
    legend_title: str = "Dataset",
    # ---------- NEW: font controls ----------
    fontsize: int = 16,                 # base size
    title_size: Optional[int] = None,   # defaults: fontsize+2
    label_size: Optional[int] = None,   # defaults: fontsize
    tick_size: Optional[int] = None,    # defaults: fontsize-2
    legend_size: Optional[int] = None,  # defaults: fontsize-2
    legend_title_size: Optional[int] = None,  # defaults: fontsize-1
    annotation_size: Optional[int] = None     # defaults: fontsize-3
):
    """
    Plot grouped bar charts with uncertainty.
    """

    # ---------- defaults for sizes ----------
    if title_size is None:         title_size = fontsize + 2
    if label_size is None:         label_size = fontsize
    if tick_size is None:          tick_size = max(8, fontsize - 2)
    if legend_size is None:        legend_size = max(8, fontsize - 2)
    if legend_title_size is None:  legend_title_size = max(8, fontsize - 1)
    if annotation_size is None:    annotation_size = max(8, fontsize - 3)

    if datasets is None:
        datasets = list(data.keys())
    if dataset_names is None:
        dataset_names = list(data.keys())

    n_methods = len(methods)
    x = np.arange(n_methods)
    m = len(datasets)
    if m == 0:
        raise ValueError("No datasets to plot.")

    bar_width = bar_total_width / m

    if ax is None:
        fig, ax = plt.subplots(figsize=(7 + 0.5 * n_methods, 5.5))
    else:
        fig = ax.figure

    def _yerr_for_dataset(ds: str):
        if errors is None or ds not in errors or errors[ds] is None:
            return None
        e = errors[ds]
        if isinstance(e, dict):
            lower = np.asarray(e.get("lower"), dtype=float)
            upper = np.asarray(e.get("upper"), dtype=float)
            if len(lower) != n_methods or len(upper) != n_methods:
                raise ValueError(f"Asymmetric errors for '{ds}' must match number of methods ({n_methods}).")
            return np.vstack([lower, upper])
        else:
            e_arr = np.asarray(e, dtype=float)
            if len(e_arr) != n_methods:
                raise ValueError(f"Symmetric errors for '{ds}' must match number of methods ({n_methods}).")
            return e_arr

    for i, ds in enumerate(datasets):
        if ds not in data:
            raise KeyError(f"Dataset '{ds}' not found in `data`.")
        y = np.asarray(data[ds], dtype=float)
        if len(y) != n_methods:
            raise ValueError(f"Dataset '{ds}' has {len(y)} values, expected {n_methods} (same as methods).")

        x_pos = x - bar_total_width/2 + (i + 0.5) * bar_width
        yerr = _yerr_for_dataset(ds)

        # Pick color if provided
        if isinstance(colors, dict):
            color = colors.get(ds, None)
        elif isinstance(colors, list):
            color = colors[i] if i < len(colors) else None
        else:
            color = None

        bars = ax.bar(
            x_pos, y, width=bar_width, label=dataset_names[i],
            yerr=yerr, capsize=capsize, color=color,
            hatch=('///' if (hatches and i < len(hatches) and hatches[i]) else None),
            edgecolor="white"
        )

        if annotate:
            for rect, val in zip(bars, y):
                height = rect.get_height()
                ax.annotate(
                    f"{val:.2f}",
                    xy=(rect.get_x() + rect.get_width()/2.0, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=annotation_size,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0)
                )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=tick_size)
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)

    if title:
        ax.set_title(title, fontsize=title_size)

    leg = ax.legend(title=legend_title, prop={"size": legend_size}, title_fontsize=legend_title_size)
    ax.margins(x=0.02)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.tick_params(axis="x", labelsize=tick_size)

    fig.tight_layout()
    return fig, ax


# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Dict, List, Optional, Union

# YErrType = Union[List[float], np.ndarray, Dict[str, List[float]]]

def plot_grouped_bars_with_custom_buckets(
    data: Dict[str, List[float]],
    methods: List[str],
    datasets: Optional[List[str]] = None,
    # --- NEW: grouping controls ---
    bucket_by: Optional[Union[List[str], Dict[str, str]]] = None,  # e.g. ["single","single","single","two","two","two","three"]
    bucket_order: Optional[List[str]] = None,                       # e.g. ["single omic","two omics","three omics"]
    bucket_names_map: Optional[Dict[str, str]] = None,              # map raw bucket keys -> pretty labels
    bucket_gap: float = 0.35,                                       # gap (in bar-width units) between buckets within a method cluster
    show_bucket_dividers: bool = True,                              # thin separators inside each cluster
    show_bucket_labels: bool = True,                                # small labels above clusters
    # --- errors/colors/appearance (unchanged APIs) ---
    errors: Optional[Dict[str, YErrType]] = None,
    title: Optional[str] = None,
    ylabel: str = "Performance",
    xlabel: str = "Method",
    bar_total_width: float = 0.8,
    capsize: float = 4.0,
    annotate: bool = False,
    colors: Optional[Union[List[str], Dict[str, str]]] = None,
    ax: Optional[plt.Axes] = None,
    dataset_names: Optional[List[str]] = None,
    hatches: Optional[List[bool]] = None,
    legend_title: str = "Dataset",
    # --- font controls (kept identical defaults/behavior) ---
    fontsize: int = 16,
    title_size: Optional[int] = None,
    label_size: Optional[int] = None,
    tick_size: Optional[int] = None,
    legend_size: Optional[int] = None,
    legend_title_size: Optional[int] = None,
    annotation_size: Optional[int] = None
):
    """
    Plot grouped bar charts (by method on x-axis) with datasets arranged into custom buckets
    (e.g., 'single omic', 'two omics', 'three omics') inside each method's cluster.
    """

    # ---------- defaults for sizes ----------
    if title_size is None:         title_size = fontsize + 2
    if label_size is None:         label_size = fontsize
    if tick_size is None:          tick_size = max(8, fontsize - 2)
    if legend_size is None:        legend_size = max(8, fontsize - 2)
    if legend_title_size is None:  legend_title_size = max(8, fontsize - 1)
    if annotation_size is None:    annotation_size = max(8, fontsize - 3)

    # ---------- dataset list / names ----------
    if datasets is None:
        datasets = list(data.keys())
    if dataset_names is None:
        dataset_names = datasets

    n_methods = len(methods)
    x = np.arange(n_methods)
    m = len(datasets)
    if m == 0:
        raise ValueError("No datasets to plot.")

    # ---------- build bucket assignment for each dataset ----------
    # bucket_key for each dataset (raw key); pretty name via bucket_names_map
    if bucket_by is None:
        # default: all in one bucket
        bucket_keys = ["_all"] * m
        if bucket_order is None:
            bucket_order = ["_all"]
        if bucket_names_map is None:
            bucket_names_map = {"_all": "All datasets"}
    else:
        if isinstance(bucket_by, dict):
            bucket_keys = [bucket_by[ds] for ds in datasets]
        else:
            if len(bucket_by) != m:
                raise ValueError("Length of bucket_by list must match number of datasets.")
            bucket_keys = list(bucket_by)

        # bucket order default: in the order of first appearance
        if bucket_order is None:
            seen = []
            for bk in bucket_keys:
                if bk not in seen:
                    seen.append(bk)
            bucket_order = seen

        if bucket_names_map is None:
            # pretty=raw by default
            bucket_names_map = {bk: bk for bk in set(bucket_keys)}

    # group datasets per bucket (preserving original dataset order within each bucket)
    bucket_to_indices: Dict[str, List[int]] = {bk: [] for bk in bucket_order}
    for i, bk in enumerate(bucket_keys):
        if bk not in bucket_to_indices:
            # if user passed bucket_by containing a bucket not listed in bucket_order, append it
            bucket_to_indices[bk] = []
            bucket_order.append(bk)
        bucket_to_indices[bk].append(i)

    # counts and internal layout calculations
    counts = [len(bucket_to_indices[bk]) for bk in bucket_order]
    G = len(bucket_order)
    total_bars = sum(counts)
    # effective "units" across one method's cluster = bars + gaps_between_buckets
    effective_units = total_bars + bucket_gap * max(0, G - 1)
    bar_width = bar_total_width / effective_units

    # compute left-to-right slot index for each dataset inside a cluster
    # we place buckets in bucket_order; within each bucket we place their datasets in original order
    slot_index_for_dataset = [None] * m
    slot = 0.0
    bucket_start_slot = {}
    bucket_end_slot = {}

    for gi, bk in enumerate(bucket_order):
        if gi > 0:
            slot += bucket_gap  # add gap before this bucket
        bucket_start_slot[bk] = slot
        for ds_idx in bucket_to_indices[bk]:
            slot_index_for_dataset[ds_idx] = slot
            slot += 1.0  # next dataset within the bucket
        bucket_end_slot[bk] = slot

    # center the cluster around x (same as your original: cluster centered at each method tick)
    cluster_left = -bar_total_width / 2.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(7 + 0.5 * n_methods, 5.5))
    else:
        fig = ax.figure

    # ---------- helper: yerr retrieval identical to your original ----------
    def _yerr_for_dataset(ds: str):
        if errors is None or ds not in errors or errors[ds] is None:
            return None
        e = errors[ds]
        if isinstance(e, dict):
            lower = np.asarray(e.get("lower"), dtype=float)
            upper = np.asarray(e.get("upper"), dtype=float)
            if len(lower) != n_methods or len(upper) != n_methods:
                raise ValueError(f"Asymmetric errors for '{ds}' must match number of methods ({n_methods}).")
            return np.vstack([lower, upper])
        else:
            e_arr = np.asarray(e, dtype=float)
            if len(e_arr) != n_methods:
                raise ValueError(f"Symmetric errors for '{ds}' must match number of methods ({n_methods}).")
            return e_arr

    # ---------- draw bars ----------
    bar_artists = []
    for i, ds in enumerate(datasets):
        if ds not in data:
            raise KeyError(f"Dataset '{ds}' not found in `data`.")
        y = np.asarray(data[ds], dtype=float)
        if len(y) != n_methods:
            raise ValueError(f"Dataset '{ds}' has {len(y)} values, expected {n_methods} (same as methods).")

        # horizontal offset for this dataset inside each method cluster
        offset = cluster_left + (slot_index_for_dataset[i] + 0.5) * bar_width
        x_pos = x + offset

        yerr = _yerr_for_dataset(ds)

        # color selection (same semantics as your original)
        if isinstance(colors, dict):
            color = colors.get(ds, None)
        elif isinstance(colors, list):
            color = colors[i] if i < len(colors) else None
        else:
            color = None

        bars = ax.bar(
            x_pos, y, width=bar_width, label=dataset_names[i],
            yerr=yerr, capsize=capsize, color=color,
            hatch=('///' if (hatches and i < len(hatches) and hatches[i]) else None),
            edgecolor="white"
        )
        bar_artists.append(bars)

        if annotate:
            for rect, val in zip(bars, y):
                height = rect.get_height()
                ax.annotate(
                    f"{val:.2f}",
                    xy=(rect.get_x() + rect.get_width()/2.0, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=annotation_size,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, lw=0)
                )

    # ---------- optional: bucket dividers and labels (cosmetic helpers) ----------
    if show_bucket_dividers and G > 1:
        for gi in range(G - 1):
            # position of divider between bucket gi and gi+1, expressed in slots
            left_end = bucket_end_slot[bucket_order[gi]]
            divider_slot = left_end + (bucket_gap / 2.0)
            x_offset = cluster_left + divider_slot * bar_width
            for xc in x:
                ax.axvline(x=xc + x_offset, ymin=0.02, ymax=0.98, linestyle=":", linewidth=1.0, alpha=0.6)

    if show_bucket_labels:
        # label centered above each bucket for the *first* method's cluster, then mirror across others
        y_top = ax.get_ylim()[1]
        y_text = y_top * 1.02  # slightly above current top; will re-tighten later
        for bk in bucket_order:
            start = cluster_left + bucket_start_slot[bk] * bar_width
            end   = cluster_left + bucket_end_slot[bk]   * bar_width
            center_offset = (start + end) / 2.0
            for xc in x:
                ax.text(xc + center_offset, y_text, bucket_names_map.get(bk, bk),
                        ha="center", va="bottom", fontsize=max(8, fontsize - 3), alpha=0.9)
        # expand ylim a touch to make room
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.12)

    # ---------- axes/legend identical styling ----------
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=tick_size)
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)

    if title:
        ax.set_title(title, fontsize=title_size)

    leg = ax.legend(title=legend_title, prop={"size": legend_size}, title_fontsize=legend_title_size)
    ax.margins(x=0.02)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.tick_params(axis="x", labelsize=tick_size)

    fig.tight_layout()
    return fig, ax
