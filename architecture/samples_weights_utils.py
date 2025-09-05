import numpy as np

def make_weight_fn_percentile(y_all, alpha=2.0, eps=1e-6, normalize="global", clip=None):
    """
    Build w(y) = (|F(y)-0.5| + eps)^alpha from ALL labels y_all.
    normalize: "global" -> mean(w(y_all))=1, "batch" -> renorm per call, or None.
    """
    y_all = np.asarray(y_all)
    y_all = y_all[~np.isnan(y_all)]
    sorted_y = np.sort(y_all)
    n = len(sorted_y)

    def _raw_w(y):
        y = np.asarray(y)
        # empirical CDF via searchsorted
        p = np.searchsorted(sorted_y, y, side="right") / (n + 1e-12)
        w = (np.abs(p - 0.5) + eps) ** alpha
        if clip is not None:
            lo, hi = clip
            if lo is not None: w = np.maximum(w, lo)
            if hi is not None: w = np.minimum(w, hi)
        return w

    if normalize == "global":
        Z = _raw_w(sorted_y).mean() + 1e-12
        return lambda y: _raw_w(y) / Z
    elif normalize == "batch":
        return lambda y: (_raw_w(y) / (_raw_w(y).mean() + 1e-12))
    else:
        return _raw_w


def make_weight_fn_mad(y_all, alpha=2.0, eps=1e-6, normalize="global", clip=None):
    y_all = np.asarray(y_all)
    y_all = y_all[~np.isnan(y_all)]
    med = np.median(y_all)
    mad = np.median(np.abs(y_all - med))
    scale = 1.4826 * (mad + 1e-12)

    def _raw_w(y):
        z = np.abs(np.asarray(y) - med) / scale
        w = (z + eps) ** alpha
        if clip is not None:
            lo, hi = clip
            if lo is not None: w = np.maximum(w, lo)
            if hi is not None: w = np.minimum(w, hi)
        return w

    if normalize == "global":
        Z = _raw_w(y_all).mean() + 1e-12
        return lambda y: _raw_w(y) / Z
    elif normalize == "batch":
        return lambda y: (_raw_w(y) / (_raw_w(y).mean() + 1e-12))
    else:
        return _raw_w