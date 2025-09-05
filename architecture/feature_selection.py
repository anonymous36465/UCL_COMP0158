from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold 
import numpy as np
import pandas as pd


class SelectiveReducerWrapper:
    """
    Applies a feature selection or decomposition method (e.g., SelectKBest, TruncatedSVD)
    to a specified feature view in a multi-view dataset, keeping other views unchanged.

    Parameters:
        reducer: Class of the sklearn selector or decomposer (not an instance).
        params (dict): Parameters to initialize the reducer.
        dataset_name (str): The dataset name as specified in config["views"].
        featurename_to_reduce (str): The feature prefix to identify the target view.
        requires_y (bool): Whether the reducer requires a target y during fitting.
    """

    def __init__(self, reducer, params=None, dataset_name="gexpr_and_histone",
                 featurename_to_reduce="gexpr", requires_y=True):
        self.reducer_class = reducer
        self.params = params or {}
        self.reducer = self.reducer_class(**self.params)
        self.requires_y = requires_y

        self.dataset_name = dataset_name
        self.featurename_to_reduce = featurename_to_reduce

        self.to_reduce_idxs = []
        self.other_idxs = []
        self.selected_to_reduce_idxs = []

    def fit_transform(self, data):
        self._split_indices(data)

        X_reduce = data.xs[:, self.to_reduce_idxs]
        y = data.ys.ravel() if self.requires_y else None

        self.reducer.fit(X_reduce, y)
        X_selected = self.reducer.transform(X_reduce)

        # Save selected indices only if available
        if hasattr(self.reducer, "get_support"):
            self.selected_to_reduce_idxs = self.reducer.get_support(indices=True)
        else:
            self.selected_to_reduce_idxs = None

        return self._combine(data, X_selected)

    def transform(self, data):
        X_reduce = data.xs[:, self.to_reduce_idxs]
        X_selected = self.reducer.transform(X_reduce)
        return self._combine(data, X_selected)

    def _split_indices(self, data):
        prefix = self.dataset_name + "_" + self.featurename_to_reduce
        self.to_reduce_idxs = [i for i, (_, v) in enumerate(data.features) if v.startswith(prefix)]
        self.other_idxs = [i for i, (_, v) in enumerate(data.features) if not v.startswith(prefix)]

    def _combine(self, data, X_selected):
        X_other = data.xs[:, self.other_idxs]
        other_feats = [data.features[i] for i in self.other_idxs]

        if self.selected_to_reduce_idxs is not None:
            selected_feats = [
                data.features[self.to_reduce_idxs[i]] for i in self.selected_to_reduce_idxs
            ]
        else:
            # For decomposers: use generic feature names
            n_components = X_selected.shape[1]
            reducer_name = self.reducer_class.__name__
            selected_feats = [f"{reducer_name}_{i}" for i in range(n_components)]

        data_new = data._copy(np.arange(len(data.ys)))
        data_new.xs = np.hstack((X_other, X_selected))
        data_new.features = other_feats + selected_feats

        return data_new



# class VarianceThresholdWrapper:
#     def __init__(self, threshold=0.01):
#         self.selector = VarianceThreshold(threshold=threshold)

#     def fit(self, dataset):
#         self.selector.fit(dataset.xs)
    
#     def transform(self, dataset):
#         X_selected = self.selector.transform(dataset.xs)

#         data_new = dataset._copy(np.arange(len(dataset.ys)))
#         data_new.xs = X_selected
#         selected_indices = np.where(self.selector.get_support())[0]
#         data_new.features = [dataset.features[i] for i in selected_indices ]
#         return data_new

#     def fit_transform(self, dataset):
#         self.fit(dataset)
#         return self.transform(dataset)
    
   
import numpy as np

class SubsetSelectorWrapper:
    """
    A generic wrapper for sklearn-style feature selection methods.

    Can be used with any sklearn-compatible feature selector (e.g., VarianceThreshold, SelectKBest) on a ConcatMultiViewDataset data object.

    Parameters:
    - selector: Class of the sklearn selector (not an instance).
    - params (dict): Parameters to initialize the selector with.
    - requires_y (bool): whether the selector expects to see .fit(X, y) instead of just .fit(X)
    """

    def __init__(self, selector, params=None, requires_y=False):
        self.selector_class = selector
        self.params = params or {}
        self.selector = self.selector_class(**self.params)
        self.requires_y = requires_y

    def fit(self, dataset):
        if self.requires_y:
            self.selector.fit(dataset.xs, dataset.ys.ravel())
        else:
            self.selector.fit(dataset.xs)

    def transform(self, dataset):
        # do nothing with an empty array
        if isinstance(dataset, (list, np.ndarray)) and len(dataset) == 0:
            return dataset
        if len(dataset.ys) == 0:
            return dataset

        X_selected = self.selector.transform(dataset.xs)

        data_new = dataset._copy(np.arange(len(dataset.ys)))
        data_new.xs = X_selected

        # Only update feature names if the selector has get_support()
        # which is the case for sklearn.subset_selector object but isn't for sklearn.decomposition
        if hasattr(self.selector, "get_support"):
            selected_indices = np.where(self.selector.get_support())[0]
            data_new.features = [dataset.features[i] for i in selected_indices]
        else:
            # Otherwise just name the components, e.g., svd_0, svd_1, ...
            n_components = X_selected.shape[1]
            data_new.features = [f"{self.selector_class.__name__}_{i}" for i in range(n_components)]

        return data_new

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

