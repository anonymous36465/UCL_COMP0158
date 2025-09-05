import numpy as np
import pandas as pd
import copy

class MultiViewDataset:
    """
    Class to hold multiple views of tabular data to support genomics datasets
    """
    def __init__(self):
        """
        Constructor to initiate the empty datastructures for loading data.
        """
        self.ids = []                   # Holds the ids used to integrate views across datasets
        self.data_views = {}            # Holds the actual data / different views
        self.labels = pd.DataFrame()    # Holds the targets / responses for supervised learning

    def load_data_view(self, view_name : str, data_fn : str, selected_columns : list = None, id_col : int = 0, 
                       preprocess = lambda x : x):
        """
        Loads in a dataset with options to select particular columns (if data is tabular).

        args:
            view_name (str)                 : Name for the loaded data
            data_fn (str)                   : Path to data
            selected_columns (list)         : Columns to be selected from the data
            id_col (int)                    : Column index to be used as the id column
            preprocess (fn DataFrame -> DataFrame) : Preprocessing function to apply to dataframe

        returns:
            dict : Returns a dictionary of duplicated ids and differences in features
        """
        out = {"common_ids" : [], "new_ids" : [], "common_features" : [], "old_features" : [],
               "new_features" : [], "old_ids" : []}
        data_cols = pd.read_csv(data_fn, index_col=None, nrows=1).columns.to_list()
        if selected_columns is not None:
            selected_columns = list(set(selected_columns).intersection(data_cols))
            df = pd.read_csv(data_fn, usecols=[data_cols[id_col]] + selected_columns)
        else:
            df = pd.read_csv(data_fn)
        df = preprocess(df.dropna().set_index(data_cols[id_col]).astype(np.float32))

        out["new_ids"] = list(set(df.index) - set(self.ids))                  # Find ids for new samples
        out["old_ids"] = list(set(self.ids) - set(df.index))                  # Find ids for old samples
        out["common_ids"] = list(set(df.index).intersection(self.ids))
        # Align indices across views and labels
        self._align_indices(out["new_ids"])
        # Add new view / data
        if view_name in self.data_views.keys():
            # If view already exists, ensure to align features
            out["new_features"] = list(set(df.columns) - set(self.data_views[view_name].columns))
            out["old_features"] = list(set(self.data_views[view_name].columns) - set(df.columns))
            out["common_features"] = list(set(df.columns).intersection(self.data_views[view_name].columns))
            new_cols = pd.DataFrame(np.full((len(self.ids), len(out["new_features"])), np.nan),
                                    columns=out["new_features"], index=self.ids).astype(np.float32)
            self.data_views[view_name] = pd.concat((self.data_views[view_name], new_cols), axis=1)
            self.data_views[view_name].loc[df.index, df.columns] = df
        else:
            # Add new view
            old_samples = pd.DataFrame(np.full((len(out["old_ids"]), df.shape[1]), fill_value=np.nan), 
                                           index=list(out["old_ids"]), columns=df.columns).astype(np.float32)
            # Combine old and new and rearrange to keep in order with self.ids
            self.data_views[view_name] = pd.concat([df, old_samples]).loc[self.ids]
            out["new_features"] = df.columns.to_list()
        self.data_views[view_name] = self.data_views[view_name][sorted(self.data_views[view_name].columns)]
        return out
    
    def get_views(self):
        """
        Accessor method for getting available views

        returns:
            list[str]
        """
        return list(self.data_view.keys())
    
    def get_labels(self):
        """
        Accessor method for getting available labels

        returns:
            list[str]
        """
        return self.labels.columns.to_list()

    def load_data_label(self, data_fn : str, id_col : int = 0):
        """
        Loads in a label. Assumes a dataframe containing labels / targets for supervised training

        args:
            data_fn (str) : Path to the data
            id_col (int) : Column index to be used as the id column
        """
        out = {"common_ids" : [], "new_ids" : [], "common_labels" : [], "old_labels" : [],
               "new_labels" : [], "old_ids" : []}
        df = pd.read_csv(data_fn, index_col=id_col)
        out["new_ids"] = list(set(df.index) - set(self.ids))                  # Find ids for new samples
        out["old_ids"] = list(set(self.ids) - set(df.index))                  # Find ids for old samples
        out["common_ids"] = list(set(df.index).intersection(self.ids))
        # Align indices across views and labels
        self._align_indices(out["new_ids"])
        out["new_labels"] = list(set(df.columns) - set(self.labels.columns))
        out["old_labels"] = list(set(self.labels.columns) - set(df.columns))
        out["common_labels"] = list(set(df.columns).intersection(self.labels.columns))
        self.labels[out["new_labels"]] = np.nan
        self.labels.loc[df.index, out["new_labels"]] = df.loc[:, out["new_labels"]]
        self.labels.loc[df.index, out["common_labels"]] = df.loc[:, out["common_labels"]]
        self.labels = self.labels[sorted(self.labels.columns)]
        return out

    def _align_indices(self, new_idxs : list[str]):
        """
        Internal method for aligning the indices across the different views and labels

        args:
            new_idxs (list[str]) : List of index values that are new compared to existing
        
        returns:
            None
        """
        self.ids = sorted(self.ids + new_idxs)
        # Amend old views
        for view, data in self.data_views.items():
            # Fill other views with empty rows for new samples
            new_samples = pd.DataFrame(np.full((len(new_idxs), data.shape[1]), fill_value=np.nan), 
                                        index=new_idxs, columns=data.columns)
            # Combine old and new and rearrange to keep in order with self.ids
            self.data_views[view] = pd.concat([data, new_samples]).loc[self.ids]
        # Amend label indices
        new_labels = pd.DataFrame(np.full((len(new_idxs), self.labels.shape[1]), fill_value=np.nan),
                                  index=new_idxs, columns=self.labels.columns)
        self.labels = pd.concat([self.labels, new_labels]).loc[self.ids]
    
    def align_views(self):
        """
        Placeholder method to be overwritten for subclasses to specify how to align the different views
        """
        pass
    
    def _get_train_test_split(self, train_proportion : float, seed : int = 42):
        """
        Internal method which returns a random split of the data by indices

        args:
            train_proportion (float) : Numeric value between 0 and 1 specifying size of train proportion
            seed (int) : Integer used to determine the randomness of the split
        """
        rng = np.random.default_rng(seed)
        idxs = np.arange(len(self.ids))
        split = int(len(idxs) * train_proportion)
        rng.shuffle(idxs)
        train_idxs = idxs[:split]
        test_idxs = idxs[split:]
        return train_idxs, test_idxs
    
    def _get_k_splits(self, n_splits : int, seed : int = 42):
        """
        Internal method which returns k splits of the data as indices

        args:
            n_splits (int) : Positive integer specifying the number of splits
            seed : Integer used to determine the randomness of the split

        returns:
            folds (list, list) : tuple of lists containing indices for train and validation
                                 folds
        """
        rng = np.random.default_rng(seed)
        idxs = np.arange(len(self.ids))
        splits = range(0, len(self.ids), len(self.ids) // n_splits)
        rng.shuffle(idxs)
        if n_splits > 1 :
            folds = []
            for i in range(n_splits):
                val_split = idxs[splits[i]:splits[i+1]] if i < n_splits - 1 else idxs[splits[i]:]
                train_split = list(set(idxs) - set(val_split))
                folds.append((train_split, val_split))
            return folds
        else:
            return [(idxs, [])]
    
    def _get_specific_split(self, train_ids, val_ids, test_ids, seed : int = 42):
        """
        Internal method which returns the numeric index of list of sample ids for each split.
        Enforces mutual exclusivity between sets

        args:
            train_ids (list or float) : list of training sample ids or float to describe proportion of dataset
            val_ids (list or float) : list of validation sample ids or float to describe proportion of dataset
            test_ids (list or float) : list of test sample ids or float to describe proportion of dataset
            seed (int) : Integer used to determine randomness of split
        
        returns:
            (list, list, list) : train, val, test numeric ids
        """
        # Keep track of samples not in a split
        remaining = set(self.ids)
        rng = np.random.default_rng(seed)
        # Enforce test set first
        if type(test_ids) is list:
            test_numeric_ids = [self.ids.index(x) for x in test_ids if x in self.ids]
            remaining = remaining - set(test_ids)
        else:
            test_samples = sorted(list(remaining))
            rng.shuffle(test_samples)
            n_samples = int(len(self.ids) * test_ids)
            if n_samples > len(remaining):
                raise Exception(f"Test split proportion is too large : {test_ids}")
            test_samples = test_samples[:n_samples]
            test_numeric_ids = [self.ids.index(x) for x in test_samples if x in self.ids]
            remaining = remaining - set(test_samples)
        # Enforce validation set next
        if type(val_ids) is list:
            val_numeric_ids = [self.ids.index(x) for x in val_ids if x in self.ids]
            remaining = remaining - set(val_ids)
        else:
            val_samples = sorted(list(remaining))
            rng.shuffle(val_samples)
            n_samples = int(len(self.ids) * val_ids)
            if n_samples > len(remaining):
                raise Exception(f"Validation split proportion is too large : {val_ids}")
            val_samples = val_samples[:n_samples]
            val_numeric_ids = [self.ids.index(x) for x in val_samples if x in self.ids]
            remaining = remaining - set(val_samples)
        # Enforce training set last
        if type(train_ids) is list:
            train_numeric_ids = [self.ids.index(x) for x in train_ids if x in self.ids]
            remaining = remaining - set(train_ids)
        else:
            train_samples = sorted(list(remaining))
            rng.shuffle(train_samples)
            n_samples = int(len(self.ids) * train_ids)
            if n_samples > len(remaining):
                raise Exception(f"Validation split proportion is too large : {train_ids}")
            train_samples = train_samples[:n_samples]
            train_numeric_ids = [self.ids.index(x) for x in train_samples if x in self.ids]
            remaining = remaining - set(train_samples)
        # Check mutual exclusiveness
        if len(set(train_numeric_ids).intersection(set(val_numeric_ids)).intersection(set(test_numeric_ids))) > 0:
            raise Exception("Training, validation, and test sets are not mutually exclusive")
        return train_numeric_ids, val_numeric_ids, test_numeric_ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Placeholder function to be overwritten for specific downstream processing methods
        """
        xs = {k : v.iloc[idx, :] for k,v in self.data_views.items()}
        ys = self.labels.iloc[idx, :]
        return xs, ys
    
class ConcatMultiViewDataset(MultiViewDataset):
    """
    Instantiation of MultiViewDataset that returns data as a flat concatenation of views
    """
    def __init__(self):
        super().__init__()
        self.xs = None    # To hold the concatenated data
        self.ys = None    # To hold the labels
        self.features = None    # To hold feature names after alignment
        self.alignment_ids = None   # To hold ids for alignment between views
    
    def align_views(self, method : str, view_aligner : dict = {}, drop_labels=True):
        """
        Method to call after loading all the data desired. Aligns the views by concatenating them
        into one dataframe prefixed by the view name.

        args:
            method (str) : Flag to specify which method to use for dealing with NA values
            view_aligner (dict) : Dictionary containing function to extract identifier from each view
                                    to use as alignment between views
            drop_labels (bool) : whether to drop samples with missing labels
        """
        # Copy individual views for computation but retain original if needed
        # Rename view features with view name prefix to ensure unique feature names
        views = {}
        self.alignment_ids = []
        for k, v in self.data_views.items():
            views[k] = v.copy(deep=True)
            if k in view_aligner.keys():
                self.alignment_ids += [view_aligner[k](c) for c in v.columns]
            else:
                self.alignment_ids += v.columns.to_list()
        # pad each view with missing set of genes / alignment ids
        # To enable putting different views of the same gene contiguously in the dataframe
        # To facilitate the Diagonal layer from the original pnet paper because
        # If a sparse layer is used instead it becomes extremely slow
        # So we waste some extra parameters and input features because of zero filled values
        column_set = set(self.alignment_ids)
        self.alignment_ids = []
        for k, v in views.items():
            missing_columns = column_set - set(v.columns)
            zero_fill = pd.DataFrame(np.zeros((v.shape[0], len(missing_columns))), columns=list(missing_columns), index=v.index)
            views[k] = pd.concat((v, zero_fill), axis=1)
            views[k] = views[k].loc[:, sorted(views[k].columns)]
            self.alignment_ids += list(views[k].columns)
            views[k].columns = [f"{k}_{c}" for c in views[k].columns]
        self.xs = pd.concat(views, axis=1)
        column_order = np.argsort(self.alignment_ids)
        self.xs = self.xs.iloc[:, column_order]
        self.alignment_ids = list(np.array(self.alignment_ids)[column_order])
        # Save columns as features for reference if neded
        self.features = self.xs.columns.to_list()
        # Save data as tensors for actual computation later
        self.xs = self.xs.to_numpy()
        # Convert labels to tensors as well for computation
        self.ys = self.labels.to_numpy()
        
        # Deal with NAs
        if method == "zero fill":
            self.xs[np.isnan(self.xs)] = 0.0
        elif method == "drop samples":
            valid_samples = np.isnan(self.xs).sum(axis=1) == 0
            for k,v in self.data_views.items():
                v = v.loc[valid_samples]
            self.labels = self.labels.loc[valid_samples]
            self.xs = self.xs[valid_samples, :]
            self.ys = self.ys[valid_samples, :]
            self.ids = list(np.array(self.ids)[valid_samples])
            print(self.ids)
        elif method == "drop features":
            valid_features = np.isnan(self.xs).sum(axis=0) == 0
            self.xs = self.xs[:, valid_features]
            self.features = list(np.array(self.features)[valid_features])
            self.alignment_ids = list(np.array(self.alignment_ids)[valid_features])
        
        if drop_labels:
            valid_samples = np.isnan(self.ys).sum(axis=1) == 0
            self.xs = self.xs[valid_samples, :]
            self.ys = self.ys[valid_samples, :]
            self.ids = list(np.array(self.ids)[valid_samples])
            valid_labels = self.ys.sum(axis=0) > 0
            self.ys = self.ys[:, valid_labels]
            self.labels = self.labels.loc[:, valid_labels]
    
    def get_features(self):
        """
        Accessor method for obtaining features of the dataset
        """
        return self.features
    
    def get_alignment_ids(self):
        """
        Accessor method for obtaining alignment ids of the dataset
        """
        return self.alignment_ids
    
    def get_train_test_split(self, train_proportion : float, seed : int = 42):
        """
        Method for obtaining a random train test split

        args:
            train_proportion (float) : Number between 0 and 1 to determine size of train split
            seed (int) : Integer to seed the random number generator for random split
        
        returns:
            ConcatMultiViewDataset, ConcatMultiViewDataset : Tuple with train test datasets
        """
        train_idxs, test_idxs = self._get_train_test_split(train_proportion, seed)
        train_data = self._copy(train_idxs)
        test_data = self._copy(test_idxs)
        return train_data, test_data
    
    def get_k_splits(self, n_splits : int, seed : int = 42):
        """
        Method for obtaining k folds for doing crossvalidation as a generator

        args:
            n_splits (int) : Positive integer specifying the number of splits
            seed : Integer used to determine the randomness of the split

        returns:
            ConcatMultiViewDataset, ConcatMultiViewDataset : Returns train and test datasets
        """
        for train_idxs, val_idxs in self._get_k_splits(n_splits, seed):
            train_df = self._copy(train_idxs)
            test_df = self._copy(val_idxs)
            yield (train_df, test_df)
    
    def get_specific_split(self, train_ids, val_ids, test_ids, seed: int = 42):
        """
        Method for obtaining specific splits based on input id lists

        args:
            train_ids (list or float) : list of training sample ids or training size proportion
            val_ids (list or float) : list of validation sample ids or validatoin size proportion
            test_ids (list or float) : list of test sample ids or test size proportion
            seed (int) : seed for random number generation if some splits are defined by proportions
        
        returns:
            (ConcatMultiViewDataset, ConcatMultiViewDataset, ConcatMultiViewDataset) : train, val, test dataset objects
        """
        train, val, test = self._get_specific_split(train_ids, val_ids, test_ids, seed)
        return self._copy(train), self._copy(val), self._copy(test)

    def _copy(self, idxs : list[int]):
        """
        Internal method to copy dataset for train test splits. Creates a deep copy of
        current data based on provided indices

        args:
            idxs (list[int]) : Indices to be used when copying over data
        
        returns:
            ConcatMultiViewDataset
        """
        out = copy.deepcopy(self)
        # filter ids
        out.ids = [out.ids[i] for i in idxs]
        # Filter views
        for k,v in out.data_views.items():
            out.data_views[k] = v.iloc[idxs, :]
        # Copy over labels
        out.labels = out.labels.iloc[idxs, :]
        # Copy over concatenated data and labels
        if self.xs is not None:
            out.xs = out.xs[idxs, :]
        if self.ys is not None:
            out.ys = out.ys[idxs, :]
        return out
    
    def __getitem__(self, idx):
        return self.xs[idx, :], self.ys[idx, :]
#