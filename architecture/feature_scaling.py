
from sklearn.preprocessing import StandardScaler
from numpy import concatenate
class StandardScalerProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
    def fit(self, dataset):
        # Handle list of datasets
        if isinstance(dataset, list):
            all_xs = [d.xs for d in dataset if d.xs.shape[0] > 0]  # Only non-empty datasets
            if all_xs:  # Check if list is not empty
                self.scaler.fit(concatenate(all_xs, axis=0))
            else:
                print(":warning: Warning: No data to fit scaler.")
        else:
            if dataset.xs.shape[0] > 0:
                self.scaler.fit(dataset.xs)
            else:
                print(":warning: Warning: No data to fit scaler.")
    def transform(self, dataset):
        # If a list of datasets is passed
        if isinstance(dataset, list):
            for d in dataset:
                if d.xs.shape[0] > 0:
                    d.xs = self.scaler.transform(d.xs)
                    # :white_check_mark: Debug info for each dataset in the list
                    # print(":white_check_mark: [Scaling Complete]")
                    # print(f"→ Scaled features shape: {d.xs.shape}")
                    # print(f"→ Mean of first 5 features: {d.xs[:, :5].mean(axis=0)}")
                    # print(f"→ Std of first 5 features: {d.xs[:, :5].std(axis=0)}\n")
                else:
                    print(":warning: Warning: Skipping scaling for empty dataset.")
            return dataset
        else:
            if dataset.xs.shape[0] > 0:
                dataset.xs = self.scaler.transform(dataset.xs)
                # :white_check_mark: Debug info
                # print(":white_check_mark: [Scaling Complete]")
                # print(f"→ Scaled features shape: {dataset.xs.shape}")
                # print(f"→ Mean of first 5 features: {dataset.xs[:, :5].mean(axis=0)}")
                # print(f"→ Std of first 5 features: {dataset.xs[:, :5].std(axis=0)}\n")
            else:
                print(":warning: Warning: Skipping scaling for empty dataset.")
            return dataset
    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)