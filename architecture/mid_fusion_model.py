import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_prob=0.0):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_prob)]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_prob)]
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class MLP_mid_fusion:
    def __init__(self, datasets, head_input_dims = None, input_dim=500, feature_layers=3, combining_layers=[256, 128, 1],
                 feature_hidden_dim=128, learning_rate=1e-3, num_epochs=10, dropout_prob=0.0, identity_head_indices=None):
        self.datasets = datasets
        self.num_heads = len(datasets)
        self.input_dim = input_dim
        self.head_input_dims = head_input_dims or [input_dim] * self.num_heads 
        self.feature_hidden_dim = feature_hidden_dim
        self.feature_layers = feature_layers
        self.combining_layers = combining_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.dropout_prob = dropout_prob
        self.identity_head_set = set(identity_head_indices or [])

        self._build_model()

    def _build_model(self):
        self.model = nn.Module()

        feature_extractors = []
        self.head_output_dims = [] 
        for i, in_dim in enumerate(self.head_input_dims):
            if i in self.identity_head_set:
                feature_extractors.append(nn.Identity())
                self.head_output_dims.append(in_dim)  # identity keeps size
            else:
                feature_extractors.append(
                    FeatureExtractor(in_dim, self.feature_hidden_dim, self.feature_layers, self.dropout_prob)
                )
                self.head_output_dims.append(self.feature_hidden_dim)

        self.model.feature_extractors = nn.ModuleList(feature_extractors)

        # First combining layer must match the SUM of head output dims
        comb_in = sum(self.head_output_dims)
        regressor_layers = [nn.Linear(comb_in, self.combining_layers[0]), nn.ReLU(), nn.Dropout(self.dropout_prob)]
        for i in range(len(self.combining_layers) - 2):
            regressor_layers += [
                nn.Linear(self.combining_layers[i], self.combining_layers[i+1]),
                nn.ReLU(),
                nn.Dropout(self.dropout_prob),
            ]
        regressor_layers += [nn.Linear(self.combining_layers[-2], self.combining_layers[-1])]
        self.model.classifier = nn.Sequential(*regressor_layers)

        self.model.to(device)

    def _forward(self, x_tensor):
        split_inputs = torch.split(x_tensor, self.head_input_dims, dim=1)
        features = [extractor(chunk) for extractor, chunk in zip(self.model.feature_extractors, split_inputs)]
        fused = torch.cat(features, dim=1)
        return self.model.classifier(fused)

    def fit(self, X, y, batch_size=32, log=True):
        epochs = self.num_epochs
        lr = self.learning_rate
        # Convert NumPy -> Tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = self._forward(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if log:
                print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss / len(loader):.4f}")

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        self.model.eval()
        with torch.no_grad():
            preds = self._forward(X_tensor)

        return preds.cpu().numpy().squeeze()


class MLP_mid_fusion_scheduled:
    def __init__(
        self,
        datasets,
        head_input_dims=None,
        input_dim=500,
        feature_layers=3,
        combining_layers=[256, 128, 1],
        feature_hidden_dim=128,
        learning_rate=1e-3,
        num_epochs=10,
        dropout_prob=0.0,
        identity_head_indices=None,
        active_head_indices=None,      # NEW: which modalities to use in this run
        weight_decay=0.0,              # NEW: scikit-like alpha
        scheduler="plateau",           # NEW: None or "plateau"
        sched_factor=0.5,
        sched_patience=2,
        sched_min_lr=1e-6,
        early_stopping_patience=None,  # e.g. 5; None disables
        seed=42,
    ):
        self.datasets = datasets
        self.all_head_input_dims = head_input_dims or [input_dim] * len(datasets)
        self.feature_hidden_dim = feature_hidden_dim
        self.feature_layers = feature_layers
        self.combining_layers = combining_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.dropout_prob = dropout_prob

        self.identity_head_set = set(identity_head_indices or [])
        self.active_head_indices = sorted(active_head_indices if active_head_indices is not None
                                          else list(range(len(self.all_head_input_dims))))
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.sched_factor = sched_factor
        self.sched_patience = sched_patience
        self.sched_min_lr = sched_min_lr
        self.early_stopping_patience = early_stopping_patience

        # reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self._build_model()

    def _build_model(self):
        self.model = nn.Module()

        # Slice heads to only the active subset
        self.head_input_dims = [self.all_head_input_dims[i] for i in self.active_head_indices]

        feature_extractors, head_out_dims = [], []
        for j, in_dim in enumerate(self.head_input_dims):
            global_idx = self.active_head_indices[j]
            if global_idx in self.identity_head_set:
                feature_extractors.append(nn.Identity())
                head_out_dims.append(in_dim)
            else:
                feature_extractors.append(
                    FeatureExtractor(in_dim, self.feature_hidden_dim, self.feature_layers, self.dropout_prob)
                )
                head_out_dims.append(self.feature_hidden_dim)

        self.model.feature_extractors = nn.ModuleList(feature_extractors)
        comb_in = sum(head_out_dims)

        if len(self.combining_layers) == 1:
            self.model.classifier = nn.Linear(comb_in, self.combining_layers[0])
        else:
            reg = [nn.Linear(comb_in, self.combining_layers[0]), nn.ReLU(), nn.Dropout(self.dropout_prob)]
            for i in range(len(self.combining_layers) - 2):
                reg += [nn.Linear(self.combining_layers[i], self.combining_layers[i+1]), nn.ReLU(), nn.Dropout(self.dropout_prob)]
            reg += [nn.Linear(self.combining_layers[-2], self.combining_layers[-1])]
            self.model.classifier = nn.Sequential(*reg)

        self.model.to(device)

    def _forward(self, x_tensor):
        # Always split for ALL modalities then pick the active ones (keeps your external X layout unchanged)
        all_chunks = torch.split(x_tensor, self.all_head_input_dims, dim=1)
        active_chunks = [all_chunks[i] for i in self.active_head_indices]
        feats = [extractor(chunk) for extractor, chunk in zip(self.model.feature_extractors, active_chunks)]
        fused = torch.cat(feats, dim=1)
        return self.model.classifier(fused)

    def fit(self, X, y, batch_size=32, log=True):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1)

        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.scheduler == "plateau":
            sched = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=self.sched_factor,
                patience=self.sched_patience, min_lr=self.sched_min_lr
            )
        else:
            sched = None

        best = math.inf
        patience_left = self.early_stopping_patience

        self.model.train()
        for epoch in range(self.num_epochs):
            total = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = self._forward(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                total += loss.item()

            epoch_loss = total / len(loader)
            if log:
                print(f"Epoch [{epoch+1}/{self.num_epochs}] loss={epoch_loss:.5f}  lr={optimizer.param_groups[0]['lr']:.2e}")

            if sched is not None:
                sched.step(epoch_loss)

            if self.early_stopping_patience is not None:
                if epoch_loss + 1e-6 < best:
                    best = epoch_loss
                    patience_left = self.early_stopping_patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        if log:
                            print("Early stopping.")
                        break

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        self.model.eval()
        with torch.no_grad():
            preds = self._forward(X_tensor)
        return preds.cpu().numpy().squeeze()
