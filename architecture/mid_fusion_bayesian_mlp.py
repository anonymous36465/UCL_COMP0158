import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Reuse the exact loss definitions from your bayesian_regressor module
from architecture.bayesian_regressor import fixed_var_nll, nll_with_pred_var

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


class BayesianMLP_MidFusion:
    """
    Mid-fusion MLP with MC Dropout + (optional) heteroscedastic head, trained with label-noise-aware NLL.
    API mirrors your MLP_mid_fusion but adds y_sem + Bayesian bits.
    """
    def __init__(self,
                 datasets,
                 head_input_dims=None,
                 input_dim=500,
                 feature_layers=3,
                 combining_layers=(256, 128, 1),
                 feature_hidden_dim=128,
                 learning_rate=1e-3,
                 num_epochs=10,
                 batch_size=32,
                 dropout_prob=0.1,
                 predict_variance=True,
                 weight_decay=0.0,
                 mc_samples=20,
                 default_y_sem=None,
                 seed: int = 42,
                scheduler="plateau",           # NEW: None or "plateau"
                sched_factor=0.5,
                sched_patience=2,
                sched_min_lr=1e-6,
                early_stopping_patience=None,  # e.g. 5; None disables
        ):
        self.datasets = datasets
        self.num_heads = len(datasets)
        self.input_dim = input_dim
        self.head_input_dims = head_input_dims or [input_dim] * self.num_heads
        self.feature_hidden_dim = feature_hidden_dim
        self.feature_layers = feature_layers
        self.combining_layers = list(combining_layers)
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dropout_prob = dropout_prob
        self.predict_variance = predict_variance
        self.weight_decay = weight_decay
        self.mc_samples = mc_samples
        self.default_y_sem = default_y_sem
        self.scheduler = scheduler
        self.sched_factor = sched_factor
        self.sched_patience = sched_patience
        self.sched_min_lr = sched_min_lr
        self.early_stopping_patience = early_stopping_patience

        torch.manual_seed(seed)
        np.random.seed(seed)

        self._build_model()

    def _build_model(self):
        self.model = nn.Module()
        # Per-head feature extractors
        self.model.feature_extractors = nn.ModuleList([
            FeatureExtractor(in_dim, self.feature_hidden_dim, self.feature_layers, self.dropout_prob)
            for in_dim in self.head_input_dims
        ])

        # Combining regressor with Dropout (for MC Dropout)
        if len(self.combining_layers) == 1:
            self.model.classifier = nn.Linear(self.feature_hidden_dim * self.num_heads, self.combining_layers[0])
        else:
            layers = [nn.Linear(self.feature_hidden_dim * self.num_heads, self.combining_layers[0]),
                    nn.ReLU(), nn.Dropout(self.dropout_prob)]
            for i in range(len(self.combining_layers) - 2):
                layers += [nn.Linear(self.combining_layers[i], self.combining_layers[i+1]),
                        nn.ReLU(), nn.Dropout(self.dropout_prob)]

            # Final head: 2 units (mean, log_var) if predict_variance else 1 unit (mean only)
            out_dim = 2 if self.predict_variance else 1
            layers += [nn.Linear(self.combining_layers[-2], out_dim)]
            self.model.classifier = nn.Sequential(*layers)

        self.model.to(device)

    def _forward_raw(self, x_tensor):
        """
        Returns:
            if predict_variance: (mean, log_var) each [B,1]
            else: (mean, None)
        """
        split_inputs = torch.split(x_tensor, self.head_input_dims, dim=1)
        features = [extractor(chunk) for extractor, chunk in zip(self.model.feature_extractors, split_inputs)]
        fused = torch.cat(features, dim=1)
        out = self.model.classifier(fused)
        if self.predict_variance:
            mean = out[..., :1]
            log_var = out[..., 1:]
            return mean, log_var
        else:
            return out, None

    # ---------- Data handling ----------
    def _make_loader(self, X, y, y_sem=None, shuffle=True):
        if y_sem is None:
            y_sem = self.default_y_sem

        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.float32)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)

        if y_sem is None:
            sem_t = torch.zeros(len(X_t), 1, dtype=torch.float32)
        else:
            sem_t = torch.as_tensor(y_sem, dtype=torch.float32)
            if sem_t.ndim == 0:        # scalar
                sem_t = sem_t.view(1, 1).repeat(len(X_t), 1)
            elif sem_t.ndim == 1:      # (N,)
                sem_t = sem_t.unsqueeze(1)  # -> (N,1)

        ds = TensorDataset(X_t, y_t, sem_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    # ---------- Training / Eval ----------
    def fit(self, X, y, y_sem=None, log=True):
        loader = self._make_loader(X, y, y_sem=y_sem, shuffle=True)
        criterion = nn.MSELoss()
        optim_ = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if self.scheduler == "plateau":
            sched = optim.lr_scheduler.ReduceLROnPlateau(
                optim_, mode="min", factor=self.sched_factor,
                patience=self.sched_patience, min_lr=self.sched_min_lr
            )
        else:
            sched = None
        
        best = math.inf
        patience_left = self.early_stopping_patience

        self.model.train()
        for epoch in range(self.num_epochs):
            total = 0.0
            for xb, yb, semb in loader:
                xb, yb, semb = xb.to(device), yb.to(device), semb.to(device)

                optim_.zero_grad()
                y_mean, y_log_var = self._forward_raw(xb)

                if self.predict_variance:
                    loss = nll_with_pred_var(y_mean, y_log_var, yb, semb)
                else:
                    loss = fixed_var_nll(y_mean, yb, semb)

                loss.backward()
                optim_.step()
                total += loss.item()
            epoch_loss = total /len(loader)
            if log:
                print(f"Epoch [{epoch+1}/{self.num_epochs}] Loss: {total/len(loader):.6f}")
            
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

    @torch.no_grad()
    def evaluate_nll(self, X, y, y_sem=None, mc_samples=None):
        """
        MC-averaged NLL on a dataset, including label noise and (optionally) predicted aleatoric noise.
        """
        loader = self._make_loader(X, y, y_sem=y_sem, shuffle=False)
        mc = self.mc_samples if mc_samples is None else mc_samples

        # Enable dropout during MC estimation
        self.model.train()
        total = 0.0
        count = 0
        for xb, yb, semb in loader:
            xb, yb, semb = xb.to(device), yb.to(device), semb.to(device)
            nll_acc = 0.0
            for _ in range(mc):
                y_mean, y_log_var = self._forward_raw(xb)
                if self.predict_variance:
                    nll_acc += nll_with_pred_var(y_mean, y_log_var, yb, semb).item()
                else:
                    nll_acc += fixed_var_nll(y_mean, yb, semb).item()
            total += (nll_acc / mc) * xb.size(0)
            count += xb.size(0)
        return total / max(count, 1)

    @torch.no_grad()
    def predict(self, X, mc_samples=None, return_std=False):
        """
        Returns:
            mean predictions; if return_std=True, also predictive std combining
            epistemic (MC Dropout) + aleatoric (if predict_variance=True).
        """
        mc = self.mc_samples if mc_samples is None else mc_samples
        X_t = torch.as_tensor(X, dtype=torch.float32).to(device)

        # Keep dropout active for MC sampling
        self.model.train()

        preds = []
        ale_vars = []
        for _ in range(mc):
            y_mean, y_log_var = self._forward_raw(X_t)
            preds.append(y_mean)
            if self.predict_variance:
                ale_vars.append(torch.exp(y_log_var))

        preds = torch.stack(preds, dim=0)                 # [MC, B, 1]
        mean_pred = preds.mean(dim=0)                     # [B, 1]
        epi_var = preds.var(dim=0, unbiased=False)        # [B, 1]

        if self.predict_variance:
            ale_var = torch.stack(ale_vars, dim=0).mean(dim=0)  # [B, 1]
            total_var = ale_var + epi_var
        else:
            total_var = epi_var

        if return_std:
            return mean_pred.cpu().numpy().squeeze(), torch.sqrt(total_var).cpu().numpy().squeeze()
        else:
            return mean_pred.cpu().numpy().squeeze()
