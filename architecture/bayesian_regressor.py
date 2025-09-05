import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Losses that account for label noise ----------
@torch.jit.script
def fixed_var_nll(y_pred: torch.Tensor, y_true: torch.Tensor, sem: torch.Tensor):
    """
    Negative log-likelihood with *known* label noise (SEM).
    sem can be scalar, shape [B], or [B, 1].
    """
    var = sem**2
    # Ensure broadcast to match shape of y_pred/y_true
    if var.dim() == 1:
        var = var.unsqueeze(-1)
    # ( (y - yhat)^2 / (2*var) + 0.5*log(2*pi*var) )
    return (((y_true - y_pred) ** 2) / (2.0 * var) + 0.5 * torch.log(2.0 * math.pi * var)).mean()

@torch.jit.script
def nll_with_pred_var(y_mean: torch.Tensor, y_log_var: torch.Tensor, y_true: torch.Tensor, label_sem: torch.Tensor):
    """
    Heteroscedastic NLL where network also predicts data noise.
    Total variance = exp(log_var_pred) + label_sem^2  (variances add).
    label_sem can be 0 (scalar) if unknown.
    """
    pred_var = torch.exp(y_log_var)
    lab_var = label_sem ** 2
    if lab_var.dim() == 1:
        lab_var = lab_var.unsqueeze(-1)
    total_var = pred_var + lab_var
    return (((y_true - y_mean) ** 2) / (2.0 * total_var) + 0.5 * torch.log(2.0 * math.pi * total_var)).mean()

# ---------- Simple MLP with MC Dropout ("Bayesian-ish") ----------
class SimpleBayesianMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 128), dropout_prob=0.1, predict_variance=True):
        """
        predict_variance=True -> final head outputs [mean, log_var]
        """
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout_prob)]
            last = h
        out_dim = 2 if predict_variance else 1
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
        self.predict_variance = predict_variance

    def forward(self, x):
        out = self.net(x)
        if self.predict_variance:
            mean = out[..., :1]
            log_var = out[..., 1:]
            return mean, log_var
        else:
            return out, None

class BayesianRegressor:
    def __init__(
        self,
        input_dim,
        hidden_dims=(128, 128),
        dropout_prob=0.1,
        learning_rate=1e-3,
        num_epochs=20,
        batch_size=64,
        predict_variance=True,
        weight_decay=0.0,
        mc_samples=20,
        seed: int = 42,
        default_y_sem=None,   # <--- NEW
    ):
        self.model = SimpleBayesianMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_prob=dropout_prob,
            predict_variance=predict_variance,
        ).to(device)

        self.lr = learning_rate
        self.epochs = num_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.predict_variance = predict_variance
        self.mc_samples = mc_samples
        self.default_y_sem = default_y_sem   # <--- NEW

        torch.manual_seed(seed); np.random.seed(seed)

    def _make_loader(self, X, y, y_sem=None, shuffle=True):
        # Fallback to default if not provided
        if y_sem is None:
            y_sem = self.default_y_sem

        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.float32)
        if y_t.ndim == 1:
            y_t = y_t.unsqueeze(1)

        # normalize SEM shapes
        if y_sem is None:
            sem_t = torch.zeros(len(X_t), 1, dtype=torch.float32)
        else:
            sem_t = torch.as_tensor(y_sem, dtype=torch.float32)
            if sem_t.ndim == 0:          # scalar
                sem_t = sem_t.view(1, 1).repeat(len(X_t), 1)
            elif sem_t.ndim == 1:        # (N,)
                sem_t = sem_t.unsqueeze(1)  # -> (N,1)

        ds = TensorDataset(X_t, y_t, sem_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X, y, y_sem=None, log=True):
        loader = self._make_loader(X, y, y_sem=y_sem, shuffle=True)
        opt = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb, semb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                semb = semb.to(device)

                opt.zero_grad()
                y_mean, y_log_var = self.model(xb)

                if self.predict_variance:
                    loss = nll_with_pred_var(y_mean, y_log_var, yb, semb)
                else:
                    # Use fixed known label noise only
                    loss = fixed_var_nll(y_mean, yb, semb)

                loss.backward()
                opt.step()
                epoch_loss += loss.item()

            if log:
                print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {epoch_loss / len(loader):.6f}")

    @torch.no_grad()
    def predict(self, X, mc_samples=None, return_std=False):
        """
        mc_samples: if None, uses self.mc_samples.
        Returns mean predictions; if return_std, returns (mean, std) where std includes
        epistemic (MC Dropout) and aleatoric (if predicted) uncertainty.
        """
        mc = self.mc_samples if mc_samples is None else mc_samples
        X_t = torch.as_tensor(X, dtype=torch.float32).to(device)

        # Enable dropout at inference for MC sampling
        self.model.train()
        preds = []
        ale_vars = []

        for _ in range(mc):
            y_mean, y_log_var = self.model(X_t)
            preds.append(y_mean)
            if self.predict_variance:
                ale_vars.append(torch.exp(y_log_var))

        preds = torch.stack(preds, dim=0)              # [MC, B, 1]
        mean_pred = preds.mean(dim=0)                  # [B, 1]
        epi_var = preds.var(dim=0, unbiased=False)     # [B, 1]

        if self.predict_variance:
            ale_var = torch.stack(ale_vars, dim=0).mean(dim=0)  # [B, 1]
            total_var = ale_var + epi_var
        else:
            total_var = epi_var

        if return_std:
            return mean_pred.cpu().numpy().squeeze(), torch.sqrt(total_var).cpu().numpy().squeeze()
        else:
            return mean_pred.cpu().numpy().squeeze()

    @torch.no_grad()
    def evaluate_nll(self, X, y, y_sem=None, mc_samples=None):
        """
        Computes mean NLL under the model by averaging the MC estimate of NLL.
        If predict_variance=True, uses model aleatoric + label noise.
        If predict_variance=False, uses fixed label noise only.
        """
        mc = self.mc_samples if mc_samples is None else mc_samples
        loader = self._make_loader(X, y, y_sem=y_sem, shuffle=False)

        self.model.train()
        total = 0.0
        count = 0
        for xb, yb, semb in loader:
            xb, yb, semb = xb.to(device), yb.to(device), semb.to(device)
            nll_acc = 0.0
            for _ in range(mc):
                y_mean, y_log_var = self.model(xb)
                if self.predict_variance:
                    nll_acc += nll_with_pred_var(y_mean, y_log_var, yb, semb).item()
                else:
                    nll_acc += fixed_var_nll(y_mean, yb, semb).item()
            total += nll_acc / mc * xb.size(0)
            count += xb.size(0)
        return total / count