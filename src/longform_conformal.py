"""Conformal claim filtering for long-form QA: Original, Group-wise and Adaptive.

Scores. CCP is stored as negative values in [-1, 0], larger meaning more uncertain. We threshold
z = log(1 + CCP) and keep the claims with z <= t. The conformity score of a prompt is V, the largest
threshold that keeps no incorrect claim (longform_data.safe_threshold).

Adaptive. A quantile regressor f(x) of V given the prompt embedding is fitted on cal1. On cal2 we take
the lower conformal quantile q of V - f(x), and a test prompt uses the threshold f(x) + q. Shifting
log(1 + CCP) by f(x) is the multiplicative normalisation of the paper (Eqs. 10-11).
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from longform_data import INCORRECT_LABEL, safe_threshold

METRICS = ("coverage", "removed", "removed_true", "retained")


def log_uncertainty(ccp):
    ccp = np.asarray(ccp, dtype=float)
    if np.any(ccp < -1.0 - 1e-8) or np.any(ccp > 1e-8):
        raise ValueError("expected CCP scores in [-1, 0]")
    return np.log(np.maximum(1.0 + ccp, 1e-6))


def prepare(prompts, score_key):
    """Per-prompt claim scores z, claim labels, conformity scores V and categories."""
    z = [log_uncertainty(p["unc_scores"][score_key]) for p in prompts]
    labels = [np.asarray(p["labels"]) for p in prompts]
    V = np.asarray([safe_threshold(zi, li) for zi, li in zip(z, labels)])
    categories = np.asarray([int(p["category"]) for p in prompts])
    return z, labels, V, categories


def lower_conformal_quantile(values, alpha):
    """k-th smallest value with k = floor((n + 1) * alpha), or -inf if k = 0.
    For an exchangeable test score V, P(V < result) <= alpha."""
    values = np.sort(np.asarray(values, dtype=float))
    k = int(np.floor((len(values) + 1) * alpha + 1e-12))
    return values[k - 1] if k >= 1 else -np.inf


def paired_split(n, seed, cal1=0.3, cal2=0.4, test=0.3):
    """cal1 / cal2 / test indices; every method is evaluated on the same test prompts."""
    cal_idx, test_idx = train_test_split(np.arange(n), test_size=test, random_state=seed, shuffle=True)
    cal1_idx, cal2_idx = train_test_split(cal_idx, test_size=cal2 / (cal1 + cal2), random_state=seed)
    return cal1_idx, cal2_idx, test_idx


def evaluate_kept(labels, kept):
    """Metrics of one prompt given the mask of retained claims."""
    correct = labels != INCORRECT_LABEL
    return {
        "coverage": float(not np.any(labels[kept] == INCORRECT_LABEL)),
        "removed": 1.0 - kept.sum() / len(labels),
        "removed_true": 1.0 - (kept & correct).sum() / correct.sum() if correct.any() else np.nan,
        "retained": float(kept.sum()),
    }


def evaluate_threshold(z, labels, threshold):
    return evaluate_kept(labels, z <= threshold)


def aggregate_by_category(per_prompt, categories, n_categories):
    """Mean of each metric within each category."""
    out = {}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in per_prompt], dtype=float)
        out[metric] = np.asarray([np.nanmean(values[categories == c]) if np.any(categories == c) else np.nan
                                  for c in range(n_categories)])
    return out


# Quantile regressors of V given the prompt embedding. "adaptive" is the one of the paper; the others are
# the variants of the ablation (Table 4), each differing from it in one setting.
REGRESSORS = {
    "adaptive": {},
    "adaptive_d4": {"d": 4},
    "adaptive_d16": {"d": 16},
    "adaptive_d32": {"d": 32},
    "adaptive_h16": {"hidden_dim": 16, "lr": 5e-4, "max_epochs": 2000},
    "adaptive_2layers": {"n_layers": 2},
    "adaptive_early_stopping": {"weight_decay": 1e-3, "max_epochs": 3000, "patience": 200},
    "adaptive_bagging": {"n_bags": 5},
}


class MLPQuantile:
    """MLP on the first d embedding components, trained with the pinball loss (Adam, full batch). The
    output bias starts at the marginal quantile. With patience > 0, training stops early on a held-out
    20% of the training set; with n_bags > 1, the predictions of networks with different seeds are averaged."""

    def __init__(self, d=8, hidden_dim=64, n_layers=1, weight_decay=1e-4, lr=1e-3, max_epochs=1000,
                 patience=0, n_bags=1, seed=0):
        self.d, self.hidden_dim, self.n_layers = d, hidden_dim, n_layers
        self.weight_decay, self.lr, self.max_epochs = weight_decay, lr, max_epochs
        self.patience, self.n_bags, self.seed = patience, n_bags, seed

    def fit(self, X, y, q):
        self.q = q
        self.nets = [self._fit_one(X[:, :self.d], y, self.seed * 1000 + b) for b in range(self.n_bags)]
        return self

    def predict(self, X):
        with torch.no_grad():
            x = torch.tensor(X[:, :self.d], dtype=torch.float32)
            return np.mean([net(x)[:, 0].numpy() for net in self.nets], axis=0)

    def _fit_one(self, X, y, seed):
        torch.manual_seed(seed)
        if self.patience == 0:
            return self._train(X, y, None, None)
        tr, va = train_test_split(np.arange(len(X)), test_size=0.2, random_state=seed)
        return self._train(X[tr], y[tr], X[va], y[va])

    def _train(self, X, y, X_val, y_val):
        layers, width = [], X.shape[1]
        for _ in range(self.n_layers):
            layers += [nn.Linear(width, self.hidden_dim), nn.ReLU()]
            width = self.hidden_dim
        net = nn.Sequential(*layers, nn.Linear(width, 1))
        q = torch.tensor([self.q], dtype=torch.float32)
        with torch.no_grad():
            net[-1].weight.mul_(0.1)
            net[-1].bias.copy_(torch.tensor(np.quantile(y, q.numpy()), dtype=torch.float32))
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def loss(xb, yb):
            e = yb[:, None] - net(xb)
            return torch.max(q * e, (q - 1) * e).mean(0).sum()

        xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        if X_val is None:
            for _ in range(self.max_epochs):
                opt.zero_grad()
                loss(xt, yt).backward()
                opt.step()
            return net.eval()

        xv, yv = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
        best, best_state, since = np.inf, None, 0
        for _ in range(self.max_epochs):
            net.train()
            opt.zero_grad()
            loss(xt, yt).backward()
            opt.step()
            net.eval()
            with torch.no_grad():
                v = float(loss(xv, yv))
            if v < best - 1e-6:
                best, since = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                since += 1
                if since >= self.patience:
                    break
        net.load_state_dict(best_state)
        return net.eval()


def run_split(z, labels, V, categories, embeddings, alphas, seed, regressors, cal1=0.3, cal2=0.4, test=0.3,
              n_categories=8):
    """Evaluate all methods on one random split.

    Original: global split conformal on cal1 + cal2. Group-wise: the same within each category
    (uses the labels). Adaptive (one per regressor in REGRESSORS): fitted on cal1, residual quantile on cal2.
    Returns {method: {metric: [n_alpha, n_categories]}}.
    """
    cal1_idx, cal2_idx, test_idx = paired_split(len(V), seed, cal1, cal2, test)
    cal_idx = np.concatenate((cal1_idx, cal2_idx))
    scaler = StandardScaler().fit(embeddings[cal1_idx])
    X1, X2, Xt = (scaler.transform(embeddings[i]) for i in (cal1_idx, cal2_idx, test_idx))
    out = {}

    def record(method, a_idx, thresholds):
        res = out.setdefault(method, {m: np.full((len(alphas), n_categories), np.nan) for m in METRICS})
        per_prompt = [evaluate_threshold(z[i], labels[i], thresholds[k]) for k, i in enumerate(test_idx)]
        for metric, values in aggregate_by_category(per_prompt, categories[test_idx], n_categories).items():
            res[metric][a_idx] = values

    for a_idx, alpha in enumerate(alphas):
        record("orig", a_idx, np.full(len(test_idx), lower_conformal_quantile(V[cal_idx], alpha)))
        q_cat = np.array([lower_conformal_quantile(V[cal_idx][categories[cal_idx] == c], alpha)
                          for c in range(n_categories)])
        record("groupwise", a_idx, q_cat[categories[test_idx]])
        for name in regressors:
            reg = MLPQuantile(seed=seed, **REGRESSORS[name]).fit(X1, V[cal1_idx], alpha)
            q_res = lower_conformal_quantile(V[cal2_idx] - reg.predict(X2), alpha)
            record(name, a_idx, reg.predict(Xt) + q_res)
    return out
