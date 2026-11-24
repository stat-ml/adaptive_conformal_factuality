"""Conformal prediction sets for multiple-choice QA (MMLU): Original and Adaptive.

The nonconformity score of a question is 1 - p(true answer). Original takes the conformal quantile of
the scores on cal2. Adaptive first fits, on cal1, an MLP that predicts the conditional (1 - alpha)-quantile
of the score from the question embedding, divides the scores by it and takes the conformal quantile of
the ratios on cal2. Splits are stratified by subject; both methods are evaluated on the same test questions.
"""

import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

SUBJECTS = [  # category id = position in this list
    "marketing", "professional_accounting", "college_computer_science", "formal_logic",
    "high_school_computer_science", "computer_security", "machine_learning", "clinical_knowledge",
    "high_school_biology", "anatomy", "college_chemistry", "college_medicine", "professional_medicine",
    "business_ethics", "public_relations", "management",
]
EPS = 1e-8


def load_mcqa_data(scores_path, embeddings_path, subjects=None):
    """Question-level arrays: answer probabilities averaged over the prompts of a question, true answer,
    subject id and embedding. With `subjects`, only those are kept; the embedding file must then have
    one row per kept line."""
    data = [json.loads(line) for line in open(scores_path, encoding="utf-8")]
    if subjects:
        data = [row for row in data if row["subject"] in subjects]
    embeddings = np.load(embeddings_path)
    assert len(embeddings) == len(data), "embeddings must have one row per line of the scores file"

    groups = {}
    for i, row in enumerate(data):
        groups.setdefault((row["subject"], row["j"]), []).append(i)
    keys = sorted(groups)
    probs = np.zeros((len(keys), len(data[0]["scores"])), dtype=np.float32)
    targets = np.zeros(len(keys), dtype=np.int64)
    categories = np.zeros(len(keys), dtype=np.int64)
    emb = np.zeros((len(keys), embeddings.shape[1]), dtype=np.float32)
    for qi, key in enumerate(keys):
        idxs = groups[key]
        probs[qi] = np.array([data[i]["scores"] for i in idxs], dtype=np.float32).mean(axis=0)
        targets[qi] = int(data[idxs[0]]["target"])
        categories[qi] = int(data[idxs[0]]["category"])
        emb[qi] = embeddings[idxs[0]]
    return probs, targets, categories, emb


def three_way_split(idxs, frac_cal1, frac_cal2, seed):
    perm = np.random.default_rng(seed).permutation(idxs)
    n1 = int(np.floor(frac_cal1 * len(idxs)))
    n2 = int(np.floor((frac_cal1 + frac_cal2) * len(idxs)))
    return perm[:n1], perm[n1:n2], perm[n2:]


def conformal_quantile(scores, alpha):
    """The ceil((n + 1)(1 - alpha))-th smallest score (the largest one if that exceeds n)."""
    s = np.sort(np.asarray(scores).reshape(-1))
    k = min(max(int(np.ceil((s.size + 1) * (1 - alpha))), 1), s.size)
    return float(s[k - 1])


def coverage(probs, targets, threshold, idx, scale=None):
    """Fraction of questions in idx whose prediction set {y: (1 - p(y)) / scale <= threshold} holds the
    true answer (scale = 1 for Original)."""
    scores = 1.0 - probs[idx]
    if scale is not None:
        scores = scores / np.maximum(scale[idx], EPS)[:, None]
    return float(np.mean(scores[np.arange(idx.size), targets[idx]] <= threshold))


class QuantileDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MultiQuantileNet(nn.Module):
    """Two hidden layers; one softplus output (> 0) per quantile level."""

    def __init__(self, input_dim, hidden_dim, n_quantiles):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, n_quantiles)
        self.softplus = nn.Softplus()

    def forward(self, x):
        return self.softplus(self.head(self.backbone(x)))


def pinball_loss(pred, target, q):
    e = target - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


def fit_quantile_net(X, y, quantiles, hidden_dim, n_epochs, lr, weight_decay, device):
    """Full-batch Adam on the pinball loss averaged over the quantile levels."""
    net = MultiQuantileNet(X.shape[1], hidden_dim, len(quantiles)).to(device)
    loader = DataLoader(QuantileDataset(X, y), batch_size=len(X), shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    q = torch.tensor(quantiles, dtype=torch.float32, device=device)
    for _ in range(n_epochs):
        net.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            preds = net(Xb)
            loss = 0.0
            for j in range(preds.shape[1]):
                loss = loss + pinball_loss(preds[:, j], yb, q[j])
            (loss / preds.shape[1]).backward()
            opt.step()
    return net


def run_mcqa(probs, targets, categories, emb, alphas, n_seeds, frac_cal1, frac_cal2, hidden_dim, n_epochs,
             lr, weight_decay, device="cpu"):
    """Coverage of Original and Adaptive, arrays (splits, alphas, subjects + overall)."""
    groups = np.sort(np.unique(categories))
    group_idx = {g: np.where(categories == g)[0] for g in groups}
    names = [SUBJECTS[g] for g in groups] + ["overall"]
    scores = 1.0 - probs[np.arange(len(targets)), targets]
    out = {m: np.full((n_seeds, len(alphas), len(names)), np.nan) for m in ("orig", "adaptive")}

    for seed in range(n_seeds):
        parts = [three_way_split(group_idx[g], frac_cal1, frac_cal2, seed) for g in groups if len(group_idx[g]) >= 3]
        cal1, cal2, test = (np.concatenate(p) for p in zip(*parts))

        scaler = StandardScaler()
        X_cal1 = scaler.fit_transform(emb[cal1])
        torch.manual_seed(seed)
        net = fit_quantile_net(X_cal1, scores[cal1], 1.0 - alphas, hidden_dim, n_epochs, lr, weight_decay, device)
        with torch.no_grad():
            qhat = net(torch.tensor(scaler.transform(emb), dtype=torch.float32, device=device)).cpu().numpy()
        qhat = np.maximum(qhat, EPS)  # (questions, alphas)

        for a, alpha in enumerate(alphas):
            t_orig = conformal_quantile(scores[cal2], alpha)
            t_adaptive = conformal_quantile(scores[cal2] / qhat[cal2, a], alpha)
            evals = [np.intersect1d(test, group_idx[g]) for g in groups] + [test]
            for gi, idx in enumerate(evals):
                if idx.size == 0:
                    continue
                out["orig"][seed, a, gi] = coverage(probs, targets, t_orig, idx)
                out["adaptive"][seed, a, gi] = coverage(probs, targets, t_adaptive, idx, scale=qhat[:, a])
    return names, out
