"""Boosted baseline: conditional boosting of claim scores (Cherian et al., 2024).

The boosting objective follows their medical-QA notebook (conformal-safety repository); the final
group-conditional calibration uses their conditionalconformal.CondConf.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linprog
from sklearn.preprocessing import StandardScaler

from longform_data import CATEGORIES, INCORRECT_LABEL


ScoreKey = Tuple[str, str]


@dataclass(frozen=True)
class ScoreSpec:
    key: ScoreKey
    # All boosted columns are oriented so larger means more likely correct.
    higher_raw_is_incorrect: bool


DEFAULT_SCORE_SPECS: Tuple[ScoreSpec, ...] = (
    ScoreSpec(("claim", "MaximumClaimProbability"), True),
    ScoreSpec(("claim", "PerplexityClaim"), True),
    ScoreSpec(("claim", "MaxTokenEntropyClaim"), True),
    ScoreSpec(("claim", "PointwiseMutualInformationClaim"), False),
    ScoreSpec(("claim", "CCP_claim_no_context"), True),
    ScoreSpec(("claim", "CCP_claim_fact_pref"), True),
    ScoreSpec(("claim", "FrequencyScoringClaim"), True),
    ScoreSpec(("claim", "TokenSARClaim"), True),
)


class ClaimFeatureTransformer:
    """Orient and standardize candidate claim scores using cal1 only."""

    def __init__(
        self,
        specs: Sequence[ScoreSpec] = DEFAULT_SCORE_SPECS,
        clip: float = 10.0,
        transform: str = "zscore",
    ):
        self.specs = tuple(specs)
        self.clip = float(clip)
        if transform not in {"zscore", "confidence_rank"}:
            raise ValueError("transform must be 'zscore' or 'confidence_rank'")
        self.transform_name = transform

    def fit(self, data: Sequence[dict], indices: Sequence[int]):
        columns = [[] for _ in self.specs]
        for i in indices:
            for j, spec in enumerate(self.specs):
                if spec.key not in data[i]["unc_scores"]:
                    raise KeyError(f"Missing boosting score {spec.key!r}")
                columns[j].append(np.asarray(data[i]["unc_scores"][spec.key], dtype=float))
        concatenated = [np.concatenate(x) for x in columns]
        self.means_ = np.asarray([np.nanmean(x) for x in concatenated])
        self.scales_ = np.asarray([np.nanstd(x) for x in concatenated])
        self.scales_[~np.isfinite(self.scales_) | (self.scales_ < 1e-12)] = 1.0
        self.oriented_sorted_ = []
        for values, spec in zip(concatenated, self.specs):
            oriented = -values if spec.higher_raw_is_incorrect else values
            self.oriented_sorted_.append(np.sort(oriented[np.isfinite(oriented)]))
        return self

    def transform_prompt(self, prompt: dict) -> np.ndarray:
        cols = []
        for j, spec in enumerate(self.specs):
            raw = np.asarray(prompt["unc_scores"][spec.key], dtype=float)
            if self.transform_name == "zscore":
                values = (raw - self.means_[j]) / self.scales_[j]
                if spec.higher_raw_is_incorrect:
                    values = -values
                values = np.nan_to_num(
                    values, nan=0.0, posinf=self.clip, neginf=-self.clip,
                )
                values = np.clip(values, -self.clip, self.clip)
            else:
                oriented = -raw if spec.higher_raw_is_incorrect else raw
                reference = self.oriented_sorted_[j]
                values = np.searchsorted(reference, oriented, side="right") / (len(reference) + 1)
                values = np.nan_to_num(values, nan=0.5, posinf=1.0, neginf=0.0)
                values = np.clip(values, 0.0, 1.0)
            cols.append(values)
        return np.column_stack(cols)

    def transform(self, data: Sequence[dict]) -> List[np.ndarray]:
        return [self.transform_prompt(d) for d in data]


class PromptFeatureTransformer:
    """Cherian-style prompt metadata plus category indicators.

    Continuous columns are prompt length, claim count, total claim length, and
    the mean/std of the baseline CCP score.  Seven category indicators plus an
    intercept encode all eight reported groups without collinearity.
    """

    baseline_key: ScoreKey = ("claim", "CCP_claim_fact_pref")

    def _raw(self, prompt: dict) -> np.ndarray:
        scores = np.asarray(prompt["unc_scores"][self.baseline_key], dtype=float)
        claims = prompt["claims"]
        continuous = np.asarray([
            np.log1p(len(str(prompt["prompt"]))),
            np.log1p(len(claims)),
            np.log1p(sum(len(str(x)) for x in claims)),
            np.nanmean(scores),
            np.nanstd(scores),
        ])
        category = np.zeros(len(CATEGORIES) - 1)
        if int(prompt["category"]) < len(category):
            category[int(prompt["category"])] = 1.0
        return np.concatenate((continuous, category))

    def fit(self, data: Sequence[dict], indices: Sequence[int]):
        raw = np.vstack([self._raw(data[i]) for i in indices])
        self.scaler_ = StandardScaler().fit(raw[:, :5])
        return self

    def transform(self, data: Sequence[dict]) -> np.ndarray:
        raw = np.vstack([self._raw(d) for d in data])
        continuous = self.scaler_.transform(raw[:, :5])
        return np.column_stack((np.ones(len(data)), continuous, raw[:, 5:]))


def prompt_error_scores(
    claim_features: Sequence[np.ndarray], labels: Sequence[np.ndarray], beta: torch.Tensor,
) -> torch.Tensor:
    """Maximum boosted confidence of an incorrect claim; -1 if none exist."""
    result = []
    for features, prompt_labels in zip(claim_features, labels):
        scores = torch.abs(torch.as_tensor(features, dtype=beta.dtype) @ beta)
        incorrect = torch.as_tensor(np.asarray(prompt_labels) == INCORRECT_LABEL)
        result.append(scores[incorrect].max() if incorrect.any() else beta.new_tensor(-1.0))
    return torch.stack(result)


def _notebook_basis(
    phi: np.ndarray,
    scores: np.ndarray,
    tau: np.ndarray,
    duals: np.ndarray,
    quantile: float,
) -> np.ndarray:
    """Match get_current_basis in the authors' released notebooks."""
    basis = np.logical_and(
        ~np.isclose(duals, quantile - 1.0),
        ~np.isclose(duals, quantile),
    )
    if basis.sum() == phi.shape[1]:
        return np.where(basis)[0]
    predictions = (phi @ tau).flatten()
    active_indices = np.where(basis)[0]
    interpolated = np.argsort(np.abs(scores - predictions))[:phi.shape[1]]
    candidates = np.setdiff1d(interpolated, active_indices)
    missing = phi.shape[1] - basis.sum()
    if missing < len(candidates):
        for candidate_indices in combinations(candidates, missing):
            candidate_phi = phi[np.concatenate((active_indices, candidate_indices))]
            if np.isfinite(np.linalg.cond(candidate_phi)):
                basis[np.asarray(candidate_indices)] = True
                break
    else:
        basis[candidates] = True
    if basis.sum() == phi.shape[1]:
        indices = np.where(basis)[0]
        if (
            np.linalg.matrix_rank(phi[indices], tol=1e-9) == phi.shape[1]
            and np.linalg.cond(phi[indices]) < 1e10
        ):
            return indices

    # The released medical-QA notebook catches any singular-basis exception
    # and silently uses a zero threshold. The paper instead recommends score or
    # design dithering. A deterministic rank-revealing fallback avoids silently
    # zeroing the gradient while preserving the notebook basis whenever stable.
    active = np.where(np.logical_and(
        ~np.isclose(duals, quantile - 1.0),
        ~np.isclose(duals, quantile),
    ))[0]
    residual_order = np.argsort(np.abs(scores - predictions))
    chosen: List[int] = []
    rank = 0
    for idx in np.concatenate((active, residual_order)):
        if int(idx) in chosen:
            continue
        candidate = chosen + [int(idx)]
        new_rank = np.linalg.matrix_rank(phi[candidate], tol=1e-9)
        if new_rank > rank:
            chosen.append(int(idx))
            rank = new_rank
        if rank == phi.shape[1]:
            return np.asarray(chosen)
    raise RuntimeError("Could not find a stable full-rank quantile-regression basis")


def differentiable_quantile_regression(
    phi: np.ndarray, scores: torch.Tensor, quantile: float,
) -> torch.Tensor:
    """LP basis selection followed by the differentiable fit used by Cherian et al."""
    detached = scores.detach().cpu().numpy()
    result = linprog(
        -detached,
        A_eq=phi.T,
        b_eq=np.zeros(phi.shape[1]),
        bounds=[(quantile - 1.0, quantile)] * len(detached),
        method="highs-ds",
        options={"presolve": False},
    )
    if not result.success:
        raise RuntimeError(f"Conditional quantile LP failed: {result.message}")
    tau_lp = -np.asarray(result.eqlin.marginals)
    basis = _notebook_basis(
        phi, detached, tau_lp, np.asarray(result.x), quantile,
    )
    left_inverse = np.linalg.inv(phi[basis].T @ phi[basis]) @ phi[basis].T
    return torch.as_tensor(left_inverse, dtype=scores.dtype) @ scores[basis]


def train_conditional_boosting(
    claim_features: Sequence[np.ndarray], labels: Sequence[np.ndarray],
    phi: np.ndarray, indices: Sequence[int], alpha: float, seed: int,
    steps: int = 1000, lr: float = 1e-3, train_fraction: float = 0.8,
) -> np.ndarray:
    """Optimize the fixed-alpha conditional-boosting efficiency objective."""
    indices = np.asarray(indices)
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie strictly between zero and one")
    beta = torch.full((claim_features[0].shape[1],), 0.25, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([beta], lr=lr)
    rng = np.random.default_rng(seed)

    for _ in range(steps):
        perm = rng.permutation(indices)
        n_train = max(phi.shape[1] + 1, int(np.floor(train_fraction * len(perm))))
        n_train = min(n_train, len(perm) - 1)
        train_idx, holdout_idx = perm[:n_train], perm[n_train:]

        optimizer.zero_grad()
        train_scores = prompt_error_scores(
            [claim_features[i] for i in train_idx], [labels[i] for i in train_idx], beta,
        )
        tau = differentiable_quantile_regression(phi[train_idx], train_scores, 1.0 - alpha)
        cutoffs = torch.as_tensor(phi[holdout_idx], dtype=beta.dtype) @ tau

        prompt_losses = []
        for i, cutoff in zip(holdout_idx, cutoffs):
            scores = torch.abs(torch.as_tensor(claim_features[i], dtype=beta.dtype) @ beta)
            prompt_losses.append(torch.sigmoid(cutoff - scores).mean())
        loss = torch.stack(prompt_losses).mean()
        loss.backward()
        optimizer.step()

    return beta.detach().cpu().numpy()


def boosted_claim_scores(claim_features: Sequence[np.ndarray], beta: np.ndarray) -> List[np.ndarray]:
    return [np.abs(features @ beta) for features in claim_features]


def boosted_on_split(
    data: Sequence[dict], alpha: float, seed: int,
    cal1_idx: np.ndarray, cal2_idx: np.ndarray, test_idx: np.ndarray,
    boost_steps: int = 1000, boost_lr: float = 1e-3,
    boost_train_fraction: float = 0.8, jitter: float = 1e-6,
    score_transform: str = "confidence_rank",
) -> List[np.ndarray]:
    """Boost on cal1, conditionally calibrate on cal2; return kept masks for test prompts."""
    from conditionalconformal import CondConf

    labels = [np.asarray(d["labels"]) for d in data]
    claim_transformer = ClaimFeatureTransformer(transform=score_transform).fit(data, cal1_idx)
    claim_features = claim_transformer.transform(data)
    phi = PromptFeatureTransformer().fit(data, cal1_idx).transform(data)

    beta = train_conditional_boosting(
        claim_features, labels, phi, cal1_idx, alpha, seed,
        steps=boost_steps, lr=boost_lr, train_fraction=boost_train_fraction,
    )
    scores = boosted_claim_scores(claim_features, beta)
    if jitter:
        jitter_rng = np.random.default_rng(seed + 10_000)
        scores = [x + jitter_rng.uniform(0.0, jitter, size=x.shape) for x in scores]

    conformity = np.asarray([
        np.max(scores[i][labels[i] == INCORRECT_LABEL])
        if np.any(labels[i] == INCORRECT_LABEL) else -1.0
        for i in cal2_idx
    ])
    condconf = CondConf(score_fn=lambda x, y: y, Phi_fn=lambda x: x, seed=seed)
    condconf.setup_problem(phi[cal2_idx], conformity)

    kept = []
    for i in test_idx:
        def predict(exact):
            return float(np.asarray(condconf.predict(
                quantile=1.0 - alpha, x_test=phi[i:i + 1], score_inv_fn=lambda value, _: value,
                randomize=True, exact=exact)).squeeze())
        try:
            cutoff = predict(exact=True)
        except np.linalg.LinAlgError:
            # CondConf's exact path can hit a singular basis (in one Gemma split, for every test
            # prompt); its LP solution gives the cutoff instead.
            cutoff = predict(exact=False)
        kept.append(scores[i] > cutoff)
    return kept
