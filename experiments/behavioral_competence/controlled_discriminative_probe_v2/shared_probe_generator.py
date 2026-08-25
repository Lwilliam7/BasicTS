"""SharedControlledProbeGenerator: a window-ONLY conditioned perturbation
generator for the "one controlled question asked identically of every
frozen expert" experiment (controlled_discriminative_probe_v2).

This is a deliberately NEW module, not an edit of
`experiments/behavioral_competence/probe_generator.py`. LearnedProbe v1's
`ProbeGenerator.make_probe` takes a per-expert `forecast_summary` argument
and is applied to a `window_norm` tensor built from *that expert's own*
checkpoint scaler (`rt.mean`/`rt.std` in `run_learned_probe.py::run_batch`),
so two different experts scored on the same historical window in v1 receive
two different raw perturbations. That is the specific mechanism this new
experiment is designed to rule in or out (see module docstring of
`run_controlled_discriminative_probe_v2.py`), so `ProbeGenerator` itself is
frozen and left untouched; this class instead implements

    delta_t = G(X_t)          (never delta_t,e = G(X_t, forecast_summary_t,e))

by construction: its `forward`/`make_probe` signature has no expert-specific
input at all, and it is invoked exactly ONCE per window batch, before the
per-expert loop, in `run_controlled_discriminative_probe_v2.py`. The
identical `x_probe` tensor object is then reused for every expert's forward
pass, which makes
    x_probe[t, Expert A] == x_probe[t, Expert B] == x_probe[t, Expert C]
a structural guarantee rather than something that merely happens to hold
(verified anyway, empirically, as an integrity check).

The magnitude constraint is intentionally the SAME structural form as
`ProbeGenerator` (`probe_generator.py`, unmodified, EPS_DEFAULT=0.05):
    delta = eps * historical_std * tanh(raw_delta)
so |delta| <= eps * historical_std everywhere, with the same near-zero
mean-shift and temporal-smoothness soft penalties applied downstream via
the UNMODIFIED `perturbation_penalties` from `probe_generator.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS_DEFAULT = 0.05


class SharedControlledProbeGenerator(nn.Module):
    """Input: ONLY a canonical (expert-identity-independent) normalization of
    the current historical window. No expert identity, no per-expert
    forecast summary, no predicted error, no target. Architecturally a
    trimmed copy of `ProbeGenerator` (probe_generator.py) with the
    `forecast_proj`/expert-forecast-summary branch removed entirely -- not
    merely unused, but structurally absent, so there is no way for any
    expert-specific quantity to reach `raw_delta`."""

    def __init__(self, num_features: int, hidden: int = 32, eps: float = EPS_DEFAULT) -> None:
        super().__init__()
        self.eps = eps
        self.window_proj = nn.Linear(num_features, hidden)
        self.temporal_conv = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_features))

    def raw_delta(self, window_norm: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.window_proj(window_norm))  # [B,L,hidden]
        h = F.relu(self.temporal_conv(h.transpose(1, 2))).transpose(1, 2)  # [B,L,hidden]
        return self.head(h)  # [B,L,F] raw, unbounded

    def make_probe(self, history_raw: torch.Tensor, window_norm: torch.Tensor, hist_std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        delta_raw = self.raw_delta(window_norm)
        delta = self.eps * hist_std.unsqueeze(1) * torch.tanh(delta_raw)
        return history_raw + delta, delta


def canonical_window_norm(history_raw: torch.Tensor, canonical_std: torch.Tensor) -> torch.Tensor:
    """Expert-identity-independent normalization of the current window, used
    as the SharedControlledProbeGenerator's only input. Per-window mean
    centering (uses only values already inside this same causally-legal
    window -- no leakage) scaled by `canonical_std`, the dataset-level scaler
    std already threaded through every group A/B/C feature and every MAE/MSE
    computation in this experiment family (`Bundle.std`, sourced from a
    single fixed `final_60/DLinear/best_expert.pt` checkpoint per dataset in
    `generalization/run_generalization_study.py::load_new_bundle` --
    independent of which experts end up in the K=3 core, and identical for
    every expert queried). This IS the "common normalization shared by all
    experts" required by Section 7: it is reused, not reconstructed, because
    `Bundle.std` already plays exactly that role everywhere else in this
    project."""
    stdv = canonical_std.view(1, 1, -1).clamp_min(1e-8)
    return (history_raw - history_raw.mean(dim=1, keepdim=True)) / stdv


def precompute_shared_random_delta(history_raw_all: torch.Tensor, eps: float, seed: int) -> torch.Tensor:
    """Deterministic SharedRandomProbe control (Section 18B): ONE seeded,
    smooth, epsilon-bounded perturbation per window, generated for the ENTIRE
    split's history tensor at once (mirrors
    `common.py::perturb_noise`/`build_perturbation_cache`'s convention of
    generating from a fixed seed over the full raw-history array) so that a
    given absolute window always maps to the same delta regardless of how it
    is later batched -- required for the OOF-fold / final-deployment scoring
    passes to be reproducible. NOT target-dependent, NOT trained: this
    function has no learnable parameters and is never touched by an
    optimizer. Smoothed with the same 5-tap moving-average kernel used by
    `common.py::perturb_smooth`, then passed through the identical
    eps*std*tanh(.) magnitude constraint as every other perturbation in this
    experiment family, so RandomProbe and the learned probes share a
    comparable smoothness/magnitude budget (Section 38 diagnostic)."""
    hist_std = history_raw_all.std(dim=1, keepdim=True).clamp_min(1e-6)  # [N,1,F]
    gen = torch.Generator().manual_seed(seed)
    raw = torch.randn(history_raw_all.shape, generator=gen, dtype=torch.float32)
    n, length, feats = raw.shape
    x = raw.permute(0, 2, 1)  # [N,F,L]
    x_padded = F.pad(x, (2, 2), mode="replicate")
    kernel = torch.full((feats, 1, 5), 1.0 / 5, dtype=torch.float32)
    smoothed = F.conv1d(x_padded, kernel, groups=feats)[..., :length].permute(0, 2, 1)  # [N,L,F]
    delta = eps * hist_std * torch.tanh(smoothed)
    return delta
