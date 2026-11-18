"""Unweighted against information-weighted moments for the pooled gate, with cells that carry almost no signal."""
import numpy as np

def ls(d, e):
    return (d * e).sum(0) / (d * d).sum(0)

def fit(d, e, gap, weighted):
    n = d.shape[0]; h = n // 2
    la, lb = ls(d[: h - gap // 2], e[: h - gap // 2]), ls(d[h + gap - gap // 2:], e[h + gap - gap // 2:])
    lam, S = ls(d, e), (d * d).sum(0)
    const = (d * e).sum() / S.sum()
    w = S / S.sum() if weighted else np.full(S.size, 1 / S.size)
    tau2 = max(0.0, float((w * (la - (w * la).sum()) * (lb - (w * lb).sum())).sum()))
    noise = max(float((w * (lam - const) ** 2).sum()) - tau2, 0.0)
    s2 = noise * S.mean() / S if weighted else noise * (1 / S) / np.mean(1 / S)
    keep = tau2 / (tau2 + s2) if tau2 > 0 else np.zeros_like(S)
    return np.clip(const, 0, 1), np.clip(lam, 0, 1), np.clip(const + keep * (lam - const), 0, 1)

def ar(n, c, rho, rng):
    x = np.zeros((n, c)); z = rng.standard_normal((n, c))
    for t in range(1, n):
        x[t] = rho * x[t - 1] + z[t]
    return x

def run(name, C, T, tau, noise, scale_fn, reps=5):
    out = []
    for r in range(reps):
        rng = np.random.default_rng(r)
        lam_true = np.clip(0.5 + tau * rng.standard_normal(C), 0, 1)
        scale = scale_fn(rng, C)
        d, dv = scale * ar(T, C, 0.9, rng), scale * ar(T, C, 0.9, rng)
        e, ev = lam_true * d + noise * ar(T, C, 0.9, rng), lam_true * dv + noise * ar(T, C, 0.9, rng)
        base = np.abs(ev - lam_true * dv).mean()
        row = []
        for weighted in (False, True):
            const, cell, eb = fit(d, e, 48, weighted)
            row += [100 * (np.abs(ev - x * dv).mean() / base - 1) for x in (const, cell, eb)]
        out.append(row)
    m = np.mean(out, 0)
    print(f"{name:40s} constant {m[0]:+.3f}  per-cell {m[1]:+.3f}  pooled unweighted {m[2]:+.3f}  pooled weighted {m[5]:+.3f}")

even = lambda rng, C: 1.0
uneven = lambda rng, C: np.exp(rng.standard_normal(C))
dead = lambda rng, C: np.where(rng.random(C) < 0.3, 0.02, 1.0) * np.exp(0.5 * rng.standard_normal(C))
for name, C, T, tau, noise, fn in [
    ("even cells, equal weights, noisy", 2000, 600, 0.0, 6.0, even), ("even cells, wide spread, clean", 400, 3000, 0.3, 1.0, even),
    ("uneven cells, mild spread, noisy", 2000, 600, 0.15, 4.0, uneven), ("uneven cells, wide spread", 2000, 600, 0.3, 2.0, uneven),
    ("30% near-empty cells, equal weights", 2000, 600, 0.0, 3.0, dead), ("30% near-empty cells, wide spread", 2000, 600, 0.3, 1.0, dead),
    ("30% near-empty cells, mild, noisy", 2000, 600, 0.1, 4.0, dead)]:
    run(name, C, T, tau, noise, fn)
