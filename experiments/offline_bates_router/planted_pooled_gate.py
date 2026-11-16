"""Split-half version: the spread of the true per-cell weights is the covariance, across cells, of the weights fitted
on the first and on the second half of the fitting windows. Noise in the two halves does not covary, so it drops out."""
import numpy as np

def ls(d, e):
    return (d * e).sum(0) / (d * d).sum(0)

def fit(d, e, gap):
    n = d.shape[0]; h = n // 2
    la, lb = ls(d[: h - gap // 2], e[: h - gap // 2]), ls(d[h + gap // 2:], e[h + gap // 2:])
    lam, S = ls(d, e), (d * d).sum(0)
    const = (d * e).sum() / (d * d).sum()
    tau2 = max(0.0, float(np.mean((la - la.mean()) * (lb - lb.mean()))))
    noise = max(float(np.mean((lam - const) ** 2)) - tau2, 0.0)            # mean sampling variance of the full fit
    s2 = noise * (1 / S) / np.mean(1 / S)                                   # spread over cells by their information
    shrink = tau2 / (tau2 + s2) if tau2 > 0 else np.zeros_like(s2)
    return np.clip(const, 0, 1), np.clip(lam, 0, 1), np.clip(const + shrink * (lam - const), 0, 1), tau2, float(np.mean(shrink))

def ar(n, c, rho, rng):
    x = np.zeros((n, c)); z = rng.standard_normal((n, c))
    for t in range(1, n):
        x[t] = rho * x[t - 1] + z[t]
    return x

def run(name, C, T, tau, noise, rho=0.9, gap=48, reps=5, hetero=False):
    out = []
    for r in range(reps):
        rng = np.random.default_rng(r)
        lam_true = np.clip(0.5 + tau * rng.standard_normal(C), 0, 1)
        scale = np.exp(rng.standard_normal(C)) if hetero else 1.0          # some cells carry far more signal
        d, dv = scale * ar(T, C, rho, rng), scale * ar(T, C, rho, rng)
        e = lam_true * d + noise * ar(T, C, rho, rng)
        ev = lam_true * dv + noise * ar(T, C, rho, rng)
        const, cell, eb, tau2, sh = fit(d, e, gap)
        loss = lambda lam: np.abs(ev - lam * dv).mean()
        base = loss(lam_true)
        out.append([100 * (loss(x) / base - 1) for x in (const, cell, eb)] + [tau2 ** 0.5, sh])
    m = np.mean(out, 0)
    print(f"{name:36s} constant {m[0]:+.3f}%  per-cell {m[1]:+.3f}%  pooled {m[2]:+.3f}%   est. spread {m[3]:.3f} (true {tau}), mean keep {m[4]:.2f}")

run("many noisy cells, equal weights", 2000, 600, 0.0, 6.0)
run("many noisy cells, mild spread",   2000, 600, 0.1, 6.0)
run("many cells, wide spread, noisy",  2000, 600, 0.3, 6.0)
run("few clean cells, wide spread",    60,   3000, 0.3, 1.0)
run("many clean cells, no spread",     2000, 3000, 0.0, 1.0)
run("uneven cells, mild spread, noisy", 2000, 600, 0.15, 4.0, hetero=True)
run("uneven cells, wide spread",       2000, 600, 0.3, 2.0, hetero=True)
