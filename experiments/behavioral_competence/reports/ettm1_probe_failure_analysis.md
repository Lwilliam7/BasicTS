# Why is LearnedProbe-Rank Worse Than C-Rank on ETTm1?

Diagnostic-only investigation. No model retrained, no hyperparameter changed, no test data accessed. All analysis is on already-saved router_val predictions/competence scores plus one no-grad inference pass through the already-trained frozen experts (to reconstruct the probe's forecast response, which was not itself saved -- only the input-space perturbation was).

Reproduction check: C-Rank MAE = 0.249199 (known ~0.249199), LearnedProbe-Rank MAE = 0.249857 (known ~0.249857). Both match.

## 1. Per-window failure decomposition

- Fraction improved: 0.284, hurt: 0.325, tied (identical ranking -> identical prediction): 0.391
- Mean delta: +0.000658, median: +0.000000
- P90/P95/P99 harmful delta: 0.013376 / 0.021049 / 0.038970
- **Top-10 worst windows account for only 2.0% of total harmful delta** -- this is diffuse, not catastrophic.

**Verdict: (A) many small losses, not (B) a few catastrophic failures.**

## 2. Rank-transition categories

| Category | Windows | C-Rank MAE | LearnedProbe-Rank MAE | Mean paired delta |
|---|---:|---:|---:|---:|
| A_identical | 4460 | 0.263384 | 0.263384 | `+0.000000` |
| B_top2_swap | 2174 | 0.257535 | 0.257587 | `+0.000052` |
| C_bottom2_swap | 1855 | 0.241834 | 0.242390 | `+0.000556` |
| D_best_flips_to_worst | 1721 | 0.226434 | 0.228522 | `+0.002088` |
| E_other_reorder | 1203 | 0.225467 | 0.227776 | `+0.002309` |

Categories D (best flips to C's worst) + E (other multi-position reorder) are only 2924/11413 = 25.6% of windows, but contribute 84.8% of the total positive (harmful) delta mass. Simple adjacent swaps (B, C) are nearly harmless.

**Verdict: the regression is concentrated in the most drastic ranking changes, not simple adjacent reordering.**

## 3. True expert-rank analysis

| Scorer | Top-1 acc | Top-2 recall | Mean rank of true best | Mean pairwise accuracy |
|---|---:|---:|---:|---:|
| C | 0.353 | 0.669 | 1.978 | 0.552 |
| LearnedProbe | 0.326 | 0.640 | 2.033 | 0.558 |

LearnedProbe's **overall pairwise accuracy is slightly higher** (0.558 vs 0.552 -- consistent with its better Spearman) but its **top-1 accuracy and top-2 recall are both lower**. On 315 windows (2.8% of all windows), LearnedProbe's pairwise accuracy is strictly better than C's yet the final forecast is worse (mean delta in that group: +0.002257).

**This is the direct mechanism for 'better Spearman, worse MAE': LearnedProbe gets more of the low-stakes pairwise comparisons right (e.g. correctly ranking the 2nd- and 3rd-place experts) while getting the highest-stakes comparison -- who is #1 -- wrong more often than C.**

## 4. Cost-weighted ranking errors

| Scorer | # mistakes | Fraction | Total cost | Mean cost/mistake |
|---|---:|---:|---:|---:|
| C | 7382 | 0.647 | 224.907 | 0.030467 |
| LearnedProbe | 7688 | 0.674 | 242.731 | 0.031573 |

**LearnedProbe does NOT make fewer-but-costlier mistakes on ETTm1 -- it makes MORE mistakes (67.4% vs 64.7%), each on average also slightly more expensive.** The 'fewer but more expensive' hypothesis does not hold here.

## 5. Expert-specific analysis

| Expert | Actual mean MAE | C rank-1 % | Learned rank-1 % | MAE\|C ranks 1st | MAE\|Learned ranks 1st |
|---|---:|---:|---:|---:|---:|
| DLinear | 0.2643 | 0.373 | 0.416 | 0.2764 | 0.2694 |
| PatchTST | 0.2618 | 0.378 | 0.493 | 0.2573 | 0.2479 |
| TimesNet | 0.2699 | 0.250 | 0.091 | 0.2344 | 0.2664 |

**TimesNet is severely under-promoted**: C ranks it 1st on 25% of windows, LearnedProbe only 9%. When LearnedProbe *does* rank TimesNet 1st, the conditional MAE is 0.2664 -- worse than when C ranks it 1st (0.2344), and close to TimesNet's unconditional average (0.2699). C's confidence in TimesNet is much better calibrated than LearnedProbe's.

## 6. Expert-separation analysis

| Feature | Bin | Windows | C-Rank MAE | LearnedProbe-Rank MAE | Delta |
|---|---|---:|---:|---:|---:|
| separation_best_second | low | 3805 | 0.212784 | 0.212584 | `-0.000201` |
| separation_best_second | medium | 3804 | 0.230885 | 0.231353 | `+0.000468` |
| separation_best_second | high | 3804 | 0.303937 | 0.305645 | `+0.001708` |
| separation_second_third | low | 3805 | 0.224800 | 0.226326 | `+0.001526` |
| separation_second_third | medium | 3804 | 0.242666 | 0.243281 | `+0.000615` |
| separation_second_third | high | 3804 | 0.280137 | 0.279972 | `-0.000166` |
| forecast_disagreement | low | 3805 | 0.195687 | 0.196600 | `+0.000913` |
| forecast_disagreement | medium | 3804 | 0.254094 | 0.254455 | `+0.000361` |
| forecast_disagreement | high | 3804 | 0.297830 | 0.298531 | `+0.000701` |

The 'harm concentrates when experts are nearly equivalent' hypothesis is **NOT supported** -- it is close to the opposite. Delta is smallest/slightly favorable when best-vs-second separation is low (-0.000201) and largest when separation is high (+0.001708). LearnedProbe's mistakes are *worse specifically when the stakes are highest*, not when experts are interchangeable. Forecast-disagreement bins show a less clean, non-monotonic pattern (see table) and do not tell a consistent story on their own.

## 7. Probe-response analysis (beneficial vs tied vs harmful windows)

| Group | Windows | Mean magnitude | Early energy | Late energy | Mean change | Mean cosine change |
|---|---:|---:|---:|---:|---:|---:|
| beneficial | 3245 | 0.001872 | 0.001794 | 0.001949 | 0.001220 | 5.41e-07 |
| neutral | 4460 | 0.002048 | 0.001960 | 0.002135 | 0.001337 | 6.14e-07 |
| harmful | 3708 | 0.001930 | 0.001864 | 0.001995 | 0.001242 | 5.20e-07 |

No distinct probe-response signature separates beneficial from harmful windows on ETTm1 -- magnitude, energy location, and forecast-response statistics are nearly identical between the two groups. **The failure is not explained by the probe behaving differently on harmful windows; it is explained by what the competence scorer does with a similar-looking probe response.**

## 8. Time-series regime analysis

| Group | Windows | Disagreement | Trend | Volatility | Lag-1 autocorr | Spectral entropy |
|---|---:|---:|---:|---:|---:|---:|
| beneficial | 3245 | 0.0827 | 0.4070 | 0.6022 | 0.8762 | 0.3893 |
| neutral | 4460 | 0.0984 | 0.4088 | 0.6548 | 0.8932 | 0.3945 |
| harmful | 3708 | 0.0825 | 0.4011 | 0.6089 | 0.8745 | 0.3882 |

Regime features are also nearly identical between beneficial and harmful windows -- no recognizable forecasting regime (trend, volatility, autocorrelation, entropy) distinguishes where the probe helps vs hurts on ETTm1.

## 9. Horizon and variable decomposition

**By horizon** (regret = LearnedProbe-Rank minus C-Rank, normalized per-location error):

| Horizon | Mean regret |
|---:|---:|
| 0 | `+0.000053` |
| 1 | `+0.000653` |
| 2 | `+0.000682` |
| 3 | `+0.000508` |
| 4 | `+0.000969` |
| 5 | `+0.000688` |
| 6 | `+0.000831` |
| 7 | `+0.000703` |
| 8 | `+0.000639` |
| 9 | `+0.000744` |
| 10 | `+0.000822` |
| 11 | `+0.000609` |

**By variable:**

| Variable | Mean regret |
|---:|---:|
| 0 | `+0.001199` |
| 1 | `+0.000922` |
| 2 | `+0.000788` |
| 3 | `+0.000642` |
| 4 | `+0.000857` |
| 5 | `+0.000045` |
| 6 | `+0.000156` |

Regret is positive on 12/12 horizons and 7/7 variables -- **broadly distributed, not concentrated in one horizon or one variable.** Worst single horizon: 4 (+0.000969). Worst single variable: 0 (+0.001199). Full horizon x variable grid in `ettm1_probe_failure_horizon_variable.csv`.

## 10. Is the probe itself the problem? (C vs Fixed-D vs LearnedProbe)

- Fixed-D changes C's top-1 pick on 55.1% of windows (mean cost when it does: +0.000692).
- LearnedProbe changes C's top-1 pick on 44.7% of windows -- **less often** than Fixed-D, but at **higher average cost** when it does (+0.001272 vs Fixed-D's +0.000692).
- Mean confidence margin (predicted 2nd-best minus best): C=0.0119, Fixed-D=0.0139, **LearnedProbe=0.0220** -- nearly 2x C's margin.

**LearnedProbe is the most confident of the three scorers on ETTm1, despite having the worst top-1 accuracy there. This is a genuine 'confidently wrong' signature, not shared by Fixed-D.**

## 11. Counterfactual diagnostics (unattainable at forecast time -- diagnostic only)

- 5098 windows (44.7%) where C and LearnedProbe disagree on the top-1 pick.
- On those disagreement windows: C-Rank MAE = 0.239469, LearnedProbe-Rank MAE = 0.240741 -- C wins there too.
- Retrospective, unattainable oracle (best of the two, per window, using targets): 0.246185 -- meaningfully below both, showing real per-window heterogeneity that neither method exploits, but this is not a deployable rule.

## 12. Cross-dataset comparison (ETTh2, Weather, Electricity vs ETTm1)

| Dataset | Frac. hurt | Mean delta | C top-1 acc | Learned top-1 acc | Frac. Learned changes C top-1 | Margin C | Margin Learned |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTm1 | 0.325 | `+0.000658` | 0.353 | 0.326 | 44.7% | 0.0119 | 0.0220 |
| ETTh2 | 0.297 | `-0.003767` | 0.352 | 0.460 | 68.4% | 0.0367 | 0.0240 |
| Weather | 0.238 | `-0.000587` | 0.399 | 0.444 | 41.0% | 0.0072 | 0.0210 |
| Electricity | 0.121 | `-0.001991` | 0.545 | 0.629 | 30.8% | 0.0102 | 0.0216 |

**The key structural difference**: on every successful dataset, LearnedProbe's top-1 accuracy is *higher* than C's; on ETTm1 alone, it is *lower*. The confidence-margin pattern (LearnedProbe more confident than C) is consistent across ALL datasets, including the successful ones -- so higher confidence alone does not explain the ETTm1 failure. What's different on ETTm1 is that the increased confidence is attached to a *worse* ranking rather than a better one. See `ettm1_probe_failure_cross_dataset.csv` for full detail.

## Final answers

**1. Many small errors or a few large failures?** Many small losses -- the worst 10 windows account for only 2.0% of total harm.
**2. Which rank changes drive the regression?** The most drastic reorderings (best flips to C's worst-ranked expert, or other multi-position reorders) -- 25.6% of windows but 84.8% of total harm. Simple adjacent swaps are nearly harmless.
**3. Fewer but costlier mistakes?** No -- LearnedProbe makes *more* top-1 mistakes than C on ETTm1, each on average also slightly more expensive.
**4. Is one expert over/under-promoted?** Yes -- TimesNet is severely under-promoted (25%->9% rank-1 rate), and when it is picked, LearnedProbe's picks are worse-conditioned than C's.
**5. Does harm concentrate at low expert separation?** No -- if anything the opposite: harm is largest when the best-vs-second gap is largest, i.e. when it matters most.
**6. Concentrated by variable/horizon?** No -- regret is positive on 12/12 horizons and 7/7 variables; broadly distributed, not localized to one cell.
**7. Distinct probe-response signature on harmful windows?** No -- probe magnitude, energy location, and response statistics are nearly identical between beneficial and harmful windows.
**8. Why can Spearman improve while MAE worsens?** Because Spearman/pairwise-accuracy rewards getting the *2nd vs 3rd place* comparison right, which the rank-weighting rule barely rewards (0.30 vs 0.10), while the *1st place* call -- the one that matters most for the 0.60 weight -- is where LearnedProbe is specifically worse on ETTm1.
**9. General weakness or ETTm1-specific?** The mechanism (aggregate ranking metrics not tracking top-1 accuracy) is general and could recur elsewhere, but the specific manifestation here -- TimesNet under-promotion, confidently-wrong top-1 calls -- looks tied to how this dataset's three experts (DLinear/PatchTST/TimesNet) actually perform, which the successful datasets' expert sets don't share.
**10. General research problem or isolated failure?** A bit of both, and worth stating precisely: the *diagnostic gap* between aggregate ranking correlation and top-1/rank-weighted forecasting quality is a real, general phenomenon that this analysis exposed cleanly -- Spearman is not a reliable proxy for how a rank-weighted ensemble will perform, because it doesn't weight the top comparison specially. But the specific *failure mode on ETTm1* (TimesNet under-promotion, larger-but-wrong confidence) has not been shown to generalize to other datasets in this analysis -- ETTh2/Weather/Electricity all show the *opposite* top-1 pattern (LearnedProbe's top-1 accuracy exceeds C's there). So: the measurement gap is general; the ETTm1 outcome itself looks dataset-specific.

No fix is proposed here, per instructions -- this is diagnosis only.

## Hard rule compliance

```text
TEST SET ACCESSED: NO
TEST CACHE LOADED: NO
TEST METRICS COMPUTED: NO
```
