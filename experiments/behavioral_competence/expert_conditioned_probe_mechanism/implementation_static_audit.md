# Static Audit: Expert-Conditioned Probe Mechanism

This audit was generated before running the expensive development experiment.

| Check | Status | Detail |
|---|---|---|
| experiment_directory | PASS | C:\Users\luwil\OneDrive\Documents\Code\BasicTS\experiments\behavioral_competence\expert_conditioned_probe_mechanism |
| device_selection | PASS | {"cuda_available": true, "cuda_version": "12.4", "device": "cuda", "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU", "torch_version": "2.6.0+cu124"} |
| test_access_policy | PASS | All loaders use router_train/router_val paths and fhv.refuse_test; runner has no test path construction except negative manifest text. |
| rank_weights | PASS | rule_fixed_rank gives [0.5, 0.3333333333333333, 0.16666666666666666] for K=3. |
| datasets | PASS | ETTh1,ETTh2,ETTm1,Weather,Electricity |
| purged_oof | PASS | N_PURGE_FOLDS=2, MIN_TRAIN_FRACTION=0.4, compute_legal_and_common reused from V2. |
| arm1 | PASS | C_Rank_Passive: 15 A+B+C features -> CompetenceScorer(15) -> fixed rank weights. |
| arm2 | PASS | MatchedNeuralPassive: ProbeGenerator trunk inputs (history_norm, Group-B forecast summary), no perturbation, no expert call, six learned z features + passive 15 -> CompetenceScorer(21). |
| arm3 | PASS | DeltaOnly: original ProbeGenerator produces expert-conditioned delta_k; no perturbed expert call; six predeclared delta statistics + passive 15 -> CompetenceScorer(21). |
| arm4 | PASS | OriginalLearnedProbe: original ProbeGenerator, expert-conditioned delta_k, frozen expert(x+delta_k), six probe_response_features + passive 15 -> CompetenceScorer(21). |
| training_match | PASS | seed=7, AdamW lr=0.001, weight_decay=0.0001, max_epochs=8, patience=3, batch=128, Huber+rank loss for all arms; perturb penalties only when a delta exists. |
| query_isolation | PASS | Only OriginalLearnedProbe calls rt.predict_differentiable(x+delta); DeltaOnly and MatchedNeuralPassive do not query experts after representation creation. |
| comparison_match | PASS | Every arm uses the same frozen K=3 core, target, normalization source, train folds, and final fixed rank combiner. |
