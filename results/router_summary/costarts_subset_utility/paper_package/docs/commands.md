# Reproduction Commands

## Build subset-state caches

```powershell
python scripts\build_costarts_subset_states.py --split both --subset-sampling-mode exhaustive --force --print-examples 3
```

## Train improved subset-utility COSTARTS

```powershell
python scripts\train_costarts_subset_utility_router.py --device cpu --max-epochs 50 --patience 10 --batch-size 1024
```

## Sequential rollout evaluation

```powershell
python scripts\evaluate_costarts_subset_utility_rollouts.py --device cpu --mode all --finalizer both --temperature 1.0 --detailed-limit 25
```

## Cost sweep

```powershell
python scripts\evaluate_costarts_cost_sweep.py --device cpu --batch-size 1024
```

## Final comparison

```powershell
python scripts\evaluate_costarts_final_comparison.py --device cpu --batch-size 1024
```

## Ablations

```powershell
python scripts\run_costarts_subset_utility_ablations.py --device cpu --max-epochs 50 --patience 10 --batch-size 1024
```

## Paper package

```powershell
python scripts\build_costarts_paper_package.py --device cpu --batch-size 1024
```
