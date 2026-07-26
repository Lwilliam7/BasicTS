# Architecture Pseudocode

## Old COSTARTS

```text
input history x [B,96,7]
encode x into one window embedding
predict:
  query logits [B,M]
  stop logits [B,K]
  expert error map [B,M]
choose a fixed query order once
choose final expert from queried prefix using predicted error
no queried forecast is fed back into the state
```

## Improved Subset-Utility COSTARTS

```text
offline:
  for each window and subset S of queried experts:
    store history, target, queried mask, queried forecasts
    compute true expert errors and marginal utility for each unqueried expert
    label optimal next action as QUERY expert or STOP

training:
  encode history [B,96,7]
  encode queried mask [B,M]
  encode queried forecasts [B,|S|,12,7]
  fuse representations
  predict:
    action logits [B,M+1]
    utility map [B,M]
    queried-subset scores [B,M]
    sparse mix logits [B,M]
  optimize action, utility, pairwise ranking, and optional mix losses

inference:
  start with S = empty
  repeat:
    score QUERY actions and STOP
    query selected expert only
    update S and reveal its forecast
  finalize with equal average of queried expert forecasts
```
