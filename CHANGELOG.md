# Changelog

## 0.1.0 — 2026-11-23

- Train actual dense numeric ridge regression on CPU, fitting population feature scaling and target mean on training rows only.
- Evaluate separate validation against the trained model and training-mean baseline with MAE/RMSE and nullable/negative R².
- Save a strict inspectable JSON model and reload it for ordered ID-preserving batch prediction; include a standalone BCL-only C# scorer.
- Validate bounded JSONL, exact numeric feature schemas, unique/disjoint IDs, finite values/model arrays and recursive duplicate properties.
- Preserve source bytes and refuse existing artifacts; record actual source/model/output hashes, with tests, synthetic examples/docs and three-platform installed-tool CI.

Saved-model holdout `evaluate` and optional RMSE gating are fixed future work in the [roadmap](ROADMAP.md), not implemented CLI options. No neural network, categorical preprocessing or automatic lambda search is provided.
