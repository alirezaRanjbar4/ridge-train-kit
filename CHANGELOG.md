# Changelog

## 0.2.0 — Unreleased

- Evaluate a saved schema-1 model on labeled numeric JSONL without refitting or writing files. Report model/input byte hashes, MAE/RMSE and nullable/negative R².
- Add optional finite nonnegative `--max-rmse` gate: strictly greater fails with exit 1; equality passes. Report-only evaluation exits 0; invalid inputs exit 2 without partial JSON.
- Add a synthetic holdout, hand-derived metric/gate boundary tests and [holdout contract](docs/HOLDOUT.md). The validated closing release remains pending.

## 0.1.0 — 2026-11-23

- Train actual dense numeric ridge regression on CPU, fitting population feature scaling and target mean on training rows only.
- Evaluate separate validation against the trained model and training-mean baseline with MAE/RMSE and nullable/negative R².
- Save a strict inspectable JSON model and reload it for ordered ID-preserving batch prediction; include a standalone BCL-only C# scorer.
- Validate bounded JSONL, exact numeric feature schemas, unique/disjoint IDs, finite values/model arrays and recursive duplicate properties.
- Preserve source bytes and refuse existing artifacts; record actual source/model/output hashes, with tests, synthetic examples/docs and three-platform installed-tool CI.

Saved-model holdout `evaluate` and optional RMSE gating are not in 0.1.0; they are implemented in unreleased source 0.2.0 above. No neural network, categorical preprocessing or automatic lambda search is provided.
