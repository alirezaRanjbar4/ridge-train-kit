# Fixed first scope

Scope fixed on 2026-11-23 before initial publication.

- [x] Usable v0.1.0: bounded named-numeric JSONL, actual CPU ridge fitting with training-only scaling, separate validation and mean baseline metrics, inspectable portable JSON model, strict saved-model batch prediction, no-overwrite artifacts, source/output hashes, standalone C# scoring example, tests/docs/three-platform CI and verified release.
- [x] Read-only `evaluate` implemented in source 0.2.0 for a saved model and labeled holdout, with actual model/data hashes, regression metrics and optional maximum-RMSE gate; stored transforms/coefficients only, hand-checked gate/error/preservation regressions. See [holdout design](docs/HOLDOUT.md).
- [ ] Validated closing release containing that holdout workflow, complete bundled guides/examples, clean installation, supported-platform CI and final-commit/tag/asset verification.

After completion, keep the repository public and maintain reproducible bugs/feedback. Neural networks, LLM fine-tuning, automatic hyperparameter search, categorical encoding, imputation, confidence intervals, causal explanations, CSV conversion and alternative numerical solvers are outside this bounded scope. Do not add new requirements indefinitely.

The published 0.1.0 train/predict workflow remains independently useful and has no saved-model `evaluate` command or RMSE gate. Those features are present in source 0.2.0, pending a verified closing release. See [research](docs/RESEARCH.md) and [design](docs/DESIGN.md).
