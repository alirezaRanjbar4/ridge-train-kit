# Fixed first scope

Scope fixed on 2026-11-23 before initial publication.

- [ ] Usable v0.1.0: bounded named-numeric JSONL, actual CPU ridge fitting with training-only scaling, separate validation and mean baseline metrics, inspectable portable JSON model, strict saved-model batch prediction, no-overwrite artifacts, source/output hashes, standalone C# scoring example, tests/docs/three-platform CI and verified release.
- [ ] Read-only `evaluate` for a saved model and fresh labeled holdout, with actual model/data hashes, regression metrics and an optional maximum-RMSE gate (exit 1 on gate failure). It must use stored transforms/coefficients, never refit, preserve files and include hand-checked metric/gate/error regressions.
- [ ] Validated closing release containing that holdout workflow, complete bundled guides/examples, clean installation, supported-platform CI and final-commit/tag/asset verification.

After completion, keep the repository public and maintain reproducible bugs/feedback. Neural networks, LLM fine-tuning, automatic hyperparameter search, categorical encoding, imputation, confidence intervals, causal explanations, CSV conversion and alternative numerical solvers are outside this bounded scope. Do not add new requirements indefinitely.

The 0.1.0 train/predict workflow is independently useful. There is no saved-model `evaluate` command or RMSE gate in that version. See [research](docs/RESEARCH.md) and [design](docs/DESIGN.md).
