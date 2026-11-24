# Synthetic arithmetic and portable scoring examples

The eight training rows follow `target = 10 + 2*load + 3*workers` with load in `{-3,-1,1,3}` and workers in `{0,1}`. The four validation rows have independent IDs; `predict.jsonl` has two unlabeled rows. `constant-validation.jsonl` demonstrates null R², and `invalid-features.jsonl` demonstrates a missing feature error.

`holdout.jsonl` adds two distinct synthetic labeled rows. After training the example model, source 0.2.0 can run `dotnet run --project src/Cli -- evaluate --model artifacts/model.json --data examples/holdout.jsonl --max-rmse 2`. The gate checks RMSE, with equality passing. This is arithmetic workflow evidence, not a claim of independent real-world sampling or model quality. See [holdout contract](../docs/HOLDOUT.md).

With lambda 1, training means are load 0, workers 0.5, target 11.5. Population scales are `sqrt(5)` and 0.5. The standardized feature columns are orthogonal with squared norms 8, so coefficients are the generating standardized slopes multiplied by `8/9`. The first prediction is therefore `11.5 + (2*sqrt(5)*8/9)*(2/sqrt(5)) + (1.5*8/9)*1`, approximately **16.3888888889**. This is a hand-derived arithmetic fixture, not evidence of production accuracy.

Run the [README commands](../README.md) from a checkout, choosing fresh artifact paths. Models/reports are UTF-8 JSON; predictions are compact UTF-8/no-BOM/LF JSONL with a final LF. They preserve input ID order and include no original feature object.

## Standalone C# consumer

After training `artifacts/model.json`:

```sh
dotnet run --project examples/StandaloneScorer -- artifacts/model.json examples/features.json
```

The [project](StandaloneScorer/StandaloneScorer.csproj) uses the BCL and [this scoring formula](StandaloneScorer/Program.cs), with no NuGet dependency and no reference to the CLI project. Copying that directory with an already-validated trusted model is enough to illustrate portable inference. It is a formula example rather than a second production model validator; use the main CLI loader for its full required-field/schema checks.

`pwsh -File examples/Smoke.ps1` tests the installed tool, first prediction and independent scorer agreement, null R², input/model preservation, fresh-path refusal, bad schema and packaged final-commit metadata. See [contributing](../CONTRIBUTING.md).
