# Saved-model holdout evaluation

This command is implemented in source 0.2.0; the published 0.1.0 package has no `evaluate`. After producing a model using the [quick start](../README.md):

```sh
dotnet run --project src/Cli -- evaluate --model artifacts/model.json --data examples/holdout.jsonl
dotnet run --project src/Cli -- evaluate --model artifacts/model.json --data examples/holdout.jsonl --max-rmse 2
```

The two synthetic rows follow the same generating formula as the training example, using distinct IDs and feature values. This demonstrates the workflow, not empirical generalization. The evaluator loads the strict schema-1 JSON model and uses its existing named features, means, scales, coefficients and target mean; it never calls the fitter or updates a model. The input must be labeled, obey the normal numeric/feature/ID/byte limits and match names exactly. Reordered feature object fields are accepted. Extra top-level metadata is allowed and ignored. No output files are created by the command.

JSON report schema 1 contains algorithm, actual SHA-256 of model/input bytes, rows, `metrics` (`mae`, `rmse`, `rSquared`), `maximumRmse` and `passed`. No record IDs, raw target values or filenames are printed. Hashes/counts may still disclose information. MAE is mean absolute error; RMSE is square root of mean squared error; R² is `1 - SSE / sum((target - mean(target))²)`. R² is null when all targets are equal and can be negative. Numeric overflow is an error rather than an invalid JSON number.

Without `--max-rmse`, the last two fields are null and exit is 0. An explicit finite nonnegative invariant-formatted threshold sets `passed` true when RMSE is less than or exactly equal to the threshold. A strictly larger RMSE returns exit 1 with the full report. Invalid/missing/repeated options, unsupported/corrupt models, feature mismatch, unlabeled input, nonfinite arithmetic or file errors return exit 2 and no partial JSON. This gate has no comparison tolerance; decimal input is parsed as a binary double.

Model/data inputs are read-only; writing or overwriting shell-redirected output is the caller's responsibility. Snapshot hashes describe bytes read, not an atomic filesystem snapshot under concurrent changes. The model stores training byte SHA-256, not the training ID inventory: this command cannot prove holdout independence, detect all overlapping IDs/entities, verify chronological separation or establish model quality. Keep an independent holdout and provenance externally. See [design](DESIGN.md) and [troubleshooting](TROUBLESHOOTING.md).
