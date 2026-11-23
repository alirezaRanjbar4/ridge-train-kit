# Troubleshooting

## Existing path or exit 2

Read stderr. Artifact paths must be distinct and fresh; a file or directory already at a destination is refused. Choose new model/report/prediction paths. Do not redirect stdout onto an input or intended dataset output: shell redirection happens before the tool reads input. Failed writes can leave directories or files if cleanup also fails; inspect the destination after an I/O error.

Invalid rows/models, schema mismatch, numeric failure and bad CLI options also return 2 with no partial JSON report. The current tool has no quality-gate exit 1, `evaluate` command or implicit repair/export behavior.

## Feature schema or ID errors

Names are exact and case-sensitive. Every row must contain exactly the same 1–32 numeric features as training/the saved model; key order may vary. Feature values and targets must be finite JSON numbers within `±1e12`, not quoted strings, bools, nulls or missing values. Duplicate properties anywhere in a row are refused, even inside optional metadata.

IDs must be unique within each dataset and disjoint across training/validation. Different IDs do not establish independent entities/time periods. Keep a truly unseen test set when your experiment needs one. The tool does not detect duplicate feature vectors or shared entities automatically; replicates can be valid observations.

## Lambda, constant columns and numerical errors

Use invariant notation such as `--lambda 0.5`; decimal commas and nonfinite values are rejected. The supported interval is 0.001 through 1e9. Zero is not an ordinary-least-squares mode. Coefficients use training-standardized features and a summed squared-error objective, so lambda values are not interchangeable with tools using mean loss or different scaling.

Constant training features use unit scale and yield zero centered contribution. The model cannot learn an effect from a feature that does not vary. Very small variance, extreme extrapolation or ill-conditioned data can still produce unreliable or nonfinite results. Normal equations have precision limitations; use a suitable established solver for stronger numerical requirements. No automatic solver fallback or accuracy guarantee is present. See [design](DESIGN.md).

## R² is negative or null

Negative R² means the predictions are worse in squared error than the scored data's mean comparator used by that formula. It is a valid result. R² is `null` if the scored targets have zero variation, including one-row validation. The separate `meanBaselineValidation` metrics use the training mean and help compare against a deployable constant baseline.

The toy examples test arithmetic and round-trip behavior. Their scores do not measure production quality or prove the model generalizes. Inspect representative held-out data and the target's use case.

## Model loading or standalone scoring fails

Use an actual validated schema-1 `standardized-ridge-v1` model. The loader checks exact required fields, no unknown/duplicate fields, aligned finite arrays, positive scales and metadata. Editing a model's names/array order or dropping training statistics can change predictions or invalidate the model. JSON is inspectable, not self-authenticating.

The [standalone example](../examples/README.md) demonstrates the formula for a trusted validated model and does not replace the strict loader. Its project has no ML package or CLI-project reference. A trained model alone is sufficient for the main CLI's `predict`; the training report is not required at inference.

## Install, memory or bug reports

Use .NET 10 and download the GitHub Release nupkg/checksum into `artifacts/`; the local-only config does not fetch missing files from NuGet.org. Follow the [README](../README.md). Data/model/output limits are fixed; decoded data and working memory exceed file byte size. There is no streaming training, GPU or remote endpoint.

For a bug provide runtime/tool version, lambda, command, exit code and a minimal synthetic case with expected arithmetic/schema behavior. Review feature names/model statistics/IDs before sharing artifacts, and never upload private datasets. See [contributing](../CONTRIBUTING.md).
