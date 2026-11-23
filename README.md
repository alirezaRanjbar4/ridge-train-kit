# Ridge Train Kit

[![CI](https://github.com/alirezaRanjbar4/ridge-train-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/alirezaRanjbar4/ridge-train-kit/actions/workflows/ci.yml)

**Train a small numeric regression baseline in C# and keep the fitted model as inspectable JSON.** CPU ridge fitting, training-only feature scaling, independent validation metrics and batch prediction, using only the .NET 10 BCL at runtime.

This is useful when a simple dense linear baseline and a portable named-feature contract are enough. The JSON stores feature order, fitted means/scales and coefficients; a [standalone C# scorer](examples/StandaloneScorer/Program.cs) reads it without a reference to this tool or an ML package. Existing [ML/numerical tools](docs/RESEARCH.md) cover broader algorithms and precision needs.

It trains a traditional numeric model, not a neural network or LLM. It does not encode categories, impute missing values, tune hyperparameters or establish production accuracy. Normal-equation Cholesky fitting has precision limits; use a suitable established solver/framework when those limits matter.

## Quick start

With the .NET 10 SDK, from a checkout:

```sh
dotnet test -c Release
dotnet run --project src/Cli -- train --train examples/train.jsonl --validation examples/validation.jsonl --model artifacts/model.json --report artifacts/training.json --lambda 1
dotnet run --project src/Cli -- predict --model artifacts/model.json --data examples/predict.jsonl --output artifacts/predictions.jsonl
dotnet run --project examples/StandaloneScorer -- artifacts/model.json examples/features.json
```

Choose fresh artifact paths for another run. The synthetic training set has eight rows, two features and `target = 10 + 2*load + 3*workers`. The four validation rows are separate. With lambda 1, the first prediction (`load=2`, `workers=1`) is approximately **16.3888888889**; regularization shrinks the slopes, so it need not equal the generating value 17. The independent scorer should agree with the CLI.

These are arithmetic and compatibility examples, not an accuracy benchmark. Inspect both validation metrics and the training-mean baseline before choosing a model. See [examples](examples/README.md) and [troubleshooting](docs/TROUBLESHOOTING.md).

## Input

UTF-8 JSONL, one complete object per physical line:

```json
{"id":"train-01","features":{"load":-3,"workers":0},"target":4}
```

- `id`: a unique nonblank string, at most 256 UTF-16 code units, with no controls. Training and validation IDs must be disjoint.
- `features`: 1–32 named finite numbers with absolute value at most `1e12`. Names are nonblank, at most 64 UTF-16 code units, no controls. Every row must have exactly the training/model feature names; property order may differ.
- `target`: a finite number within `±1e12`, required for training/validation. Prediction accepts it optionally but does not use it for fitting or scoring.

Names and IDs use exact ordinal text, including case and Unicode form. Extra top-level metadata is allowed; duplicate properties anywhere in a row are rejected. Missing/extra numeric features, strings in place of numbers, nulls, blank lines and malformed/non-UTF-8 input are refused. An optional UTF-8 BOM and CRLF input are accepted. Training needs at least two rows; validation at least one. IDs detect explicit record overlap; different IDs do not prove entity, temporal or semantic independence.

## What is fitted

From training rows only, learn feature means and population standard deviations (`ddof=0`), and the target mean. Constant/underflowed-variance features use unit scale. Fit standardized coefficients to:

```text
sum_i (y_i - targetMean - z_i · coefficients)^2 + lambda * sum_j coefficients_j^2
z_j = (x_j - means_j) / scales_j
prediction = targetMean + sum_j coefficients_j * z_j
```

The target-mean intercept is unpenalized. Lambda is the penalty on **standardized** coefficients for a summed squared-error objective, not a mean-loss objective. It defaults to 1 and must be within `[0.001, 1e9]`; zero/negative/nonfinite values are refused. Validation is transformed with the stored training statistics and never changes the fitted model. There is no random seed or automatic lambda search.

See [design and numerical limits](docs/DESIGN.md) for Welford statistics, Cholesky solving and the fixed model contract.

## Model, reports and predictions

`train` creates a versioned JSON model and a separate report. The report records training/validation byte SHA-256, actual model-file SHA-256, feature names, row counts, constant features and MAE/RMSE/R² for training, validation and a constant predictor equal to the training target mean. R² can be negative; it is `null` when target variation is zero, including a single-row validation set.

`predict` loads the model JSON alone, validates its required/exact schema and feature arrays, then creates compact JSONL in source order:

```json
{"id":"query-01","prediction":16.38888888888889}
```

Its stdout summary contains row count and actual model/input/output byte hashes. The output uses UTF-8 without BOM, LF and a final LF. Prediction IDs are deliberately included for joining results; extra source fields are omitted. Reports omit raw row IDs/features/targets, while models expose learned statistics/coefficients. Models and reports are data-derived artifacts; hashes do not anonymize them.

All artifact files must be distinct fresh paths. Existing files/directories and input aliases are refused. Files are opened exclusively with best-effort cleanup on failure; writes are not a multi-file atomic transaction. Inputs are read as bounded snapshots and preserved. `0` means success; `2` means invalid options/data/model/numeric operation or file access, with stderr and no partial JSON report. There is no quality-gate exit 1 or saved-model `evaluate` command in 0.1.0.

## Install a release

Packages are [GitHub Release assets](https://github.com/alirezaRanjbar4/ridge-train-kit/releases), **not published on NuGet.org**. From a checkout, download v0.1.0 nupkg/checksum into `artifacts/`, compare SHA-256 and install:

```sh
gh release download v0.1.0 --repo alirezaRanjbar4/ridge-train-kit --dir artifacts
dotnet tool install RidgeTrainKit.Tool --version 0.1.0 --tool-path artifacts/tool --configfile nuget.local.config --no-cache
./artifacts/tool/ridge-train --version
```

Browser downloads work instead of the optional GitHub CLI. Use `Get-FileHash -Algorithm SHA256` or `sha256sum`. The config uses only the local `artifacts/` feed. Installation requires an SDK; the installed tool runs on .NET 10. No runtime package dependency, Python, native ML library, GPU, API key or paid service is needed.

## Limits and project information

Datasets: 32 MiB, 50,000 rows, one Mi UTF-16 code units per line, 32 features. Models: one MiB. Prediction output: 64 MiB. Data/matrices and prediction output are kept in memory. Finite input does not guarantee a finite extrapolated prediction/metric; numeric failure returns 2 before artifacts are written. This small dense baseline is not a precision/conditioning guarantee or a hostile-input sandbox.

- [Design](docs/DESIGN.md), [troubleshooting](docs/TROUBLESHOOTING.md), [research](docs/RESEARCH.md)
- [Contributing](CONTRIBUTING.md), [bounded roadmap](ROADMAP.md), [changelog](CHANGELOG.md), [MIT license](LICENSE)

CI tests hand-derived solutions, validation isolation, schema/metric/provenance/error behavior, formats, packs, installs and exercises the independent scorer on Windows/Linux/macOS. Saved-model holdout evaluation/gates are fixed future work, not advertised as available.
