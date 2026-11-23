# Design and numerical contract

## Read and validate before fitting/writing

`JsonData.Snapshot` reads a bounded file once, checking byte length before/during reading, hashes those exact bytes and decodes strict UTF-8. Optional UTF-8 BOM/CRLF are accepted. `DataReader` parses physical JSONL objects with the default JSON maximum depth 64, recursively rejects duplicate properties, validates IDs/targets/features, and orders feature names ordinally from training only. Validation and prediction must match those exact names. Extra metadata is ignored by fitting and prediction.

Data limits are 32 MiB, 50,000 rows, 1,048,576 UTF-16 code units per line and 1–32 features. ID length is 256, feature-name length 64, with no control characters. Numeric feature/target absolute values are at most `1e12`; all must be finite doubles. The full decoded snapshot plus numeric rows/IDs can use substantially more memory than the byte limit. These are application limits, not a sandbox or guaranteed peak-memory quota.

Training/validation must be nonempty with at least two training rows and disjoint IDs. Identity is exact, case-sensitive and not normalized. This only detects explicit record overlap; callers must choose independent data and account for entities/time/related samples themselves. The source hash describes the snapshot read, not a cross-platform filesystem snapshot guarantee during concurrent writes.

## Fit: `standardized-ridge-v1`

Compute training feature means and population variances with Welford updates, and training target mean. Scale is `sqrt(M2/n)`; zero/nonpositive or underflowed variance uses scale 1. `constantFeatures` in the report lists exactly constant training columns. There is no learned categorical encoding, imputation or validation-dependent feature selection.

Form the dense lower triangle of `A = ZᵀZ + lambda I` and `b = Zᵀ(y - targetMean)`. Solve `A w = b` using Cholesky factorization followed by forward/back substitution. No explicit matrix inverse is formed. The intercept is the unpenalized target mean. The penalty applies to standardized weights and uses a sum-of-squared-errors objective; changing row count changes its relative strength.

Lambda is finite in `[0.001, 1e9]`. Positive regularization handles ordinary constant/collinear columns, but does not guarantee numerical accuracy for every scale/condition. Nonpositive/nonfinite Cholesky pivots, invalid coefficients, nonfinite predictions or overflowed metrics are refused. Normal equations lose precision relative to stronger QR/SVD approaches; this is a small dense baseline, not a replacement for established scientific solvers. Use [existing tools](RESEARCH.md) when their capabilities are needed.

After fitting, transform/score training and separate validation data with that model. The validation baseline always predicts the **training** target mean. No validation statistic changes means, scales or coefficients. Selecting lambda based on reported validation metrics is the caller's model-selection decision, not independent final-test evidence.

## Metrics

MAE is mean absolute error; RMSE is the square root of mean squared error. R² is `1 - sum((target-prediction)^2) / sum((target-meanTargetOfScoredData)^2)`. R² may be negative. If scored target variation is zero, emit JSON `null` rather than force a finite replacement. Metrics use double precision and reject nonfinite results. Neither synthetic metric values nor a successful fit establishes generalization, calibrated uncertainty or causal feature importance.

## Portable model and scoring

Model schema 1 / algorithm `standardized-ridge-v1` stores lambda, ordinal feature names, training means/scales/coefficients, target mean, training row count and source SHA-256. It contains no validation path/hash/statistic, timestamp or machine identity. Model loading rejects missing/unknown/duplicate fields, unsupported schema/algorithm, null/misaligned arrays, nonunique/unordered names, nonfinite values, nonpositive/out-of-range scales, invalid lambda/count/hash metadata. A model file is limited to one MiB. This checks structure/consistency, not authenticity or that a given file was actually trained by this tool.

Batch scoring computes `targetMean + Σ weight_j * ((x_j - mean_j) / scale_j)` in stored feature order. The [standalone scorer](../examples/StandaloneScorer/Program.cs) demonstrates that formula with the BCL alone for an already validated, trusted model; it is deliberately not a second full model validator. No reference to the CLI project or ML package is needed.

Training is deterministic for a fixed runtime/input order/configuration and artifacts have no path/time fields. Floating-point results/model bytes are not promised bit-identical across runtimes or architectures. Tests use appropriate tolerances for arithmetic and compare stable bytes within a runtime. Source/model/output SHA-256 identifies actual bytes, not equivalent floating-point computations.

## Artifact handling and privacy

`train` preflights two distinct fresh model/report paths before fitting and rechecks before exclusive `FileMode.CreateNew` writes. `predict` preflights its output, validates the full input, computes all predictions in memory (64 MiB output limit) and only then creates its file. On a failed write, only files created by that invocation are removed, best effort; directories may remain. Multi-file operations are not atomic. No general symlink/concurrent-filesystem isolation guarantee is provided.

Reports expose feature names, counts, losses and hashes. Model JSON exposes learned feature/target statistics and coefficients. Explicit prediction files expose record IDs and predictions, retaining input order while omitting original features/metadata. Those artifacts can reveal information about supplied data; hashes are not anonymization. Keep private-data artifacts private.

Read [troubleshooting](TROUBLESHOOTING.md), [contributing](../CONTRIBUTING.md) and the [fixed completion scope](../ROADMAP.md). The initial tool has train/predict only. A separate saved-model holdout evaluation/gate is planned before the bounded closing release.
