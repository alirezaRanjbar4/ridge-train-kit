# Problem and alternatives

Reviewed on 2026-11-23. A small numeric-regression baseline can be useful to .NET services when the fitted scaling/coefficients should remain inspectable JSON and inference should use only the BCL. Our preceding text-classifier package distributes ML.NET assets and seven runtime package versions; it solves a different text-classification workflow. The narrow need chosen here is a dependency-free numeric baseline with explicit named-feature validation and portable fitted transforms, not a general ML framework replacement.

Primary sources reviewed:

- [ML.NET CLI](https://learn.microsoft.com/en-us/dotnet/machine-learning/reference/ml-net-cli-reference) already supports regression/AutoML, ZIP models and generated C# scoring/training code.
- [ML.NET cookbook](https://github.com/dotnet/machinelearning/blob/main/docs/code/MlNetCookBook.md) already explains coefficient inspection, training, model persistence and prediction. This tool does not claim .NET lacks those capabilities.
- [Math.NET regression](https://numerics.mathdotnet.com/Regression) provides established fitting APIs and explains normal-equation precision tradeoffs versus QR/SVD.
- [scikit-learn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) defines squared-error plus L2 regularization; [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) documents training statistics and population-variance scaling.

The useful niche is an inference from those scopes and the observed deployment/artifact requirements: one small C# CLI for dense numeric JSONL, bounded ridge training, training-only scaling, explicit validation/mean baseline, a strict transparent JSON contract and a BCL-only consumer. No uniqueness, unmet native-API fix, market-adoption or improved-accuracy claim is made. Solver and I/O code are independently written; no dependency source is copied.

This is traditional linear model training on CPU, not LLM training. Use established numerical/ML libraries for broader algorithms, precision requirements and production workflows beyond these [documented limits](DESIGN.md).
