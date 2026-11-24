using System.Text.Json;

namespace RidgeTrainKit;

public sealed record Model(int SchemaVersion, string Algorithm, double Lambda, string[] FeatureNames,
    double[] Means, double[] Scales, double[] Coefficients, double TargetMean, int TrainingRows, string TrainingSha256);
public sealed record Metrics(double Mae, double Rmse, double? RSquared);
public sealed record EvaluationReport(int SchemaVersion, string Algorithm, string ModelSha256, string InputSha256, int Rows, Metrics Metrics, double? MaximumRmse, bool? Passed);
public sealed record TrainingReport(int SchemaVersion, string Algorithm, double Lambda, string[] FeatureNames,
    string[] ConstantFeatures, int TrainingRows, int ValidationRows, string TrainingSha256, string ValidationSha256,
    string ModelSha256, Metrics Training, Metrics Validation, Metrics MeanBaselineValidation);

public static class Models
{
    public static (Model Model, string Sha256) Read(string path)
    {
        var snapshot = JsonData.Snapshot(path, 1024 * 1024);
        try
        {
            using var json = JsonDocument.Parse(snapshot.Text);
            JsonData.RejectDuplicates(json.RootElement);
            var model = json.RootElement.Deserialize<Model>(JsonData.Options) ?? throw new JsonException();
            Validate(model);
            return (model, JsonData.Hash(snapshot.Bytes));
        }
        catch (JsonException) { throw new InputException("Malformed model JSON; required fields, exact schema and no duplicate/unknown properties are expected."); }
        catch (InvalidOperationException) { throw new InputException("Malformed model JSON; strings must contain valid Unicode."); }
    }
    public static void Validate(Model model)
    {
        if (model.SchemaVersion != 1 || model.Algorithm != "standardized-ridge-v1" ||
            model.FeatureNames is null || model.Means is null || model.Scales is null || model.Coefficients is null ||
            model.FeatureNames.Length is < 1 or > 32 || model.Means.Length != model.FeatureNames.Length ||
            model.Scales.Length != model.FeatureNames.Length || model.Coefficients.Length != model.FeatureNames.Length ||
            model.FeatureNames.Any(name => string.IsNullOrWhiteSpace(name) || name.Length > 64 || name.Any(char.IsControl)) ||
            !model.FeatureNames.SequenceEqual(model.FeatureNames.Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal), StringComparer.Ordinal) ||
            model.Means.Any(value => !double.IsFinite(value) || Math.Abs(value) > 1e12) ||
            model.Scales.Any(value => !double.IsFinite(value) || value <= 0 || value > 2e12) ||
            model.Coefficients.Any(value => !double.IsFinite(value)) || !double.IsFinite(model.TargetMean) || Math.Abs(model.TargetMean) > 1e12 ||
            model.TrainingRows is < 2 or > DataReader.MaxRows || model.TrainingSha256 is null ||
            model.TrainingSha256.Length != 64 || !model.TrainingSha256.All(Uri.IsHexDigit))
            throw new InputException("Unsupported or inconsistent numeric model schema.");
        Regression.CheckLambda(model.Lambda);
    }
}

public static class Regression
{
    public static void CheckLambda(double value)
    {
        if (!double.IsFinite(value) || value is < 0.001 or > 1e9) throw new InputException("Lambda must be finite and between 0.001 and 1000000000.");
    }
    public static Model Fit(Dataset training, double lambda = 1)
    {
        CheckLambda(lambda);
        if (training.Rows.Length < 2 || training.Rows.Any(row => row.Target is null)) throw new InputException("Training needs at least two labeled rows.");
        var size = training.FeatureNames.Length;
        var means = new double[size]; var squares = new double[size]; var targetMean = 0.0; var count = 0;
        foreach (var row in training.Rows)
        {
            count++;
            for (var j = 0; j < size; j++)
            {
                var delta = row.Features[j] - means[j];
                means[j] += delta / count;
                squares[j] += delta * (row.Features[j] - means[j]);
            }
            targetMean += (row.Target!.Value - targetMean) / count;
        }
        var scales = squares.Select(value => value > 0 ? Math.Sqrt(value / count) : 1).ToArray();
        // If variance underflows to zero, retain a finite unit scale.
        for (var j = 0; j < size; j++) if (scales[j] == 0) scales[j] = 1;
        var matrix = new double[size, size]; var right = new double[size];
        foreach (var row in training.Rows)
        {
            var z = row.Features.Select((value, j) => (value - means[j]) / scales[j]).ToArray();
            for (var j = 0; j < size; j++)
            {
                right[j] += z[j] * (row.Target!.Value - targetMean);
                for (var k = 0; k <= j; k++) matrix[j, k] += z[j] * z[k];
            }
        }
        for (var j = 0; j < size; j++) matrix[j, j] += lambda;
        var lower = new double[size, size];
        for (var j = 0; j < size; j++)
            for (var k = 0; k <= j; k++)
            {
                var value = matrix[j, k];
                for (var h = 0; h < k; h++) value -= lower[j, h] * lower[k, h];
                if (!double.IsFinite(value) || (j == k && value <= 0)) throw new InputException("Numeric solve failed; review feature scale and regularization.");
                lower[j, k] = j == k ? Math.Sqrt(value) : value / lower[k, k];
            }
        var intermediate = new double[size]; var coefficients = new double[size];
        for (var j = 0; j < size; j++)
        {
            var value = right[j]; for (var k = 0; k < j; k++) value -= lower[j, k] * intermediate[k];
            intermediate[j] = value / lower[j, j];
        }
        for (var j = size - 1; j >= 0; j--)
        {
            var value = intermediate[j]; for (var k = j + 1; k < size; k++) value -= lower[k, j] * coefficients[k];
            coefficients[j] = value / lower[j, j];
        }
        var model = new Model(1, "standardized-ridge-v1", lambda, training.FeatureNames, means, scales, coefficients, targetMean, count, training.Sha256);
        Models.Validate(model);
        return model;
    }
    public static double Predict(Model model, double[] features)
    {
        if (features.Length != model.FeatureNames.Length) throw new InputException("Feature count does not match the model.");
        var value = model.TargetMean;
        for (var j = 0; j < features.Length; j++) value += model.Coefficients[j] * ((features[j] - model.Means[j]) / model.Scales[j]);
        if (!double.IsFinite(value)) throw new InputException("Prediction is not finite; review input scale and model.");
        return value;
    }
    public static Metrics Score(Dataset data, Func<DataRow, double> predict)
    {
        if (data.Rows.Length == 0 || data.Rows.Any(row => row.Target is null)) throw new InputException("Metrics require nonempty labeled data.");
        var mean = 0.0; var variance = 0.0; var absolute = 0.0; var squared = 0.0; var count = 0;
        foreach (var row in data.Rows)
        {
            var target = row.Target!.Value; var prediction = predict(row); var error = target - prediction;
            absolute += Math.Abs(error); squared += error * error;
            count++; var delta = target - mean; mean += delta / count; variance += delta * (target - mean);
        }
        var mae = absolute / count; var rmse = Math.Sqrt(squared / count);
        double? r2 = variance > 0 ? 1 - squared / variance : null;
        if (!double.IsFinite(mae) || !double.IsFinite(rmse) || (r2.HasValue && !double.IsFinite(r2.Value))) throw new InputException("Metrics are not finite; review numeric scale.");
        return new(mae, rmse, r2);
    }
}
