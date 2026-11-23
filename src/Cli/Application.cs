using System.Globalization;
using System.Text;
using System.Text.Json;

namespace RidgeTrainKit;

public static class Application
{
    public static int Run(string[] args, TextWriter output, TextWriter error)
    {
        if (args is ["--version"]) { output.WriteLine("0.1.0"); return 0; }
        if (args is ["--help"] or ["-h"])
        {
            output.WriteLine("ridge-train train --train train.jsonl --validation validation.jsonl --model model.json --report report.json [--lambda 1]\nridge-train predict --model model.json --data input.jsonl --output predictions.jsonl\nCPU numeric ridge regression; training-only population standardization and portable JSON model. Exit 0 success, 2 argument/data/model/numeric/file error. No evaluate command in 0.1.0.");
            return 0;
        }
        try
        {
            if (args.Length == 0 || args[0] is not ("train" or "predict")) throw new InputException("Use train or predict; see --help.");
            var allowed = args[0] == "train" ? new[] { "--train", "--validation", "--model", "--report", "--lambda" } : ["--model", "--data", "--output"];
            var options = new Dictionary<string, string>(StringComparer.Ordinal);
            for (var i = 1; i < args.Length; i++)
            {
                var key = args[i];
                if (!allowed.Contains(key, StringComparer.Ordinal)) throw new InputException("Unknown argument; see --help.");
                if (++i >= args.Length || string.IsNullOrWhiteSpace(args[i]) || args[i].StartsWith("--", StringComparison.Ordinal)) throw new InputException("Option requires a nonblank value.");
                if (!options.TryAdd(key, args[i])) throw new InputException("Repeated option.");
            }
            string Need(string key) => options.GetValueOrDefault(key) ?? throw new InputException("Missing required option; see --help.");
            if (args[0] == "train")
            {
                var modelPath = Need("--model"); var reportPath = Need("--report");
                Artifacts.Preflight(modelPath, reportPath);
                var lambda = 1.0;
                if (options.TryGetValue("--lambda", out var number) && !double.TryParse(number, NumberStyles.Float, CultureInfo.InvariantCulture, out lambda)) throw new InputException("Lambda must use invariant numeric notation.");
                Regression.CheckLambda(lambda);
                var training = DataReader.Read(Need("--train"), requireTarget: true);
                var validation = DataReader.Read(Need("--validation"), requireTarget: true, training.FeatureNames);
                var trainingIds = training.Rows.Select(row => row.Id).ToHashSet(StringComparer.Ordinal);
                if (validation.Rows.Any(row => trainingIds.Contains(row.Id))) throw new InputException("Record ID overlap between training and validation.");
                var model = Regression.Fit(training, lambda);
                var bytes = JsonData.Bytes(model);
                var constant = training.FeatureNames.Where((_, j) => training.Rows.All(row => row.Features[j] == training.Rows[0].Features[j])).ToArray();
                var report = new TrainingReport(1, model.Algorithm, lambda, model.FeatureNames, constant, training.Rows.Length, validation.Rows.Length,
                    training.Sha256, validation.Sha256, JsonData.Hash(bytes), Regression.Score(training, row => Regression.Predict(model, row.Features)),
                    Regression.Score(validation, row => Regression.Predict(model, row.Features)), Regression.Score(validation, _ => model.TargetMean));
                Artifacts.Write((modelPath, bytes), (reportPath, JsonData.Bytes(report)));
                output.WriteLine(JsonSerializer.Serialize(report, JsonData.Options));
            }
            else
            {
                var destination = Need("--output"); Artifacts.Preflight(destination);
                var loaded = Models.Read(Need("--model"));
                var data = DataReader.Read(Need("--data"), requireTarget: false, loaded.Model.FeatureNames);
                using var memory = new MemoryStream();
                foreach (var row in data.Rows)
                {
                    var prediction = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(new { id = row.Id, prediction = Regression.Predict(loaded.Model, row.Features) }) + "\n");
                    if (memory.Length + prediction.Length > 64L * 1024 * 1024) throw new InputException("Prediction output exceeds the 64 MiB limit.");
                    memory.Write(prediction);
                }
                var bytes = memory.ToArray(); Artifacts.Write((destination, bytes));
                output.WriteLine(JsonSerializer.Serialize(new { schemaVersion = 1, rows = data.Rows.Length, modelSha256 = loaded.Sha256, inputSha256 = data.Sha256, outputSha256 = JsonData.Hash(bytes) }, JsonData.Options));
            }
            return 0;
        }
        catch (InputException exception) { error.WriteLine(exception.Message); return 2; }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException or ArgumentException)
        { error.WriteLine("Cannot read input/model or create fresh artifacts. Check paths, access and numeric/UTF-8 input."); return 2; }
    }
}
