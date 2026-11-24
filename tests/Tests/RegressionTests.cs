using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using RidgeTrainKit;

namespace Tests;

public sealed class RegressionTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "ridge-kit-" + Guid.NewGuid().ToString("N"));
    public RegressionTests() => Directory.CreateDirectory(root);
    private string PathFor(string name) => Path.Combine(root, name);
    private string Data(IEnumerable<string> rows)
    {
        var path = PathFor(Guid.NewGuid().ToString("N") + ".jsonl");
        File.WriteAllText(path, string.Join('\n', rows) + "\n", new UTF8Encoding(false));
        return path;
    }
    private static string Row(string id, double x, double? y) => JsonSerializer.Serialize(new { id, features = new { x }, target = y });
    private string Training() => Data([Row("t1", -1, -1), Row("t2", 0, 1), Row("t3", 1, 3)]);
    private string Validation() => Data([Row("v1", 2, 5), Row("v2", 3, 7)]);
    private string ModelFile(Model? model = null)
    {
        var path = PathFor(Guid.NewGuid().ToString("N") + ".json");
        File.WriteAllBytes(path, JsonData.Bytes(model ?? Regression.Fit(DataReader.Read(Training(), true))));
        return path;
    }
    private (int Code, string Output, string Error) Train(string train, string validation, string model, string report, params string[] extra) =>
        Run(new[] { "train", "--train", train, "--validation", validation, "--model", model, "--report", report }.Concat(extra).ToArray());

    [Fact]
    public void OneFeatureFitMatchesHandDerivedRidgeSolution()
    {
        // x=(-1,0,1), y=(-1,1,3), population scale sqrt(2/3).
        // Lambda=1 gives original slope 1.5, unpenalized intercept 1.
        var model = Regression.Fit(DataReader.Read(Training(), true), 1);
        Assert.Equal(0, model.Means[0], 12); Assert.Equal(Math.Sqrt(2.0 / 3), model.Scales[0], 12);
        Assert.Equal(1, model.TargetMean, 12);
        Assert.Equal(1.5, model.Coefficients[0] / model.Scales[0], 12);
        Assert.Equal(4, Regression.Predict(model, [2]), 12);
    }

    [Fact]
    public void OrthogonalTwoFeatureFitMatchesIndependentClosedForm()
    {
        var path = Data(new[] { (-1, -1), (-1, 1), (1, -1), (1, 1) }.Select((p, i) =>
            JsonSerializer.Serialize(new { id = "t" + i, target = 10 + 2 * p.Item1 - 3 * p.Item2, features = new { b = p.Item2, a = p.Item1 } })));
        var model = Regression.Fit(DataReader.Read(path, true));
        Assert.Equal(["a", "b"], model.FeatureNames); Assert.Equal([0.0, 0.0], model.Means);
        Assert.Equal(1.6, model.Coefficients[0], 12); Assert.Equal(-2.4, model.Coefficients[1], 12);
        Assert.Equal(10.8, Regression.Predict(model, [2, 1]), 12);
    }

    [Fact]
    public void ValidationValuesNeverChangeFittedModelOrScaler()
    {
        var train = Training(); var a = PathFor("a.json"); var b = PathFor("b.json");
        Assert.Equal(0, Train(train, Validation(), a, PathFor("a-report.json")).Code);
        Assert.Equal(0, Train(train, Data([Row("v1", 1e9, -1e9), Row("v2", -1e9, 1e9)]), b, PathFor("b-report.json")).Code);
        Assert.Equal(File.ReadAllBytes(a), File.ReadAllBytes(b));
    }

    [Fact]
    public void MetricsMatchHandCalculatedErrorsAndAllowNegativeRSquared()
    {
        var model = Regression.Fit(DataReader.Read(Training(), true));
        var metrics = Regression.Score(DataReader.Read(Validation(), true), row => Regression.Predict(model, row.Features));
        Assert.Equal(1.25, metrics.Mae, 12); Assert.Equal(Math.Sqrt(1.625), metrics.Rmse, 12);
        Assert.Equal(-0.625, metrics.RSquared!.Value, 12);
        var constant = DataReader.Read(Data([Row("v1", 2, 5), Row("v2", 3, 5)]), true);
        Assert.Null(Regression.Score(constant, row => Regression.Predict(model, row.Features)).RSquared);
    }

    [Fact]
    public void ConstantAndCollinearFeaturesRemainFiniteUnderPositiveRidge()
    {
        var data = DataReader.Read(Data(Enumerable.Range(0, 6).Select(i => JsonSerializer.Serialize(new
        { id = "t" + i, target = 2.0 * i + 1, features = new { constant = 7.0, a = i, b = 2 * i } }))), true);
        var model = Regression.Fit(data, 0.001);
        Assert.Equal(1, model.Scales[2], 12); Assert.Equal(0, model.Coefficients[2], 12);
        Assert.All(model.Coefficients, value => Assert.True(double.IsFinite(value)));
        Assert.True(double.IsFinite(Regression.Predict(model, [8, 16, 7])));
    }

    [Fact]
    public void FitSatisfiesIndependentPenalizedStationarityCheck()
    {
        var data = DataReader.Read(Data(Enumerable.Range(0, 9).Select(i => JsonSerializer.Serialize(new
        { id = "t" + i, target = 5 + 0.3 * i + Math.Sin(i), features = new { a = i, b = i % 3 } }))), true);
        var model = Regression.Fit(data, 2.5);
        for (var j = 0; j < 2; j++)
        {
            var residualProduct = 0.0;
            foreach (var row in data.Rows)
            {
                var prediction = model.TargetMean + Enumerable.Range(0, 2).Sum(k => model.Coefficients[k] * (row.Features[k] - model.Means[k]) / model.Scales[k]);
                residualProduct += (row.Features[j] - model.Means[j]) / model.Scales[j] * (row.Target!.Value - prediction);
            }
            Assert.Equal(model.Lambda * model.Coefficients[j], residualProduct, 9);
        }
    }

    [Fact]
    public void SavedJsonAloneSupportsOrderedBatchPredictionAndPreservesBytes()
    {
        var modelPath = ModelFile(); var modelBefore = File.ReadAllBytes(modelPath);
        var input = Data(["{\"id\":\"ق۱\",\"features\":{\"x\":2},\"metadata\":{\"keep\":true}}", "{\"id\":\"q2\",\"features\":{\"x\":3}}"]);
        var before = File.ReadAllBytes(input); var output = PathFor("predictions.jsonl");
        var result = Run(["predict", "--model", modelPath, "--data", input, "--output", output]);
        Assert.Equal(0, result.Code);
        var lines = File.ReadAllLines(output); Assert.Equal(2, lines.Length);
        using var first = JsonDocument.Parse(lines[0]);
        Assert.Equal("ق۱", first.RootElement.GetProperty("id").GetString());
        Assert.Equal(4, first.RootElement.GetProperty("prediction").GetDouble(), 12);
        using var report = JsonDocument.Parse(result.Output);
        Assert.Equal(JsonData.Hash(File.ReadAllBytes(output)), report.RootElement.GetProperty("outputSha256").GetString());
        Assert.Equal(modelBefore, File.ReadAllBytes(modelPath)); Assert.Equal(before, File.ReadAllBytes(input));
        Assert.DoesNotContain((byte)13, File.ReadAllBytes(output)); Assert.Equal(10, File.ReadAllBytes(output)[^1]);
    }

    [Fact]
    public void TrainingReportHasActualArtifactAndInputHashes()
    {
        var train = Training(); var validation = Validation(); var model = PathFor("model.json"); var report = PathFor("report.json");
        var result = Train(train, validation, model, report); Assert.Equal(0, result.Code);
        using var json = JsonDocument.Parse(File.ReadAllText(report));
        Assert.Equal(JsonData.Hash(File.ReadAllBytes(train)), json.RootElement.GetProperty("trainingSha256").GetString());
        Assert.Equal(JsonData.Hash(File.ReadAllBytes(validation)), json.RootElement.GetProperty("validationSha256").GetString());
        Assert.Equal(JsonData.Hash(File.ReadAllBytes(model)), json.RootElement.GetProperty("modelSha256").GetString());
        Assert.Equal(File.ReadAllText(report).TrimEnd('\n'), result.Output.TrimEnd('\r', '\n'));
        Assert.DoesNotContain("t1", result.Output); Assert.DoesNotContain(Path.GetFileName(train), result.Output);
    }

    [Fact]
    public void FeatureObjectOrderIsIgnoredButNamesMustMatchExactly()
    {
        var input = Data(["{\"id\":\"1\",\"target\":1,\"features\":{\"b\":2,\"a\":1}}", "{\"id\":\"2\",\"target\":2,\"features\":{\"a\":3,\"b\":4}}"]);
        Assert.Equal(["a", "b"], DataReader.Read(input, true).FeatureNames);
        Assert.Throws<InputException>(() => DataReader.Read(Data(["{\"id\":\"v1\",\"target\":1,\"features\":{\"A\":2,\"b\":3}}"]), true, ["a", "b"]));
    }

    [Fact]
    public void ExistingArtifactsAndInputAliasesAreRefusedWithoutChangingFiles()
    {
        var source = Training(); var before = File.ReadAllBytes(source);
        Assert.Equal(2, Train(source, Validation(), source, PathFor("report.json")).Code);
        Assert.Equal(before, File.ReadAllBytes(source)); Assert.False(File.Exists(PathFor("report.json")));
        Assert.Equal(2, Train(source, Validation(), PathFor("same"), PathFor("same")).Code);
        var existing = PathFor("exists"); File.WriteAllText(existing, "keep");
        Assert.Throws<InputException>(() => Artifacts.Write((existing, [1, 2]))); Assert.Equal("keep", File.ReadAllText(existing));
    }

    [Fact]
    public void OverlappingIdsAndSchemaErrorsProduceNoTrainingArtifacts()
    {
        var train = Training(); var model = PathFor("model.json"); var report = PathFor("report.json");
        Assert.Equal(2, Train(train, Data([Row("t1", 9, 20)]), model, report).Code);
        Assert.False(File.Exists(model)); Assert.False(File.Exists(report));
        Assert.Equal(2, Train(train, Data(["{\"id\":\"v\",\"target\":1,\"features\":{\"other\":2}}"]), model, report).Code);
        Assert.False(File.Exists(model)); Assert.False(File.Exists(report));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(0.0001)]
    [InlineData(1000000001)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void UnsupportedRegularizationIsRejected(double value) => Assert.Throws<InputException>(() => Regression.Fit(DataReader.Read(Training(), true), value));

    [Theory]
    [InlineData("")]
    [InlineData("{")]
    [InlineData("[]")]
    [InlineData("{\"id\":\"a\",\"target\":1,\"features\":{}}")]
    [InlineData("{\"id\":\"a\",\"target\":1,\"features\":{\"x\":\"2\"}}")]
    [InlineData("{\"id\":\"a\",\"target\":null,\"features\":{\"x\":2}}")]
    [InlineData("{\"id\":\"a\",\"target\":1,\"features\":{\"x\":1e400}}")]
    [InlineData("{\"id\":\"a\",\"target\":1e13,\"features\":{\"x\":1}}")]
    [InlineData("{\"id\":\"a\",\"target\":1,\"features\":{\"x\":1,\"x\":2}}")]
    [InlineData("{\"id\":\"a\",\"target\":1,\"features\":{\"x\":1},\"meta\":[{\"x\":1,\"x\":2}]}")]
    public void InvalidNumericRowsAreRejectedWithValueFreeLineDiagnostics(string raw)
    {
        var exception = Assert.Throws<InputException>(() => DataReader.Read(Data([raw]), true));
        Assert.Contains("line 1", exception.Message);
    }

    [Fact]
    public void EmptyDuplicateAndInvalidUtf8InputsAreRejected()
    {
        var empty = PathFor("empty"); File.WriteAllText(empty, ""); Assert.Throws<InputException>(() => DataReader.Read(empty, true));
        Assert.Throws<InputException>(() => DataReader.Read(Data([Row("a", 1, 1), Row("a", 2, 2)]), true));
        var invalid = PathFor("utf8"); File.WriteAllBytes(invalid, [0xff]); Assert.Throws<InputException>(() => DataReader.Read(invalid, true));
        var bom = PathFor("bom"); File.WriteAllText(bom, Row("a", 1, 1) + "\r\n" + Row("b", 2, 2) + "\r\n", new UTF8Encoding(true));
        Assert.Equal(JsonData.Hash(File.ReadAllBytes(bom)), DataReader.Read(bom, true).Sha256);
    }

    [Fact]
    public void MalformedOrInconsistentSavedModelsAreRejected()
    {
        var original = JsonNode.Parse(File.ReadAllText(ModelFile()))!.AsObject();
        foreach (var mutation in new Action<JsonObject>[]
        {
            json => json.Remove("means"), json => json["means"] = null,
            json => json["scales"] = new JsonArray(0.0), json => json["coefficients"] = new JsonArray(),
            json => json["schemaVersion"] = 2, json => json["extra"] = true,
            json => json["algorithm"] = "other", json => json["trainingSha256"] = "bad"
        })
        {
            var changed = original.DeepClone().AsObject(); mutation(changed); var path = PathFor(Guid.NewGuid() + ".json");
            File.WriteAllText(path, changed.ToJsonString()); Assert.Throws<InputException>(() => Models.Read(path));
        }
        var duplicate = PathFor("duplicate-model.json"); File.WriteAllText(duplicate, original.ToJsonString().Replace("\"schemaVersion\":1", "\"schemaVersion\":1,\"schemaVersion\":1"));
        Assert.Throws<InputException>(() => Models.Read(duplicate));
    }

    [Fact]
    public void NumericOverflowIsAnErrorAndNeverProducesPredictionFiles()
    {
        var model = Regression.Fit(DataReader.Read(Training(), true)) with { Coefficients = [1e308], Scales = [1e-300] };
        var path = ModelFile(model); var output = PathFor("predictions.jsonl");
        Assert.Equal(2, Run(["predict", "--model", path, "--data", Data([Row("q1", 1, 1)]), "--output", output]).Code);
        Assert.False(File.Exists(output));
    }

    [Fact]
    public void InputByteAndFeatureCountBoundsAreEnforced()
    {
        var large = PathFor("large"); using (var file = File.Create(large)) file.SetLength(DataReader.MaxBytes + 1L);
        Assert.Throws<InputException>(() => DataReader.Read(large, true));
        var features = Enumerable.Range(0, 33).ToDictionary(i => "x" + i, i => (double)i);
        Assert.Throws<InputException>(() => DataReader.Read(Data([JsonSerializer.Serialize(new { id = "1", target = 1, features })]), true));
    }

    [Fact]
    public void InvariantTrainingArtifactsMatchAcrossPathsAndCultures()
    {
        var train = Training(); var validation = Validation(); var previous = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("fa-IR");
            Assert.Equal(0, Train(train, validation, PathFor("a-model"), PathFor("a-report"), "--lambda", "1.5").Code);
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("fr-FR");
            Assert.Equal(0, Train(train, validation, PathFor("b-model"), PathFor("b-report"), "--lambda", "1.5").Code);
            Assert.Equal(File.ReadAllBytes(PathFor("a-model")), File.ReadAllBytes(PathFor("b-model")));
            Assert.Equal(File.ReadAllBytes(PathFor("a-report")), File.ReadAllBytes(PathFor("b-report")));
        }
        finally { CultureInfo.CurrentCulture = previous; }
    }

    [Theory]
    [InlineData("--unknown 1")]
    [InlineData("--lambda")]
    [InlineData("--lambda NaN")]
    [InlineData("--lambda 1,5")]
    [InlineData("--lambda 1 --lambda 2")]
    public void InvalidCliOptionsHaveExitTwoAndNoPartialJson(string options)
    {
        var result = Train(Training(), Validation(), PathFor("model"), PathFor("report"), options.Split(' '));
        Assert.Equal(2, result.Code); Assert.Empty(result.Output); Assert.NotEmpty(result.Error);
        Assert.False(File.Exists(PathFor("model"))); Assert.False(File.Exists(PathFor("report")));
    }

    [Fact]
    public void HelpVersionAndMissingArgumentsBehaveWithoutArtifacts()
    {
        Assert.Equal(0, Run(["--help"]).Code); Assert.Equal("0.2.0", Run(["--version"]).Output.Trim());
        Assert.Equal(2, Run([]).Code); Assert.Equal(2, Run(["evaluate"]).Code); Assert.Equal(2, Run(["predict"]).Code);
        Assert.Equal(2, Run(["predict", "--model", PathFor("missing"), "--data", Training(), "--output", PathFor("out")]).Code);
    }
    private static (int Code, string Output, string Error) Run(string[] args)
    {
        using var output = new StringWriter(); using var error = new StringWriter();
        var code = Application.Run(args, output, error); return (code, output.ToString(), error.ToString());
    }
    public void Dispose() => Directory.Delete(root, recursive: true);
}
