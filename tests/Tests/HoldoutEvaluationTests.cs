using System.Text;
using System.Text.Json;
using RidgeTrainKit;

namespace Tests;

public sealed class HoldoutEvaluationTests : IDisposable
{
    private readonly string root = Path.Combine(Path.GetTempPath(), "ridge-eval-" + Guid.NewGuid().ToString("N"));
    public HoldoutEvaluationTests() => Directory.CreateDirectory(root);
    private string Write(string text) { var path = Path.Combine(root, Guid.NewGuid().ToString("N") + ".json"); File.WriteAllText(path, text, new UTF8Encoding(false)); return path; }
    private string ModelFile(double coefficient = 2) => Write(Encoding.UTF8.GetString(JsonData.Bytes(new Model(1, "standardized-ridge-v1", 1, ["x"], [0], [1], [coefficient], 1, 3, new string('a', 64)))));
    private string Data(params string[] rows) => Write(string.Join('\n', rows) + "\n");
    private static string Row(string id, double x, double target) => JsonSerializer.Serialize(new { id, features = new { x }, target });
    private static (int Code, string Output, string Error) Run(params string[] args) { using var output = new StringWriter(); using var error = new StringWriter(); var code = Application.Run(args, output, error); return (code, output.ToString(), error.ToString()); }
    private static string[] Args(string model, string data) => ["evaluate", "--model", model, "--data", data];

    [Fact]
    public void SavedModelIsScoredWithHandCheckedMetricsAndNoWrites()
    {
        var model = ModelFile(); var data = Data(Row("private-1", 1, 4), Row("private-2", 2, 7));
        var beforeModel = File.ReadAllBytes(model); var beforeData = File.ReadAllBytes(data); var beforeNames = Directory.GetFileSystemEntries(root).Order().ToArray();
        var result = Run(Args(model, data));
        Assert.Equal(0, result.Code); Assert.Empty(result.Error);
        using var json = JsonDocument.Parse(result.Output); var report = json.RootElement;
        Assert.Equal(2, report.GetProperty("rows").GetInt32());
        Assert.Equal(JsonData.Hash(beforeModel), report.GetProperty("modelSha256").GetString());
        Assert.Equal(JsonData.Hash(beforeData), report.GetProperty("inputSha256").GetString());
        Assert.Equal(1.5, report.GetProperty("metrics").GetProperty("mae").GetDouble(), 12);
        Assert.Equal(Math.Sqrt(2.5), report.GetProperty("metrics").GetProperty("rmse").GetDouble(), 12);
        Assert.Equal(-1.0 / 9, report.GetProperty("metrics").GetProperty("rSquared").GetDouble(), 12);
        Assert.Equal(JsonValueKind.Null, report.GetProperty("maximumRmse").ValueKind);
        Assert.Equal(JsonValueKind.Null, report.GetProperty("passed").ValueKind);
        Assert.Equal(beforeModel, File.ReadAllBytes(model)); Assert.Equal(beforeData, File.ReadAllBytes(data));
        Assert.Equal(beforeNames, Directory.GetFileSystemEntries(root).Order().ToArray());
        Assert.DoesNotContain("private-1", result.Output); Assert.DoesNotContain(Path.GetFileName(model), result.Output);
    }

    [Fact]
    public void GateStrictBoundaryAndConstantTargetAreExplicit()
    {
        var model = ModelFile(); var data = Data(Row("a", 1, 4), Row("b", 2, 4));
        var pass = Run(Args(model, data).Concat(["--max-rmse", "1"]).ToArray());
        Assert.Equal(0, pass.Code);
        using var json = JsonDocument.Parse(pass.Output);
        Assert.Equal(1, json.RootElement.GetProperty("metrics").GetProperty("rmse").GetDouble());
        Assert.Equal(JsonValueKind.Null, json.RootElement.GetProperty("metrics").GetProperty("rSquared").ValueKind);
        Assert.True(json.RootElement.GetProperty("passed").GetBoolean());
        var fail = Run(Args(model, data).Concat(["--max-rmse", "0.9999999999999999"]).ToArray());
        Assert.Equal(1, fail.Code); Assert.Empty(fail.Error);
        using var failedJson = JsonDocument.Parse(fail.Output); Assert.False(failedJson.RootElement.GetProperty("passed").GetBoolean());
        Assert.Equal(0, Run(Args(model, data).Concat(["--max-rmse", "1.0000000000000002"]).ToArray()).Code);
    }

    [Theory]
    [InlineData("-1")]
    [InlineData("NaN")]
    [InlineData("Infinity")]
    [InlineData("1e999")]
    [InlineData("abc")]
    [InlineData("1,2")]
    public void InvalidGatesExitTwoWithoutPartialReport(string value)
    {
        var result = Run("evaluate", "--model", "unused", "--data", "unused", "--max-rmse", value);
        Assert.Equal(2, result.Code); Assert.Empty(result.Output); Assert.NotEmpty(result.Error);
    }

    [Theory]
    [InlineData("{\"id\":\"a\",\"features\":{\"x\":1}}")]
    [InlineData("{\"id\":\"a\",\"features\":{\"X\":1},\"target\":2}")]
    [InlineData("{\"id\":\"a\",\"features\":{\"x\":1},\"target\":null}")]
    [InlineData("{\"id\":\"a\",\"features\":{\"x\":1},\"target\":2,\"target\":3}")]
    [InlineData("{\"id\":\"\\ud800\",\"features\":{\"x\":1},\"target\":2}")]
    public void InvalidLabeledDataExitsTwoWithoutReport(string row)
    {
        var model = ModelFile(); var data = Data(row); var result = Run(Args(model, data));
        Assert.Equal(2, result.Code); Assert.Empty(result.Output); Assert.NotEmpty(result.Error);
    }

    [Fact]
    public void InvalidModelsNumericOverflowAndOptionsFailWithoutArtifacts()
    {
        var data = Data(Row("a", 2, 1)); var malformed = Write("{}");
        Assert.Equal(2, Run(Args(malformed, data)).Code);
        Assert.Equal(2, Run(Args(ModelFile(1e308), data)).Code);
        Assert.Equal(2, Run("evaluate", "--model", malformed, "--data", data, "--output", "x").Code);
        Assert.Equal(2, Run("evaluate", "--model", malformed, "--data", data, "--max-rmse", "1", "--max-rmse", "2").Code);
        var result = Run("evaluate", "--model", malformed, "--data", data);
        Assert.Empty(result.Output); Assert.NotEmpty(result.Error);
        var invalidUnicode = Write(Encoding.UTF8.GetString(JsonData.Bytes(new Model(1, "standardized-ridge-v1", 1, ["x"], [0], [1], [2], 1, 3, new string('a', 64)))).Replace("standardized-ridge-v1", "\\ud800"));
        var invalidModel = Run(Args(invalidUnicode, data)); Assert.Equal(2, invalidModel.Code); Assert.Empty(invalidModel.Output);
    }

    [Fact]
    public void ChangingHoldoutCannotModifyModelOrRefitScaling()
    {
        var model = ModelFile(); var before = File.ReadAllBytes(model);
        var first = Data(Row("a", 1, 4), Row("b", 2, 4));
        var second = Data(Row("a", 1, 3), Row("b", 2, 5));
        var firstResult = Run(Args(model, first)); var secondResult = Run(Args(model, second));
        using var a = JsonDocument.Parse(firstResult.Output); using var b = JsonDocument.Parse(secondResult.Output);
        Assert.Equal(1, a.RootElement.GetProperty("metrics").GetProperty("rmse").GetDouble());
        Assert.Equal(0, b.RootElement.GetProperty("metrics").GetProperty("rmse").GetDouble());
        Assert.Equal(before, File.ReadAllBytes(model));
        Assert.Equal(JsonData.Hash(before), a.RootElement.GetProperty("modelSha256").GetString());
        Assert.Equal(JsonData.Hash(before), b.RootElement.GetProperty("modelSha256").GetString());
    }
    public void Dispose() => Directory.Delete(root, true);
}
