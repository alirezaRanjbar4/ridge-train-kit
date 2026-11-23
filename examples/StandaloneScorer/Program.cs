using System.Text.Json;

// Formula demonstration for an already-validated, trusted v1 model. Not a model validator.
if (args is not [var modelPath, var featuresPath])
{
    Console.Error.WriteLine("Pass a validated model JSON and a single feature-object JSON file.");
    return 2;
}
using var model = JsonDocument.Parse(File.ReadAllText(modelPath));
using var features = JsonDocument.Parse(File.ReadAllText(featuresPath));
var root = model.RootElement;
var names = root.GetProperty("featureNames").EnumerateArray().Select(item => item.GetString()!).ToArray();
var means = root.GetProperty("means").EnumerateArray().Select(item => item.GetDouble()).ToArray();
var scales = root.GetProperty("scales").EnumerateArray().Select(item => item.GetDouble()).ToArray();
var coefficients = root.GetProperty("coefficients").EnumerateArray().Select(item => item.GetDouble()).ToArray();
var prediction = root.GetProperty("targetMean").GetDouble();
for (var j = 0; j < names.Length; j++) prediction += coefficients[j] * ((features.RootElement.GetProperty(names[j]).GetDouble() - means[j]) / scales[j]);
Console.WriteLine(JsonSerializer.Serialize(new { prediction }));
return 0;
