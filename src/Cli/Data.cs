using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace RidgeTrainKit;

public sealed class InputException(string message) : Exception(message);
public sealed record DataRow(string Id, double[] Features, double? Target);
public sealed record Dataset(string[] FeatureNames, DataRow[] Rows, string Sha256);

public static class JsonData
{
    public static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        RespectRequiredConstructorParameters = true
    };
    public static byte[] Bytes<T>(T value) => Encoding.UTF8.GetBytes(JsonSerializer.Serialize(value, Options) + "\n");
    public static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    public static (byte[] Bytes, string Text) Snapshot(string path, int limit)
    {
        using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (file.Length > limit) throw new InputException("File exceeds the documented byte limit.");
        using var memory = new MemoryStream();
        var buffer = new byte[8192]; int count;
        while ((count = file.Read(buffer)) > 0)
        {
            if (memory.Length + count > limit) throw new InputException("File exceeds the documented byte limit.");
            memory.Write(buffer, 0, count);
        }
        var bytes = memory.ToArray();
        try
        {
            var offset = bytes.AsSpan().StartsWith(new byte[] { 0xef, 0xbb, 0xbf }) ? 3 : 0;
            return (bytes, new UTF8Encoding(false, true).GetString(bytes, offset, bytes.Length - offset));
        }
        catch (DecoderFallbackException) { throw new InputException("Input must be valid UTF-8; an optional UTF-8 BOM is accepted."); }
    }
    public static void RejectDuplicates(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.Object)
        {
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (var property in element.EnumerateObject())
            {
                if (!names.Add(property.Name)) throw new JsonException();
                RejectDuplicates(property.Value);
            }
        }
        else if (element.ValueKind == JsonValueKind.Array)
            foreach (var child in element.EnumerateArray()) RejectDuplicates(child);
    }
}

public static class DataReader
{
    public const int MaxBytes = 32 * 1024 * 1024;
    public const int MaxRows = 50_000;
    public static Dataset Read(string path, bool requireTarget, string[]? expectedNames = null)
    {
        var snapshot = JsonData.Snapshot(path, MaxBytes);
        var names = expectedNames;
        var ids = new HashSet<string>(StringComparer.Ordinal);
        var rows = new List<DataRow>();
        using var reader = new StringReader(snapshot.Text);
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            var number = rows.Count + 1;
            if (rows.Count >= MaxRows || line.Length > 1024 * 1024) throw new InputException("Dataset exceeds the row-count or line-length limit.");
            try
            {
                using var document = JsonDocument.Parse(line);
                var root = document.RootElement;
                if (root.ValueKind != JsonValueKind.Object) throw new JsonException();
                JsonData.RejectDuplicates(root);
                if (!root.TryGetProperty("id", out var idNode) || idNode.ValueKind != JsonValueKind.String) throw new JsonException();
                var id = idNode.GetString()!;
                if (string.IsNullOrWhiteSpace(id) || id.Length > 256 || id.Any(char.IsControl)) throw new JsonException();
                if (!ids.Add(id)) throw new InputException($"Repeated record ID at line {number}.");
                if (!root.TryGetProperty("features", out var features) || features.ValueKind != JsonValueKind.Object) throw new JsonException();
                var actual = features.EnumerateObject().Select(property => property.Name).Order(StringComparer.Ordinal).ToArray();
                if (actual.Length is < 1 or > 32 || actual.Any(name => string.IsNullOrWhiteSpace(name) || name.Length > 64 || name.Any(char.IsControl))) throw new JsonException();
                names ??= actual;
                if (!names.SequenceEqual(actual, StringComparer.Ordinal)) throw new InputException($"Feature schema differs at line {number}; names are exact and case-sensitive.");
                var values = names.Select(name => Number(features.GetProperty(name))).ToArray();
                double? target = null;
                if (root.TryGetProperty("target", out var targetNode)) target = Number(targetNode);
                else if (requireTarget) throw new JsonException();
                rows.Add(new(id, values, target));
            }
            catch (JsonException) { throw new InputException($"Invalid JSON object, ID, numeric features or target at line {number}; duplicate properties are prohibited."); }
        }
        if (rows.Count == 0) throw new InputException("Dataset is empty.");
        return new(names!, rows.ToArray(), JsonData.Hash(snapshot.Bytes));
    }
    private static double Number(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Number || !element.TryGetDouble(out var value) || !double.IsFinite(value) || Math.Abs(value) > 1e12) throw new JsonException();
        return value;
    }
}
