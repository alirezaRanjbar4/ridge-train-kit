namespace RidgeTrainKit;

public static class Artifacts
{
    public static void Preflight(params string[] paths)
    {
        var full = paths.Select(Path.GetFullPath).ToArray();
        var comparer = OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
        if (full.Distinct(comparer).Count() != paths.Length || full.Any(path => File.Exists(path) || Directory.Exists(path)))
            throw new InputException("Artifact destinations must be distinct fresh file paths.");
    }
    public static void Write(params (string Path, byte[] Bytes)[] files)
    {
        Preflight(files.Select(file => file.Path).ToArray());
        var created = new List<string>();
        try
        {
            foreach (var item in files)
            {
                var path = Path.GetFullPath(item.Path);
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                using var file = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.None);
                created.Add(path); file.Write(item.Bytes);
            }
        }
        catch
        {
            foreach (var path in created)
            {
                try { File.Delete(path); } catch (IOException) { } catch (UnauthorizedAccessException) { }
            }
            throw;
        }
    }
}
