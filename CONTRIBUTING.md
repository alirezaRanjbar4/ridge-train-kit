# Contributing

Use a focused synthetic reproduction. Read the [design](docs/DESIGN.md), [troubleshooting](docs/TROUBLESHOOTING.md) and [bounded roadmap](ROADMAP.md). Keep private data/company code/credentials out of issues and source. Feature names, model statistics and prediction IDs can also be sensitive.

From the root with the .NET 10 SDK:

```sh
dotnet restore
dotnet test -c Release --no-restore
dotnet format --verify-no-changes --no-restore
dotnet pack src/Cli -c Release --no-restore -o artifacts
pwsh -File examples/Smoke.ps1
```

Smoke needs PowerShell 7, available on GitHub-hosted runners. It installs from the local feed, exercises real fitting and batch reload, compares an independent BCL scorer, checks hand-derived predictions/null R², refusal/source hashes and packaged commit/docs. Commit the intended source and repack first; stale commit metadata should fail its final assertion.

Tests use hand-derived one/two-feature fits and an independently calculated stationarity condition, heldout isolation, exact schemas, saved-model round trips, metrics, provenance, numeric errors and no-overwrite behavior. Do not merely mirror solver lines in tests. Keep tolerances justified. Changes to the model/algorithm contract require explicit compatibility/version decisions and docs.

The runtime has no external package reference. Test packages are not distributed with the tool. No native ML library, Python, network, model API or paid resource belongs in this scope. Contributions use the [MIT license](LICENSE).
