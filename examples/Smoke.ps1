param([string]$ExpectedCommit)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Push-Location $root
$previousCache = $env:NUGET_PACKAGES
try {
    if (-not $ExpectedCommit) {
        $ExpectedCommit = git rev-parse HEAD
        if ($LASTEXITCODE -ne 0) { throw 'Cannot determine expected Git commit.' }
    }
    $runDirectory = Join-Path $root ('artifacts/smoke-' + [guid]::NewGuid().ToString('N'))
    $toolDirectory = Join-Path $runDirectory 'tool'
    $env:NUGET_PACKAGES = Join-Path $runDirectory 'packages'
    dotnet tool install RidgeTrainKit.Tool --version 0.2.0 --tool-path $toolDirectory --configfile nuget.local.config --no-cache
    if ($LASTEXITCODE -ne 0) { throw 'Local tool installation failed.' }
    $tool = Join-Path $toolDirectory 'ridge-train'
    if ((& $tool --version) -ne '0.2.0') { throw 'Installed version mismatch.' }
    $inputs = @('examples/train.jsonl','examples/validation.jsonl','examples/predict.jsonl','examples/holdout.jsonl')
    $before = @($inputs | ForEach-Object { (Get-FileHash $_ -Algorithm SHA256).Hash })
    $model = Join-Path $runDirectory 'model.json'; $reportPath = Join-Path $runDirectory 'training.json'
    $trained = & $tool train --train $inputs[0] --validation $inputs[1] --model $model --report $reportPath --lambda 1 | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or $trained.trainingRows -ne 8 -or $trained.validationRows -ne 4 -or $trained.featureNames.Count -ne 2) { throw 'Installed numeric training failed.' }
    if ((Get-FileHash $model -Algorithm SHA256).Hash.ToLowerInvariant() -ne $trained.modelSha256) { throw 'Model provenance mismatch.' }
    $modelHash = (Get-FileHash $model -Algorithm SHA256).Hash
    $predictions = Join-Path $runDirectory 'predictions.jsonl'
    $predictionReport = & $tool predict --model $model --data $inputs[2] --output $predictions | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or $predictionReport.rows -ne 2 -or $predictionReport.outputSha256 -ne (Get-FileHash $predictions -Algorithm SHA256).Hash.ToLowerInvariant()) { throw 'Batch prediction/provenance failed.' }
    $first = Get-Content $predictions -TotalCount 1 | ConvertFrom-Json
    if ($first.id -ne 'query-01' -or [Math]::Abs($first.prediction - 16.38888888888889) -gt 1e-9) { throw 'Hand-derived ridge prediction mismatch.' }
    $independent = dotnet run --project examples/StandaloneScorer -c Release --no-restore -- $model examples/features.json | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or [Math]::Abs($independent.prediction - $first.prediction) -gt 1e-12) { throw 'Independent BCL scorer disagrees.' }
    $evaluation = & $tool evaluate --model $model --data $inputs[3] | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or $evaluation.rows -ne 2 -or $evaluation.modelSha256 -ne $modelHash.ToLowerInvariant() -or $evaluation.inputSha256 -ne $before[3].ToLowerInvariant() -or $null -ne $evaluation.maximumRmse -or $null -ne $evaluation.passed) { throw 'Holdout evaluation/hash failed.' }
    $threshold = $evaluation.metrics.rmse.ToString('R',[Globalization.CultureInfo]::InvariantCulture)
    $equal = & $tool evaluate --model $model --data $inputs[3] --max-rmse $threshold | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or -not $equal.passed) { throw 'Equal RMSE gate did not pass.' }
    $failed = & $tool evaluate --model $model --data $inputs[3] --max-rmse 0 | ConvertFrom-Json
    if ($LASTEXITCODE -ne 1 -or $failed.passed) { throw 'Failing RMSE gate not detected.' }
    $invalidEvaluation = & $tool evaluate --model $model --data $inputs[3] --max-rmse NaN 2> (Join-Path $runDirectory 'gate-error.txt')
    if ($LASTEXITCODE -ne 2 -or $invalidEvaluation) { throw 'Invalid gate produced partial output.' }
    $constantEvaluation = & $tool evaluate --model $model --data examples/constant-validation.jsonl | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or $null -ne $constantEvaluation.metrics.rSquared) { throw 'Holdout constant-target R-squared failed.' }
    $constant = & $tool train --train $inputs[0] --validation examples/constant-validation.jsonl --model (Join-Path $runDirectory 'constant-model.json') --report (Join-Path $runDirectory 'constant-report.json') | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0 -or -not ($constant.validation.PSObject.Properties.Name -ccontains 'rSquared') -or $null -ne $constant.validation.rSquared) { throw 'Constant-target R-squared contract failed.' }
    $refused = Join-Path $runDirectory 'refused.jsonl'
    & $tool predict --model $model --data examples/invalid-features.jsonl --output $refused 2> (Join-Path $runDirectory 'schema-error.txt')
    if ($LASTEXITCODE -ne 2 -or (Test-Path -LiteralPath $refused)) { throw 'Schema error produced an output.' }
    & $tool predict --model $model --data $inputs[2] --output $predictions 2> (Join-Path $runDirectory 'collision-error.txt')
    if ($LASTEXITCODE -ne 2) { throw 'Existing prediction output was not refused.' }
    if ((Get-FileHash $model -Algorithm SHA256).Hash -ne $modelHash -or ($before -join ',') -ne (@($inputs | ForEach-Object { (Get-FileHash $_ -Algorithm SHA256).Hash }) -join ',')) { throw 'Source/model changed.' }
    $zip = [IO.Compression.ZipFile]::OpenRead((Join-Path $root 'artifacts/RidgeTrainKit.Tool.0.2.0.nupkg'))
    try {
        $reader = [IO.StreamReader]::new($zip.GetEntry('RidgeTrainKit.Tool.nuspec').Open())
        try { [xml]$manifest = $reader.ReadToEnd() } finally { $reader.Dispose() }
        if ($manifest.package.metadata.id -ne 'RidgeTrainKit.Tool' -or $manifest.package.metadata.version -ne '0.2.0' -or $manifest.package.metadata.repository.commit -ne $ExpectedCommit) { throw 'Final package metadata mismatch.' }
        foreach ($entry in @('README.md','LICENSE','CONTRIBUTING.md','ROADMAP.md','CHANGELOG.md','docs/DESIGN.md','docs/RESEARCH.md','docs/TROUBLESHOOTING.md','docs/HOLDOUT.md','examples/holdout.jsonl','examples/README.md','examples/StandaloneScorer/Program.cs','examples/StandaloneScorer/StandaloneScorer.csproj')) {
            if (-not $zip.GetEntry($entry)) { throw 'Missing bundled guide or independent scorer source.' }
        }
    } finally { $zip.Dispose() }
    $trained | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $runDirectory 'verified-training.json') -Encoding utf8
    Write-Output ('Installed numeric-model smoke passed; evidence: ' + $runDirectory)
    $global:LASTEXITCODE = 0
} finally { $env:NUGET_PACKAGES = $previousCache; Pop-Location }
