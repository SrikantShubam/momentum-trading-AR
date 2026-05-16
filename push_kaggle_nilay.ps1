param(
    [string]$ExpectedOwner = "nilaypurayar0611",
    [string]$SubmissionDir = ".\kaggle_submission",
    [string]$ConfigDir = "."
)

$ErrorActionPreference = "Stop"

$resolvedConfigDir = (Resolve-Path -LiteralPath $ConfigDir).Path
$resolvedSubmissionDir = (Resolve-Path -LiteralPath $SubmissionDir).Path
$env:KAGGLE_CONFIG_DIR = $resolvedConfigDir

$metadataPath = Join-Path $resolvedSubmissionDir "kernel-metadata.json"
if (!(Test-Path -LiteralPath $metadataPath)) {
    throw "Missing kernel metadata at $metadataPath"
}

$metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
if (-not ($metadata.id -like "$ExpectedOwner/*")) {
    throw "Refusing to push: metadata id '$($metadata.id)' is not owned by '$ExpectedOwner'."
}

$mine = kaggle kernels list -m --sort-by dateCreated
$refs = @(
    $mine |
        Select-String -Pattern "^\s*([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)\s+" |
        ForEach-Object { $_.Matches[0].Groups[1].Value }
)

if ($refs.Count -gt 0 -and ($refs | Where-Object { $_ -ne $ExpectedOwner }).Count -gt 0) {
    $owners = ($refs | Sort-Object -Unique) -join ", "
    throw "Refusing to push: Kaggle token resolves to owner(s) '$owners', expected '$ExpectedOwner'."
}

kaggle kernels push -p $resolvedSubmissionDir
