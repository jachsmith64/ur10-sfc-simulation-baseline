param(
    [string]$Python = 'python',
    [Parameter(Mandatory=$true)][string]$V02Root,
    [string]$PriorResults=(Join-Path $PSScriptRoot 'inputs/previous'),
    [string]$MuJoCoRuntime,
    [string]$ScienceRuntime,
    [string[]]$Trials,
    [Parameter(Mandatory=$true)][string]$Out
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Python -PathType Leaf) { $Python = (Resolve-Path -LiteralPath $Python).Path }
$Out = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Out)
$prior = (Resolve-Path -LiteralPath $PriorResults).Path
if ($MuJoCoRuntime) { $env:B_SFC_RUNTIME_DIR = (Resolve-Path -LiteralPath $MuJoCoRuntime).Path }
if ($ScienceRuntime) { $env:FILTER_SFC_SCIENCE_DIR = (Resolve-Path -LiteralPath $ScienceRuntime).Path }
$env:OPENBLAS_NUM_THREADS = '1'
$opts = @('--output',$Out,'--prior-results',$prior)
if ($Trials) { $opts += @('--trials') + $Trials }
Push-Location -LiteralPath $V02Root
try {
    & $Python -B -I -X utf8 offline.py (Join-Path $PSScriptRoot 'run_preprocess.py') @opts
    if ($LASTEXITCODE -ne 0) { throw "Offline integration exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
