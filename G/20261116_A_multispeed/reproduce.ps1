param(
    [string]$Python = 'python',
    [string]$V02Root = (Join-Path $PSScriptRoot 'local/frozen_V02'),
    [string]$MuJoCoRuntime,
    [string]$ScienceRuntime,
    [string]$SourceRoot,
    [string]$PreparedRoot,
    [string]$HistoricalRuns,
    [string[]]$Trials,
    [switch]$WithControls,
    [Parameter(Mandatory=$true)][string]$Out
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Python -PathType Leaf) { $Python = (Resolve-Path -LiteralPath $Python).Path }
$Out = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Out)
if ($MuJoCoRuntime) { $env:B_SFC_RUNTIME_DIR = (Resolve-Path -LiteralPath $MuJoCoRuntime).Path }
if ($ScienceRuntime) { $env:FILTER_SFC_SCIENCE_DIR = (Resolve-Path -LiteralPath $ScienceRuntime).Path }
$env:OPENBLAS_NUM_THREADS = '1'
$entry = Join-Path $PSScriptRoot 'run_g.py'
$arguments = @('--out',$Out)
if ($HistoricalRuns) {
    $entry = Join-Path $PSScriptRoot 'regress_v02.py'
    $arguments += @('--historical',(Resolve-Path -LiteralPath $HistoricalRuns).Path)
} else {
    if ($SourceRoot) { $arguments += @('--source-root',(Resolve-Path -LiteralPath $SourceRoot).Path) }
    if ($PreparedRoot) { $arguments += @('--prepared-root',(Resolve-Path -LiteralPath $PreparedRoot).Path) }
    if ($Trials) { $arguments += @('--trials') + $Trials }
    if ($WithControls) { $arguments += '--with-controls' }
}
Push-Location -LiteralPath $V02Root
try {
    & $Python -B -I -X utf8 offline.py $entry @arguments
    if ($LASTEXITCODE -ne 0) { throw "Offline run failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
