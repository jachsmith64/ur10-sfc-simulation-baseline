param(
    [string]$Python='python',
    [Parameter(Mandatory=$true)][string]$V02Root,
    [Parameter(Mandatory=$true)][string]$Out,
    [string]$MuJoCoRuntime,
    [string]$ScienceRuntime,
    [string]$BaselineRuns
)
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Python -PathType Leaf) { $Python=(Resolve-Path -LiteralPath $Python).Path }
$Out=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Out)
if ($MuJoCoRuntime) { $env:B_SFC_RUNTIME_DIR=(Resolve-Path -LiteralPath $MuJoCoRuntime).Path }
if ($ScienceRuntime) { $env:FILTER_SFC_SCIENCE_DIR=(Resolve-Path -LiteralPath $ScienceRuntime).Path }
$env:OPENBLAS_NUM_THREADS='1'
$argvList=@('--output',$Out)
if ($BaselineRuns) { $argvList+=@('--baseline-runs',(Resolve-Path -LiteralPath $BaselineRuns).Path) }
Push-Location -LiteralPath $V02Root
try {
    & $Python -B -I -X utf8 offline.py (Join-Path $PSScriptRoot 'run_servo.py') @argvList
    if ($LASTEXITCODE -ne 0) { throw "Offline integration exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
