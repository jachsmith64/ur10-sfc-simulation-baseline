param(
    [string]$Python='python',
    [Parameter(Mandatory=$true)][string]$Bound,
    [Parameter(Mandatory=$true)][string]$Output,
    [Parameter(Mandatory=$true)][string]$MuJoCoRuntime,
    [Parameter(Mandatory=$true)][string]$ScienceRuntime,
    [ValidateSet('X_P05_R01','X_P05_R02','X_P10_R01')][string]$NOnly,
    [switch]$ResumeFirstReceipt
)
$ErrorActionPreference='Stop'
if (Test-Path -LiteralPath $Python -PathType Leaf) { $Python=(Resolve-Path -LiteralPath $Python).Path }
$Bound=(Resolve-Path -LiteralPath $Bound).Path
$Output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output)
$env:B_SFC_RUNTIME_DIR=(Resolve-Path -LiteralPath $MuJoCoRuntime).Path
$env:FILTER_SFC_SCIENCE_DIR=(Resolve-Path -LiteralPath $ScienceRuntime).Path
$env:OPENBLAS_NUM_THREADS='1'
$cli=@('--bound',$Bound,'--output',$Output)
if ($NOnly) { $cli+=@('--n-only',$NOnly) }
if ($ResumeFirstReceipt) { $cli+='--resume-first-receipt' }
Push-Location -LiteralPath (Join-Path $Bound 'v02_runtime')
try {
    & $Python -B -I -X utf8 offline.py (Join-Path $PSScriptRoot 'run_nls.py') @cli
    if ($LASTEXITCODE -ne 0) { throw "Offline integration exit code $LASTEXITCODE" }
} finally { Pop-Location }
