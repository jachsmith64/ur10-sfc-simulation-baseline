param(
    [string]$Python='python',
    [Parameter(Mandatory=$true)][string]$V02Root,
    [Parameter(Mandatory=$true)][string]$Prepared,
    [string]$Output,
    [string]$YOutput,
    [switch]$RefitOnly,
    [string]$MuJoCoRuntime,
    [string]$ScienceRuntime,
    [switch]$PrepareOnly
)
$ErrorActionPreference='Stop'
if (Test-Path -LiteralPath $Python -PathType Leaf) { $Python=(Resolve-Path -LiteralPath $Python).Path }
$Prepared=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Prepared)
if (!$PrepareOnly -and !$Output) { throw 'Specify -Output or -PrepareOnly' }
if ($Output) { $Output=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Output) }
if ($YOutput) { $YOutput=$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($YOutput) }
if ($RefitOnly -and (!$YOutput -or $PrepareOnly)) { throw 'RefitOnly requires -Output with completed X pairs and -YOutput' }
if ($PrepareOnly -and $YOutput) { throw 'PrepareOnly cannot also reconstruct Y' }
if ($MuJoCoRuntime) { $env:B_SFC_RUNTIME_DIR=(Resolve-Path -LiteralPath $MuJoCoRuntime).Path }
if ($ScienceRuntime) { $env:FILTER_SFC_SCIENCE_DIR=(Resolve-Path -LiteralPath $ScienceRuntime).Path }
$env:OPENBLAS_NUM_THREADS='1'
$cli=@('--prepare',$Prepared)
if (!$PrepareOnly) { $cli=@('--use-prepared',$Prepared,'--output',$Output) }
Push-Location -LiteralPath $V02Root
try {
    if (!$RefitOnly) {
        & $Python -B -I -X utf8 offline.py (Join-Path $PSScriptRoot 'run_clock_event.py') @cli
        if ($LASTEXITCODE -ne 0) { throw "Offline exit code $LASTEXITCODE" }
    }
    if ($YOutput) {
        & $Python -B -I -X utf8 offline.py (Join-Path $PSScriptRoot 'refit_y.py') --prepared $Prepared --pairs $Output --output $YOutput
        if ($LASTEXITCODE -ne 0) { throw "Offline Y reconstruction exit code $LASTEXITCODE" }
    }
} finally { Pop-Location }
