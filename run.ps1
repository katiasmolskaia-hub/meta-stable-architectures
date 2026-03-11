Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
if (-not $projectRoot) {
    $projectRoot = (Get-Location).Path
}

$pythonCandidates = @(
    "C:\Users\User\AppData\Local\Programs\Python\Python314\python.exe",
    "python"
)

$pythonCmd = $null
foreach ($candidate in $pythonCandidates) {
    if ($candidate -eq "python") {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) {
            $pythonCmd = $cmd.Source
            break
        }
    }
    elseif (Test-Path $candidate) {
        $pythonCmd = $candidate
        break
    }
}

if (-not $pythonCmd) {
    throw "Python not found. Install Python and update run.ps1 with the correct path."
}

$mplConfigDir = Join-Path $projectRoot ".matplotlib"
New-Item -ItemType Directory -Force -Path $mplConfigDir | Out-Null

$env:MPLBACKEND = "Agg"
$env:MPLCONFIGDIR = $mplConfigDir

& $pythonCmd (Join-Path $projectRoot "simulations\simulation_v2.py")
exit $LASTEXITCODE
