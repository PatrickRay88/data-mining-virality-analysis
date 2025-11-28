param(
    [string[]]$Subreddits = @(
        "technology",
        "science",
        "worldnews",
        "business",
        "economy",
        "space",
        "futurology",
        "gadgets",
        "energy",
        "technews",
        "politics"
    ),
    [int]$LoopSeconds = 300,
    [int]$Days = 7,
    [string]$PythonPath = ".\.venv\Scripts\python.exe",
    [string]$LogDirectory = ".\logs"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonPath)) {
    throw "Python path '$PythonPath' was not found. Activate the virtual environment or adjust the -PythonPath parameter."
}

$secondsToRun = $Days * 24 * 60 * 60
if ($LoopSeconds -le 0) {
    throw "LoopSeconds must be greater than zero."
}

$iterations = [int][math]::Floor($secondsToRun / $LoopSeconds)
if ($iterations -lt 1) {
    throw "The combination of Days and LoopSeconds results in fewer than one iteration."
}

if (-not (Test-Path $LogDirectory)) {
    New-Item -ItemType Directory -Path $LogDirectory | Out-Null
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $LogDirectory "snapshot_run_$timestamp.log"

$arguments = @("bin/run_snapshot_collector.py")
foreach ($sub in $Subreddits) {
    $arguments += "-s"
    $arguments += $sub
}
$arguments += "--loop-seconds"
$arguments += $LoopSeconds
$arguments += "--iterations"
$arguments += $iterations

Write-Host "Starting snapshot collector for $Days day(s) ($iterations iterations, loop every $LoopSeconds seconds)."
Write-Host "Logging output to $logFile"

& $PythonPath $arguments 2>&1 | Tee-Object -FilePath $logFile

Write-Host "Snapshot collector finished. Logs available at $logFile"