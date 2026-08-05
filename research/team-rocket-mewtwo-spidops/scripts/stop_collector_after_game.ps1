param(
    [Parameter(Mandatory = $true)][int]$CollectorProcessId,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [Parameter(Mandatory = $true)][int]$TargetGame
)

$resolvedLog = [System.IO.Path]::GetFullPath($LogPath)
$marker = '"game": ' + $TargetGame + ','

while ($true) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $CollectorProcessId" -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        exit 0
    }
    if ($process.Name -ne 'python.exe' -or $process.CommandLine -notlike '*collect_live_truth_causal_action_values.py*') {
        throw "PID $CollectorProcessId is not the expected causal collector"
    }
    if (Test-Path -LiteralPath $resolvedLog) {
        $lastLines = Get-Content -LiteralPath $resolvedLog -Tail 4
        if ($lastLines -match [regex]::Escape($marker)) {
            Stop-Process -Id $CollectorProcessId
            exit 0
        }
    }
    Start-Sleep -Seconds 1
}
