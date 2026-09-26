$ErrorActionPreference = 'Stop'
$processFile = Join-Path $PSScriptRoot '.backend-processes.json'

$records = @()
if (Test-Path -LiteralPath $processFile) {
    $storedRecords = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
    foreach ($storedRecord in $storedRecords) { $records += $storedRecord }
}
foreach ($record in $records) {
    $process = Get-Process -Id $record.id -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        continue
    }

    try {
        $expectedStart = [DateTimeOffset]::Parse(
            [string]$record.startedAt,
            [Globalization.CultureInfo]::InvariantCulture,
            [Globalization.DateTimeStyles]::RoundtripKind
        ).UtcDateTime
    }
    catch {
        Write-Warning "PID $($record.id) has an invalid startedAt value; listener cleanup will continue."
        continue
    }
    $actualStart = $process.StartTime.ToUniversalTime()
    if ([Math]::Abs(($actualStart - $expectedStart).TotalSeconds) -gt 2) {
        Write-Warning "PID $($record.id) was reused; it will not be stopped."
        continue
    }

    Stop-Process -Id $process.Id -Force
    Write-Host "Stopped $($record.name) (PID $($record.id))."
}

# A Windows venv launcher can leave its child Python process listening.
foreach ($port in 8000, 8001) {
    $connection = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($connection) {
        Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "Stopped listener on port $port (PID $($connection.OwningProcess))."
    }
}

Remove-Item -LiteralPath $processFile -Force -ErrorAction SilentlyContinue
