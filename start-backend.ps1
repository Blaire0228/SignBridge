param(
    [int]$GemmaStartupTimeoutSeconds = 900
)

$ErrorActionPreference = 'Stop'
$projectDir = $PSScriptRoot
$gemmaDir = Join-Path $projectDir 'Gemma4-TSL-RAG'
$rootPython = Join-Path $projectDir 'venv\Scripts\python.exe'
$gemmaPython = Join-Path $gemmaDir '.venv\Scripts\python.exe'
$logDir = Join-Path $projectDir 'logs'
$processFile = Join-Path $projectDir '.backend-processes.json'
$startedProcesses = @()

function Wait-Health {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][int]$TimeoutSeconds,
        [System.Diagnostics.Process]$Process
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if ($null -ne $Process -and $Process.HasExited) {
            throw "$Url process exited with code $($Process.ExitCode). Check the logs folder."
        }

        try {
            $response = Invoke-RestMethod -Uri $Url -TimeoutSec 5
            if ($response.status -eq 'ok') {
                return
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }

    throw "Timed out waiting for $Url. Check the logs folder."
}

function Test-LocalPort {
    param([Parameter(Mandatory = $true)][int]$Port)
    return $null -ne (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
}

function Test-ApiTitle {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$ExpectedTitle
    )
    try {
        $schema = Invoke-RestMethod -Uri "$Url/openapi.json" -TimeoutSec 5
        return $schema.info.title -eq $ExpectedTitle
    }
    catch {
        return $false
    }
}

if (-not (Test-Path -LiteralPath $rootPython)) {
    throw "Root backend Python was not found: $rootPython"
}
if (-not (Test-Path -LiteralPath $gemmaPython)) {
    throw "Gemma Python was not found: $gemmaPython"
}
if (-not (Test-Path -LiteralPath (Join-Path $gemmaDir '.env'))) {
    throw 'Gemma4-TSL-RAG\.env is missing.'
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

try {
    $postgres = Get-Service -Name 'postgresql*' -ErrorAction Stop | Select-Object -First 1
    if ($postgres.Status -ne 'Running') {
        Start-Service -Name $postgres.Name
        $postgres.WaitForStatus('Running', [TimeSpan]::FromSeconds(30))
    }
    Write-Host "PostgreSQL is running ($($postgres.Name))."

    $gemmaProcess = $null
    if (Test-LocalPort -Port 8001) {
        if (-not (Test-ApiTitle -Url 'http://127.0.0.1:8001' -ExpectedTitle 'Gemma TSL RAG API')) {
            throw 'Port 8001 is occupied by a process that is not the Gemma API.'
        }
        Write-Host 'Gemma API is already listening on http://127.0.0.1:8001.'
    }
    else {
        Write-Host 'Starting Gemma API. Initial model loading can take several minutes...'
        # Both models are already cached on this host. Offline mode avoids a
        # startup failure when Hugging Face is temporarily unreachable.
        $env:HF_HUB_OFFLINE = '1'
        $env:TRANSFORMERS_OFFLINE = '1'
        $gemmaProcess = Start-Process `
            -FilePath $gemmaPython `
            -ArgumentList @('-u', 'Gemma_API.py') `
            -WorkingDirectory $gemmaDir `
            -RedirectStandardOutput (Join-Path $logDir 'gemma.out.log') `
            -RedirectStandardError (Join-Path $logDir 'gemma.err.log') `
            -WindowStyle Hidden `
            -PassThru
        $startedProcesses += $gemmaProcess
    }
    Wait-Health -Url 'http://127.0.0.1:8001/health' -TimeoutSeconds $GemmaStartupTimeoutSeconds -Process $gemmaProcess
    Write-Host 'Gemma API is healthy.'

    $mainProcess = $null
    if (Test-LocalPort -Port 8000) {
        if (-not (Test-ApiTitle -Url 'http://127.0.0.1:8000' -ExpectedTitle '雙向手語翻譯 API')) {
            throw 'Port 8000 is occupied by a process that is not the main API.'
        }
        Write-Host 'Main API is already listening on http://127.0.0.1:8000.'
    }
    else {
        Write-Host 'Starting main API...'
        $env:GEMMA_API_URL = 'http://127.0.0.1:8001'
        $mainProcess = Start-Process `
            -FilePath $rootPython `
            -ArgumentList @('-u', '-m', 'uvicorn', 'main:app', '--host', '0.0.0.0', '--port', '8000') `
            -WorkingDirectory $projectDir `
            -RedirectStandardOutput (Join-Path $logDir 'backend.out.log') `
            -RedirectStandardError (Join-Path $logDir 'backend.err.log') `
            -WindowStyle Hidden `
            -PassThru
        $startedProcesses += $mainProcess
    }
    Wait-Health -Url 'http://127.0.0.1:8000/health' -TimeoutSeconds 60 -Process $mainProcess
    Write-Host 'Main API is healthy.'

    $processRecords = @()
    if (Test-Path -LiteralPath $processFile) {
        $existingRecords = Get-Content -LiteralPath $processFile -Raw | ConvertFrom-Json
        foreach ($record in $existingRecords) {
            $processRecords += $record
        }
    }
    $newProcessRecords = $startedProcesses | ForEach-Object {
        [pscustomobject]@{
            id = $_.Id
            name = $_.ProcessName
            startedAt = $_.StartTime.ToUniversalTime().ToString('o')
        }
    }
    foreach ($record in $newProcessRecords) {
        $processRecords += $record
    }
    if ($processRecords.Count -gt 0) {
        $processRecords | ConvertTo-Json | Set-Content -LiteralPath $processFile -Encoding utf8
    }

    Write-Host ''
    Write-Host 'Backend is ready at http://127.0.0.1:8000'
    Write-Host 'Android phone (same Wi-Fi): http://<YOUR_PC_LAN_IP>:8000'
    Write-Host "Logs: $logDir"
}
catch {
    foreach ($process in $startedProcesses) {
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
    throw
}
