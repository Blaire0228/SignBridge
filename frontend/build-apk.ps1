param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^https://[^\s]+$')]
    [string]$ApiBaseUrl
)

$ErrorActionPreference = 'Stop'
$normalizedApiBaseUrl = $ApiBaseUrl.TrimEnd('/')
$healthUrl = "$normalizedApiBaseUrl/health"

Write-Host "Checking deployed backend: $healthUrl"
$healthResponse = Invoke-WebRequest -UseBasicParsing -Uri $healthUrl -TimeoutSec 15
if ($healthResponse.StatusCode -ne 200) {
    throw "Backend health check failed with HTTP $($healthResponse.StatusCode)."
}

Push-Location $PSScriptRoot
try {
    flutter pub get
    if ($LASTEXITCODE -ne 0) {
        throw 'flutter pub get failed.'
    }

    flutter build apk --release --dart-define="API_BASE_URL=$normalizedApiBaseUrl"
    if ($LASTEXITCODE -ne 0) {
        throw 'flutter build apk failed.'
    }

    Write-Host "APK created at: $PSScriptRoot\build\app\outputs\flutter-apk\app-release.apk"
}
finally {
    Pop-Location
}
