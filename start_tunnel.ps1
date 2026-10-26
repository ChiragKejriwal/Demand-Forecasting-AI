# Demand Forecasting - Local Server & Cloudflare Tunnel Launcher
param(
    [int]$Port = 8503
)

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Launching Demand Forecasting Platform & Cloudflare Tunnel" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RootDir = Split-Path -Parent $ScriptDir
$PythonExe = Join-Path $RootDir "venv\Scripts\python.exe"
$CloudflaredExe = Join-Path $RootDir "cloudflared.exe"

if (-not (Test-Path $CloudflaredExe)) {
    $CloudflaredExe = "cloudflared"
}

Write-Host "`n[1/2] Starting Streamlit Server on port $Port..." -ForegroundColor Yellow
$StreamlitProc = Start-Process -FilePath $PythonExe -ArgumentList "-m streamlit run app.py --server.port $Port" -WorkingDirectory $ScriptDir -PassThru

Start-Sleep -Seconds 3

Write-Host "[2/2] Starting Cloudflare Tunnel to http://localhost:$Port..." -ForegroundColor Yellow
& $CloudflaredExe tunnel --url "http://localhost:$Port"
