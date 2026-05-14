# Starts video-bridge (TCP 9876) then backend_gateway (HTTP 8000); same PATH as run_backend_gateway.
param(
    [string] $RepoRoot = "",
    [string] $TensorRtRoot = "C:\TensorRT-10.16.1.11",
    [string] $CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
    [string] $OpenCvBin = "C:\build\opencv\x64\vc17\bin"
)

$ErrorActionPreference = "Stop"
if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$RepoRoot = (Resolve-Path $RepoRoot).Path

$prepend = @()
if (Test-Path (Join-Path $TensorRtRoot "bin")) { $prepend += (Join-Path $TensorRtRoot "bin") }
if (Test-Path $CudaRoot) { $prepend += (Join-Path $CudaRoot "bin") }
if (Test-Path $OpenCvBin) { $prepend += $OpenCvBin }
$env:PATH = ($prepend -join ";") + ";" + $env:PATH

$env:INTEGRA_PROJECT_ROOT = $RepoRoot
$ffi = Join-Path $RepoRoot "runtime-core\target\release\integra_ffi.dll"
if (-not (Test-Path $ffi)) {
    $ffi = Join-Path $RepoRoot "native\build-msvc-trt-user\Release\integra_ffi.dll"
}
if (-not (Test-Path $ffi)) {
    Write-Error "integra_ffi.dll not found. Run .\scripts\build_all.ps1"
}
$env:INTEGRA_FFI_PATH = (Resolve-Path $ffi).Path

$bridgeExe = Join-Path $RepoRoot "video-bridge\target\release\video-bridge.exe"
$gwExe = Join-Path $RepoRoot "runtime-core\target\release\backend_gateway.exe"
if (-not (Test-Path $bridgeExe) -or -not (Test-Path $gwExe)) {
    Write-Error "Missing binaries. Run .\scripts\build_all.ps1"
}

Write-Host "Starting video-bridge..." -ForegroundColor Cyan
$bp = Start-Process -FilePath $bridgeExe -WorkingDirectory $RepoRoot -PassThru -NoNewWindow
Start-Sleep -Seconds 1
if ($bp.HasExited -and $bp.ExitCode -ne 0) {
    Write-Error "video-bridge exited immediately (code $($bp.ExitCode))"
}

Write-Host "Starting backend_gateway (Ctrl+C stops both)..." -ForegroundColor Cyan
try {
    & $gwExe
}
finally {
    if (-not $bp.HasExited) {
        Stop-Process -Id $bp.Id -Force -ErrorAction SilentlyContinue
    }
}
