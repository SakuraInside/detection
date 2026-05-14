# Запуск backend_gateway: корень проекта, integra_ffi.dll (TensorRT), PATH к CUDA/TRT/OpenCV.
# Запуск из корня репозитория:  .\scripts\run_backend_gateway.ps1
# Переопределение путей: -TensorRtRoot "D:\TRT" -CudaRoot "..." -OpenCvBin "..." -FfiDll "...\integra_ffi.dll"

param(
    [string] $RepoRoot = "",
    [string] $TensorRtRoot = "C:\TensorRT-10.16.1.11",
    [string] $CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
    [string] $OpenCvBin = "C:\build\opencv\x64\vc17\bin",
    [string] $FfiDll = ""
)

$ErrorActionPreference = "Stop"

if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$RepoRoot = (Resolve-Path $RepoRoot).Path

if (-not $FfiDll) {
    $candidates = @(
        (Join-Path $RepoRoot "native\build-msvc-trt-user\Release\integra_ffi.dll"),
        (Join-Path $RepoRoot "native\build-msvc-trt-user\RelWithDebInfo\integra_ffi.dll"),
        (Join-Path $RepoRoot "native\build-msvc\Release\integra_ffi.dll"),
        (Join-Path $RepoRoot "native\build-msvc\RelWithDebInfo\integra_ffi.dll")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $FfiDll = $c
            break
        }
    }
}
if (-not $FfiDll -or -not (Test-Path $FfiDll)) {
    Write-Error "Не найден integra_ffi.dll. Соберите native (см. native/README.md) или укажите -FfiDll."
}

$trtBin = Join-Path $TensorRtRoot "bin"
if (-not (Test-Path $trtBin)) {
    Write-Warning "Нет каталога TensorRT bin: $trtBin (проверьте -TensorRtRoot)"
}

$env:INTEGRA_PROJECT_ROOT = $RepoRoot
$env:INTEGRA_FFI_PATH = (Resolve-Path $FfiDll).Path

$prepend = @()
if (Test-Path $trtBin) { $prepend += $trtBin }
if (Test-Path $CudaRoot) {
    $prepend += (Join-Path $CudaRoot "bin")
}
if (Test-Path $OpenCvBin) { $prepend += $OpenCvBin }
$env:PATH = ($prepend -join ";") + ";" + $env:PATH

$rt = Join-Path $RepoRoot "runtime-core"
if (-not (Test-Path $rt)) {
    Write-Error "Нет каталога runtime-core: $rt"
}

$config = Join-Path $RepoRoot "config.json"
if (-not (Test-Path $config)) {
    Write-Warning "Нет config.json в $RepoRoot"
}

Write-Host "INTEGRA_PROJECT_ROOT=$env:INTEGRA_PROJECT_ROOT"
Write-Host "INTEGRA_FFI_PATH=$env:INTEGRA_FFI_PATH"
Write-Host "Для видео отдельно запустите video-bridge (TCP 9876) или используйте python run.py --release." -ForegroundColor DarkGray
Write-Host "Запуск cargo из $rt ..." -ForegroundColor Cyan

Push-Location $rt
try {
    cargo run --release --bin backend_gateway
}
finally {
    Pop-Location
}
