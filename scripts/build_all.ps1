# Полная сборка: native integra_ffi (TensorRT) + backend_gateway + video-bridge.
# Запуск из корня репозитория:  .\scripts\build_all.ps1
# Пути по умолчанию совпадают с scripts/run_backend_gateway.ps1 — при необходимости передайте параметры.

param(
    [string] $RepoRoot = "",
    [string] $NativeBuildDir = "native\build-msvc-trt-user",
    [string] $OpenCvLibDir = "C:\build\opencv\x64\vc17\lib",
    [string] $OpenCvInclude = "C:\build\opencv\include",
    [string] $TensorRtRoot = "C:\TensorRT-10.16.1.11",
    [string] $CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1",
    [ValidateSet("Release", "RelWithDebInfo")]
    [string] $Configuration = "Release"
)

$ErrorActionPreference = "Stop"
if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$RepoRoot = (Resolve-Path $RepoRoot).Path

$nativeBuild = Join-Path $RepoRoot $NativeBuildDir
if (-not (Test-Path (Join-Path $nativeBuild "CMakeCache.txt"))) {
    Write-Error "No CMakeCache in $nativeBuild. Configure native first (native/README.md or native/scripts/build_engine_msvc.ps1)."
}

$env:PATH = "$(Join-Path $TensorRtRoot 'bin');$(Join-Path $CudaRoot 'bin');$env:PATH"

Write-Host "=== CMake: integra_ffi ===" -ForegroundColor Cyan
Push-Location $RepoRoot
try {
    cmake --build $nativeBuild --config $Configuration --target integra_ffi
}
finally {
    Pop-Location
}

$ffiSrc = Join-Path (Join-Path $nativeBuild $Configuration) "integra_ffi.dll"
if (-not (Test-Path $ffiSrc)) {
    Write-Error "Missing $ffiSrc"
}
$gwOut = Join-Path $RepoRoot "runtime-core\target\release"
New-Item -ItemType Directory -Force -Path $gwOut | Out-Null
Copy-Item -Force $ffiSrc (Join-Path $gwOut "integra_ffi.dll")
Write-Host "Copied: $(Join-Path $gwOut 'integra_ffi.dll')" -ForegroundColor Green

Write-Host "=== cargo: backend_gateway ===" -ForegroundColor Cyan
Push-Location (Join-Path $RepoRoot "runtime-core")
try {
    cargo build --release --bin backend_gateway
}
finally {
    Pop-Location
}

$world = Get-ChildItem -Path $OpenCvLibDir -Filter "opencv_world*.lib" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $world) {
    Write-Error "No opencv_world*.lib in $OpenCvLibDir"
}
$linkLib = $world.BaseName
Write-Host "=== cargo: video-bridge (OpenCV link: $linkLib) ===" -ForegroundColor Cyan
$env:OPENCV_LINK_LIBS = $linkLib
$env:OPENCV_LINK_PATHS = $OpenCvLibDir
$env:OPENCV_INCLUDE_PATHS = $OpenCvInclude
$env:OPENCV_DISABLE_PROBES = "cmake,pkg_config,vcpkg_cmake,vcpkg"

Push-Location (Join-Path $RepoRoot "video-bridge")
try {
    cargo build --release
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Done. Run:" -ForegroundColor Green
Write-Host "  .\scripts\run_backend_gateway.ps1"
Write-Host "  .\video-bridge\target\release\video-bridge.exe   (separate terminal if needed)"
Write-Host ""
