# Сборка integra_ffi (ONNX Runtime, CPU) pod MSVC.
# Zapusk iz kornya repo:  .\native\scripts\build_onnx_msvc.ps1

param(
    [string] $OpenCvDir    = "",
    [string] $OrtRoot      = "",
    [string] $RepoRoot     = "",
    [switch] $OrtCuda,
    [ValidateSet("Release", "RelWithDebInfo")]
    [string] $Configuration = "Release"
)

$ErrorActionPreference = "Stop"

if (-not $OpenCvDir) { $OpenCvDir = $env:INTEGRA_OPENCV_DIR }
if (-not $OrtRoot)   { $OrtRoot   = $env:ONNXRUNTIME_ROOT   }
if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

if (-not $OpenCvDir -or -not (Test-Path (Join-Path $OpenCvDir "OpenCVConfig.cmake"))) {
    Write-Error "Zadajte OpenCV: -OpenCvDir <put> ili env:INTEGRA_OPENCV_DIR (nuzhen OpenCVConfig.cmake)"
}
if (-not $OrtRoot -or -not (Test-Path (Join-Path $OrtRoot "include\onnxruntime_cxx_api.h"))) {
    Write-Error "Zadajte ONNX Runtime: -OrtRoot <put> ili env:ONNXRUNTIME_ROOT (nuzhen include\onnxruntime_cxx_api.h)"
}

$native = Join-Path $RepoRoot "native"
$build  = Join-Path $native "build-msvc-onnx"

$ortBin = Join-Path $OrtRoot "bin"
if (Test-Path $ortBin) { $env:PATH = "$ortBin;$env:PATH" }

$enableCuda = "OFF"
if ($OrtCuda) { $enableCuda = "ON" }

$cmakeArgs = @(
    "-B", $build,
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DOpenCV_DIR=$OpenCvDir",
    "-DINTEGRA_ENABLE_CUDA=$enableCuda",
    "-DINTEGRA_WITH_ONNXRUNTIME=ON",
    "-DINTEGRA_WITH_TENSORRT=OFF",
    "-DINTEGRA_ONNXRUNTIME_ROOT=$OrtRoot"
)

if ($OrtCuda) {
    $cmakeArgs += "-DINTEGRA_ORT_CUDA=ON"
}

Push-Location $native
try {
    Write-Host "=== cmake configure ==="
    cmake @cmakeArgs

    Write-Host "=== cmake build ==="
    cmake --build $build --config $Configuration --target integra_ffi

    $dll = Join-Path (Join-Path $build $Configuration) "integra_ffi.dll"
    if (-not (Test-Path $dll)) {
        Write-Error "integra_ffi.dll ne najdena: $dll"
    }

    Write-Host "=== Gotovo ==="
    Write-Host "DLL: $dll"
    Write-Host "Skopiruyte integra_ffi.dll i onnxruntime.dll ryadom s backend_gateway.exe"
    Write-Host "onnxruntime.dll nakhoditsya v: $ortBin"
}
finally {
    Pop-Location
}
