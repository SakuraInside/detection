# Сборка integra_trt_bake + integra_ffi (TensorRT) и bake models/yolo11n_fp16.engine под MSVC.
# Запуск из PowerShell:  .\native\scripts\build_engine_msvc.ps1
# Перед первым запуском задайте пути ниже (или передайте параметрами).

param(
    [string] $OpenCvDir = "",
    [string] $TensorRtRoot = "",
    [string] $RepoRoot = "",
    [ValidateSet("Release", "RelWithDebInfo")]
    [string] $Configuration = "Release"
)

$ErrorActionPreference = "Stop"

if (-not $OpenCvDir) { $OpenCvDir = $env:INTEGRA_OPENCV_DIR }
if (-not $TensorRtRoot) { $TensorRtRoot = $env:TENSORRT_ROOT }
if (-not $RepoRoot) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}

if (-not $OpenCvDir -or -not (Test-Path (Join-Path $OpenCvDir "OpenCVConfig.cmake"))) {
    Write-Error "Задайте каталог OpenCV: -OpenCvDir или `$env:INTEGRA_OPENCV_DIR (нужен OpenCVConfig.cmake)."
}
if (-not $TensorRtRoot -or -not (Test-Path (Join-Path $TensorRtRoot "include\NvInfer.h"))) {
    Write-Error "Задайте TensorRT SDK: -TensorRtRoot или `$env:TENSORRT_ROOT (нужен include\NvInfer.h)."
}

$native = Join-Path $RepoRoot "native"
$build = Join-Path $native "build-msvc-trt"
$trtLib = Join-Path $TensorRtRoot "lib"
$trtBin = Join-Path $TensorRtRoot "bin"
if (Test-Path $trtBin) {
    $env:PATH = "$trtBin;$env:PATH"
}
if (Test-Path $trtLib) {
    $env:PATH = "$trtLib;$env:PATH"
}

Push-Location $native
try {
    cmake -B $build -G "Visual Studio 17 2022" -A x64 `
        "-DOpenCV_DIR=$OpenCvDir" `
        -DINTEGRA_ENABLE_CUDA=ON `
        -DINTEGRA_WITH_TENSORRT=ON `
        "-DTENSORRT_ROOT=$TensorRtRoot"

    cmake --build $build --config $Configuration --target integra_trt_bake --target integra_ffi

    $bake = Join-Path $build $Configuration "integra_trt_bake.exe"
    if (-not (Test-Path $bake)) {
        Write-Error "Не найден $bake после сборки."
    }

    $onnx = Join-Path $RepoRoot "models\yolo11n.onnx"
    $engine = Join-Path $RepoRoot "models\yolo11n_fp16.engine"
    if (-not (Test-Path $onnx)) {
        Write-Error "Нет ONNX: $onnx"
    }

    & $bake --onnx $onnx --out $engine --fp16 --workspace-mb 4096
    Write-Host "Готово: $engine"
    Write-Host "Скопируйте integra_ffi.dll из $build\$Configuration\ рядом с backend_gateway.exe (или в PATH)."
}
finally {
    Pop-Location
}
