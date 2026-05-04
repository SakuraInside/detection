param(
    [ValidateSet("legacy", "hybrid", "external")]
    [string]$Profile = "external",
    [string]$Host = $(if ($env:APP_HOST) { $env:APP_HOST } else { "127.0.0.1" }),
    [int]$Port = $(if ($env:APP_PORT) { [int]$env:APP_PORT } else { 8000 }),
    [switch]$Reload,
    [switch]$NoApplyProfile,
    [switch]$LowMemory
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$args = @("run_stack.py", "--profile", $Profile, "--host", $Host, "--port", "$Port")
if ($Reload) { $args += "--reload" }
if ($NoApplyProfile) { $args += "--no-apply-profile" }
if ($LowMemory) { $args += "--low-memory" }

Write-Host "[run_stack.ps1] profile=$Profile host=$Host port=$Port"
python @args
