param(
    [int]$Port = 8765
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONUTF8 = '1'
$runtimePython = '.\tool_studio\runtime\python\python.exe'
$python = if (Test-Path -LiteralPath $runtimePython) { $runtimePython } else { 'python' }
& $python .\tool_studio\app.py --host 127.0.0.1 --port $Port
