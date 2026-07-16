param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$key = 'HKCU:\Software\Microsoft\Command Processor'
$outputDir = Join-Path $PSScriptRoot 'outputs'
$stdout = Join-Path $outputDir 'full_tool_audit.stdout.log'
$stderr = Join-Path $outputDir 'full_tool_audit.stderr.log'
$statusPath = Join-Path $outputDir 'full_tool_audit_runner_status.json'
$python = Join-Path $PSScriptRoot 'runtime\python\python.exe'
$audit = Join-Path $root 'tests\audit_document_tools.py'

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$original = (Get-ItemProperty -Path $key -Name AutoRun -ErrorAction Stop).AutoRun
$status = [ordered]@{ started_at = (Get-Date).ToString('o'); status = 'running'; exit_code = $null; error = $null }

try {
    Remove-ItemProperty -Path $key -Name AutoRun -ErrorAction Stop
    $process = Start-Process -FilePath $python -ArgumentList @('-S', $audit) -WorkingDirectory $root -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
    $process.WaitForExit()
    $status.status = if ($process.ExitCode -eq 0) { 'ok' } else { 'error' }
    $status.exit_code = $process.ExitCode
} catch {
    $status.status = 'error'
    $status.error = $_.Exception.Message
} finally {
    Set-ItemProperty -Path $key -Name AutoRun -Value $original
    $status.finished_at = (Get-Date).ToString('o')
    $status | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

if ($status.status -ne 'ok') { exit 1 }
