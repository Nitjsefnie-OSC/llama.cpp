param(
    [Parameter(Mandatory = $true)][string]$BinaryDir,
    [string]$Root,
    [int]$Context = 188416,
    [int]$Port = 8090,
    [string]$CudaTimingLog
)

$ErrorActionPreference = 'Stop'
if (-not $Root) { $Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent }
$BinaryDir = (Resolve-Path -LiteralPath $BinaryDir).Path
$Root = (Resolve-Path -LiteralPath $Root).Path
$env:PATH = "$Root\tools\cuda-12.9.1\bin;$env:PATH"
$serverArgs = @(
    '--slot-save-path', "$Root\slots",
    '-m', "$Root\models\Ternary-Bonsai-2-27B-PQ2_0.gguf",
    '-ngl', '99', '-ctk', 'q4_0', '-ctv', 'q4_0',
    '-b', '512', '-ub', '512', '-c', $Context,
    '--reasoning-format', 'deepseek', '--host', '0.0.0.0', '--port', $Port,
    '--alias', 'bonsai-2-27b', '--chat-template-file', "$Root\bonsai-chat-template.jinja"
)
if ($CudaTimingLog) {
    $CudaTimingLog = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($CudaTimingLog)
    # Reserve a new file before llama's logger opens it in write mode.
    $logReservation = [System.IO.File]::Open($CudaTimingLog, [System.IO.FileMode]::CreateNew)
    $logReservation.Dispose()
    # Backend INFO records use the common logger's TRACE threshold.
    $serverArgs += @('--log-file', $CudaTimingLog, '--log-verbosity', '4')
}
$stableBinaryDir = Join-Path $Root 'tools\llamacpp-prism'
if (-not [string]::Equals($BinaryDir.TrimEnd('\', '/'), $stableBinaryDir.TrimEnd('\', '/'), [StringComparison]::OrdinalIgnoreCase)) {
    & "$PSScriptRoot\bonsai-server-deploy.ps1" -BinaryDir $BinaryDir -Root $Root
    if (-not $?) { throw 'Server deployment failed' }
    $BinaryDir = $stableBinaryDir
}
$previousCudaTiming = [Environment]::GetEnvironmentVariable('DEBUG_CUDA_TIMING', 'Process')
try {
    if ($CudaTimingLog) { $env:DEBUG_CUDA_TIMING = '1' }
    & "$BinaryDir\llama-server.exe" @serverArgs
    $serverExitCode = $LASTEXITCODE
} finally {
    if ($CudaTimingLog) {
        if ($null -eq $previousCudaTiming) {
            Remove-Item Env:DEBUG_CUDA_TIMING -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable('DEBUG_CUDA_TIMING', $previousCudaTiming, 'Process')
        }
    }
}
exit $serverExitCode
