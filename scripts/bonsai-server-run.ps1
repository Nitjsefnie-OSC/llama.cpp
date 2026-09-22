param(
    [Parameter(Mandatory = $true)][string]$BinaryDir,
    [string]$Root,
    [int]$Context = 188416,
    [int]$Port = 8090,
    [string]$CudaTimingLog,
    [string]$CudaGraphStatsLog,
    [ValidateRange(1, 512)][int]$UBatch = 512
)

$ErrorActionPreference = 'Stop'
if ($CudaTimingLog -and $CudaGraphStatsLog) {
    throw 'CudaTimingLog and CudaGraphStatsLog are mutually exclusive'
}
if (-not $Root) { $Root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent }
$BinaryDir = (Resolve-Path -LiteralPath $BinaryDir).Path
$Root = (Resolve-Path -LiteralPath $Root).Path
$env:PATH = "$Root\tools\cuda-12.9.1\bin;$env:PATH"
$serverArgs = @(
    '--slot-save-path', "$Root\slots",
    '-m', "$Root\models\Ternary-Bonsai-2-27B-PQ2_0.gguf",
    '-ngl', '99', '-ctk', 'q4_0', '-ctv', 'q4_0',
    '-b', '512', '-ub', $UBatch, '-c', $Context,
    '--reasoning-format', 'deepseek', '--host', '0.0.0.0', '--port', $Port,
    '--alias', 'bonsai-2-27b', '--chat-template-file', "$Root\bonsai-chat-template.jinja"
)
$cudaLog = if ($CudaTimingLog) { $CudaTimingLog } else { $CudaGraphStatsLog }
if ($cudaLog) {
    $cudaLog = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($cudaLog)
    # Reserve a new file before llama's logger opens it in write mode.
    $logReservation = [System.IO.File]::Open($cudaLog, [System.IO.FileMode]::CreateNew)
    $logReservation.Dispose()
    # Backend INFO records use the common logger's TRACE threshold.
    $serverArgs += @('--log-file', $cudaLog, '--log-verbosity', '4')
}
$stableBinaryDir = Join-Path $Root 'tools\llamacpp-prism'
if (-not [string]::Equals($BinaryDir.TrimEnd('\', '/'), $stableBinaryDir.TrimEnd('\', '/'), [StringComparison]::OrdinalIgnoreCase)) {
    & "$PSScriptRoot\bonsai-server-deploy.ps1" -BinaryDir $BinaryDir -Root $Root
    if (-not $?) { throw 'Server deployment failed' }
    $BinaryDir = $stableBinaryDir
}
$previousCudaTiming = [Environment]::GetEnvironmentVariable('DEBUG_CUDA_TIMING', 'Process')
$previousCudaGraphStats = [Environment]::GetEnvironmentVariable('DEBUG_CUDA_GRAPH_STATS', 'Process')
try {
    if ($CudaTimingLog) { $env:DEBUG_CUDA_TIMING = '1' }
    if ($CudaGraphStatsLog) {
        # Timing instrumentation disables replay: clear it only for this invocation.
        Remove-Item Env:DEBUG_CUDA_TIMING -ErrorAction SilentlyContinue
        $env:DEBUG_CUDA_GRAPH_STATS = '1'
    }
    & "$BinaryDir\llama-server.exe" @serverArgs
    $serverExitCode = $LASTEXITCODE
} finally {
    if ($CudaTimingLog -or $CudaGraphStatsLog) {
        if ($null -eq $previousCudaTiming) {
            Remove-Item Env:DEBUG_CUDA_TIMING -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable('DEBUG_CUDA_TIMING', $previousCudaTiming, 'Process')
        }
    }
    if ($CudaGraphStatsLog) {
        if ($null -eq $previousCudaGraphStats) {
            Remove-Item Env:DEBUG_CUDA_GRAPH_STATS -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable('DEBUG_CUDA_GRAPH_STATS', $previousCudaGraphStats, 'Process')
        }
    }
}
exit $serverExitCode
