param(
    [Parameter(Mandatory = $true)][string]$BinaryDir,
    [string]$Root,
    [int]$Context = 188416,
    [int]$Port = 8090
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
& "$BinaryDir\llama-server.exe" @serverArgs
exit $LASTEXITCODE
