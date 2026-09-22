[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BinaryDir,
    [string]$Root
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $Root) { $Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot) }

function Assert-PlainPath([string]$Path) {
    $cursor = [IO.Path]::GetFullPath($Path)
    while ($cursor) {
        if (Test-Path -LiteralPath $cursor) {
            $item = Get-Item -LiteralPath $cursor -Force
            if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Reparse points are not supported: $cursor"
            }
        }
        $cursor = Split-Path -Parent $cursor
    }
}

function Assert-ToolsPath([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($tools + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Deployment path is outside workspace tools: $full"
    }
    Assert-PlainPath $full
}

function Assert-DestinationStopped {
    $processes = @(Get-CimInstance Win32_Process)
    foreach ($process in $processes) {
        if ($process.ExecutablePath) {
            if ($process.ExecutablePath.StartsWith($destination + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Destination is in use by PID $($process.ProcessId): $($process.ExecutablePath)"
            }
        } elseif ($process.Name -in $executableNames) {
            throw "Cannot inspect executable path for PID $($process.ProcessId) ($($process.Name)); deployment refused."
        }
    }
}

$Root = (Resolve-Path -LiteralPath $Root).ProviderPath
$source = (Resolve-Path -LiteralPath $BinaryDir).ProviderPath
if (-not (Test-Path -LiteralPath $Root -PathType Container) -or -not (Test-Path -LiteralPath $source -PathType Container)) {
    throw 'Root and BinaryDir must be existing directories.'
}
Assert-PlainPath $Root
Assert-PlainPath $source
$tools = [IO.Path]::GetFullPath((Join-Path $Root 'tools')).TrimEnd([IO.Path]::DirectorySeparatorChar)
$destination = Join-Path $tools 'llamacpp-prism'
if ($source.Equals($destination, [StringComparison]::OrdinalIgnoreCase) -or
    $source.StartsWith($destination + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'BinaryDir must not be the destination or a directory inside it.'
}
$required = @('llama-server.exe', 'llama-server-impl.dll', 'llama.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu.dll', 'ggml-cuda.dll')
foreach ($name in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $source $name) -PathType Leaf)) { throw "Missing required runtime file: $name" }
}
$files = @(Get-ChildItem -LiteralPath $source -File | Where-Object { $_.Extension -in '.exe', '.dll' } | Sort-Object Name)
foreach ($file in $files) { Assert-PlainPath $file.FullName }
$executableNames = @($files | Where-Object Extension -EQ '.exe' | ForEach-Object Name)
if (Test-Path -LiteralPath $destination -PathType Container) {
    $executableNames += @(Get-ChildItem -LiteralPath $destination -File -Filter '*.exe' | ForEach-Object Name)
}
$id = (Get-Date -Format 'yyyyMMddTHHmmssfff') + '-' + [guid]::NewGuid().ToString('N')
$stage = Join-Path $tools ('.llamacpp-deploy-stage-' + $id)
$backupRoot = Join-Path $tools 'llamacpp-deploy-backups'
$backup = Join-Path $backupRoot $id
$logDir = Join-Path $Root 'logs'
$receiptPath = Join-Path $logDir ('bonsai-deploy-' + $id + '.json')
foreach ($path in @($destination, $stage, $backupRoot, $backup)) { Assert-ToolsPath $path }
Assert-PlainPath $logDir
Assert-DestinationStopped
foreach ($path in @($stage, $backup, $receiptPath)) {
    if (Test-Path -LiteralPath $path) { throw "Refusing to overwrite existing deployment artifact: $path" }
}
if ((Test-Path -LiteralPath $destination) -and -not (Test-Path -LiteralPath $destination -PathType Container)) {
    throw "Destination is not a directory: $destination"
}

$receipt = [ordered]@{
    status = 'staging'
    startedUtc = [DateTime]::UtcNow.ToString('o')
    completedUtc = $null
    source = $source
    destination = $destination
    staging = $stage
    backup = $null
    files = @()
    error = $null
}
$oldMoved = $false
try {
    New-Item -ItemType Directory -Path $stage | Out-Null
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    foreach ($file in $files) {
        $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
        $target = Join-Path $stage $file.Name
        Copy-Item -LiteralPath $file.FullName -Destination $target
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $hash) {
            throw "Staged hash differs from source: $($file.Name)"
        }
        $receipt.files += [ordered]@{ name = $file.Name; sha256 = $hash; bytes = $file.Length }
    }
    $receipt.status = 'staged'
    $json = $receipt | ConvertTo-Json -Depth 6
    $stream = [IO.File]::Open($receiptPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    try {
        $bytes = [Text.Encoding]::UTF8.GetBytes($json)
        $stream.Write($bytes, 0, $bytes.Length)
    } finally { $stream.Dispose() }
    Assert-DestinationStopped
    foreach ($path in @($destination, $stage, $backupRoot, $backup)) { Assert-ToolsPath $path }
    if (Test-Path -LiteralPath $destination) {
        New-Item -ItemType Directory -Path $backupRoot -Force | Out-Null
        Assert-ToolsPath $backup
        [IO.Directory]::Move($destination, $backup)
        $oldMoved = $true
        $receipt.backup = $backup
    }
    try {
        [IO.Directory]::Move($stage, $destination)
    } catch {
        if ($oldMoved) {
            foreach ($path in @($backup, $destination)) { Assert-ToolsPath $path }
            [IO.Directory]::Move($backup, $destination)
            $receipt.status = 'rolled-back'
            $receipt.backup = $null
        }
        throw
    }
    $receipt.status = 'deployed'
} catch {
    if ($receipt.status -ne 'rolled-back') { $receipt.status = 'failed' }
    $receipt.error = $_.Exception.Message
    throw
} finally {
    $receipt.completedUtc = [DateTime]::UtcNow.ToString('o')
    if (Test-Path -LiteralPath $receiptPath -PathType Leaf) {
        [IO.File]::WriteAllText($receiptPath, ($receipt | ConvertTo-Json -Depth 6), [Text.Encoding]::UTF8)
        Write-Output $receiptPath
    }
}
