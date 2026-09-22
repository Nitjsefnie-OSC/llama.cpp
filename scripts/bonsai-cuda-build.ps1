param(
    [Parameter(Mandatory = $true)][string]$CudaRoot,
    [string]$BuildDir = 'build-bonsai-cuda',
    [int]$Jobs = 4,
    [string[]]$Targets = @('llama-bench', 'llama-server', 'test-backend-ops', 'llama-perplexity')
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$CudaRoot = (Resolve-Path -LiteralPath $CudaRoot).Path
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -ne 0 -or -not $vs) { throw 'MSVC x64 build tools not found' }
& "$vs\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
$ninja = "$vs\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$env:CUDA_PATH = $CudaRoot
$env:PATH = "$CudaRoot\bin;$env:PATH"

cmake -S $repo -B $BuildDir -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" `
    '-DCMAKE_BUILD_TYPE=Release' '-DCMAKE_CUDA_ARCHITECTURES=86-real' `
    "-DCUDAToolkit_ROOT=$CudaRoot" "-DCMAKE_CUDA_COMPILER=$CudaRoot\bin\nvcc.exe" `
    '-DGGML_CUDA=ON' '-DGGML_CUDA_FA_ALL_QUANTS=OFF' '-DGGML_CUDA_NCCL=OFF' `
    '-DGGML_NATIVE=ON' '-DLLAMA_BUILD_TESTS=ON' '-DLLAMA_BUILD_EXAMPLES=OFF' `
    '-DLLAMA_BUILD_APP=OFF' '-DLLAMA_BUILD_UI=OFF' '-DLLAMA_BUILD_HTML=OFF' `
    '-DGGML_RPC=OFF' '-DGGML_CUDA_COMPRESSION_MODE=speed'
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed: $LASTEXITCODE" }
cmake --build $BuildDir --parallel $Jobs --target @Targets
if ($LASTEXITCODE -ne 0) { throw "CMake build failed: $LASTEXITCODE" }
