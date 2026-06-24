param(
    [string]$ModelName = "UVR-MDX-NET-Voc_FT.onnx",
    [string]$ModelDir = "models/audio-separator",
    [string]$Url = "https://gh.llkk.cc/https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx",
    [Int64]$ExpectedSize = 66762490,
    [string]$ExpectedMd5 = "d21dc03e4b9ef397b47231f483af6db8"
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$targetDir = if ([System.IO.Path]::IsPathRooted($ModelDir)) {
    [System.IO.Path]::GetFullPath($ModelDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $root $ModelDir))
}
$targetPath = Join-Path $targetDir $ModelName
$partialPath = "$targetPath.part"

New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

function Test-ModelFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $item = Get-Item -LiteralPath $Path
    if ($ExpectedSize -gt 0 -and $item.Length -ne $ExpectedSize) {
        Write-Host "Size mismatch: $($item.Length), expected $ExpectedSize"
        return $false
    }
    if ($ExpectedMd5) {
        $hash = (Get-FileHash -LiteralPath $Path -Algorithm MD5).Hash.ToLowerInvariant()
        if ($hash -ne $ExpectedMd5.ToLowerInvariant()) {
            Write-Host "MD5 mismatch: $hash, expected $ExpectedMd5"
            return $false
        }
    }
    return $true
}

if (Test-ModelFile -Path $targetPath) {
    Write-Host "Model already ready: $targetPath"
    exit 0
}

if (Test-Path -LiteralPath $partialPath) {
    Remove-Item -LiteralPath $partialPath -Force
}

Write-Host "Downloading $ModelName"
Write-Host "URL: $Url"
Write-Host "Target: $targetPath"

curl.exe -L --fail --connect-timeout 20 --speed-time 60 --speed-limit 10240 -o $partialPath $Url

if (-not (Test-ModelFile -Path $partialPath)) {
    Remove-Item -LiteralPath $partialPath -Force -ErrorAction SilentlyContinue
    throw "Downloaded model failed validation."
}

Move-Item -LiteralPath $partialPath -Destination $targetPath -Force
Write-Host "Model ready: $targetPath"
