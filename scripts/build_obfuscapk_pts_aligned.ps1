param(
    [string]$ProjectRoot = "D:/code/detection_fusion_copy",
    [string]$ConfigPath = "config/extract_obfuscapk_aligned.yaml",
    [string]$ApkRoot = "D:/obf_final_out",
    [string]$PtRoot = "D:/pts_obfuscapk",
    [string]$PythonExe = "D:/IDE/miniconda/envs/malware/python.exe",
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"

$project = Resolve-Path -LiteralPath $ProjectRoot
Set-Location -LiteralPath $project

$config = Join-Path $project $ConfigPath
$vocab = Join-Path $project "config/manifest_vocab.yaml"

if (-not (Test-Path -LiteralPath $config)) {
    throw "Missing extraction config: $config"
}
if (-not (Test-Path -LiteralPath $vocab)) {
    throw "Missing formal train manifest vocab: $vocab"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Missing Python executable: $PythonExe"
}

Write-Host "Using Python:" -ForegroundColor Cyan
Write-Host "  $PythonExe"
& $PythonExe -c "import sys, yaml, torch, numpy; print('  ' + sys.executable); print('  deps: yaml/torch/numpy ok')"

$requiredDirs = @(
    "clean",
    "nop",
    "goto",
    "rename",
    "string",
    "mixed_light",
    "advanced_reflection",
    "call_indirection",
    "mixed_api_graph_manifest"
)

Write-Host "Input APK directories:" -ForegroundColor Cyan
foreach ($name in $requiredDirs) {
    $dir = Join-Path $ApkRoot $name
    if (-not (Test-Path -LiteralPath $dir)) {
        throw "Missing APK directory: $dir"
    }
    $count = (Get-ChildItem -LiteralPath $dir -File -Filter "*.apk").Count
    Write-Host ("  {0,-12} {1,6} APKs  {2}" -f $name, $count, $dir)
}

Write-Host ""
Write-Host "Using manifest vocab:" -ForegroundColor Cyan
Write-Host "  $vocab"
Write-Host ""

if ((Test-Path -LiteralPath $PtRoot) -and -not $KeepExisting) {
    Write-Host "Removing existing PT root with mismatched/old schema:" -ForegroundColor Yellow
    Write-Host "  $PtRoot"
    Remove-Item -LiteralPath $PtRoot -Recurse -Force
}

Write-Host ""
Write-Host "Building aligned Obfuscapk PTs..." -ForegroundColor Cyan
& $PythonExe scripts/build_tri_modal_pts_direct.py --config $config

Write-Host ""
Write-Host "Output PT counts:" -ForegroundColor Cyan
$outDirs = @{
    clean = "clean"
    nop = "nop"
    goto = "goto"
    method_rename = "method_rename"
    string = "string"
    combined = "combined"
    advanced_reflection = "advanced_reflection"
    call_indirection = "call_indirection"
    mixed_api_graph_manifest = "mixed_api_graph_manifest"
}
foreach ($split in $outDirs.Keys) {
    $dir = Join-Path $PtRoot $outDirs[$split]
    $count = 0
    if (Test-Path -LiteralPath $dir) {
        $count = (Get-ChildItem -LiteralPath $dir -File -Filter "*.pt").Count
    }
    Write-Host ("  {0,-14} {1,6} PTs  {2}" -f $split, $count, $dir)
}

Write-Host ""
Write-Host "Done. Next sanity check one PT should show api_ids=(2048,), manifest dims 128/64/32, and call_x feature dim 519." -ForegroundColor Green
