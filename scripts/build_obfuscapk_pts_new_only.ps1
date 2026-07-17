param(
    [string]$ProjectRoot = "D:/code/detection_fusion_copy",
    [string]$ConfigPath = "config/extract_obfuscapk_new_only.yaml",
    [string]$ApkRoot = "D:/obf_final_out",
    [string]$PtRoot = "D:/pts_obfuscapk",
    [string]$PythonExe = "D:/IDE/miniconda/envs/malware/python.exe",
    [switch]$OverwriteNew
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
    Write-Host ("  {0,-26} {1,6} APKs  {2}" -f $name, $count, $dir)
}

Write-Host ""
Write-Host "Using manifest vocab:" -ForegroundColor Cyan
Write-Host "  $vocab"
Write-Host ""

if ($OverwriteNew) {
    foreach ($name in $requiredDirs) {
        $dir = Join-Path $PtRoot $name
        if (Test-Path -LiteralPath $dir) {
            Write-Host "Removing existing new PT dir: $dir" -ForegroundColor Yellow
            Remove-Item -LiteralPath $dir -Recurse -Force
        }
    }
}

Write-Host ""
Write-Host "Building newly added Obfuscapk PTs only..." -ForegroundColor Cyan
& $PythonExe scripts/build_tri_modal_pts_direct.py --config $config

Write-Host ""
Write-Host "Output PT counts:" -ForegroundColor Cyan
foreach ($split in $requiredDirs) {
    $dir = Join-Path $PtRoot $split
    $count = 0
    if (Test-Path -LiteralPath $dir) {
        $count = (Get-ChildItem -LiteralPath $dir -File -Filter "*.pt").Count
    }
    Write-Host ("  {0,-26} {1,6} PTs  {2}" -f $split, $count, $dir)
}

Write-Host ""
Write-Host "Done. Existing nop/goto/method_rename/string/combined PT dirs were not removed." -ForegroundColor Green
