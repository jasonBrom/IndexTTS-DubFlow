$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$RuntimeRoot = if ($env:DUBBER_RUNTIME_ROOT) { $env:DUBBER_RUNTIME_ROOT } else { Join-Path $ProjectRoot ".runtime" }
$IndexDir = if ($env:INDEXTTS_DIR) { $env:INDEXTTS_DIR } else { Join-Path $RuntimeRoot "index-tts" }
$ModelDir = if ($env:INDEXTTS_MODEL_DIR) { $env:INDEXTTS_MODEL_DIR } else { Join-Path $IndexDir "checkpoints" }
$Python = Join-Path $IndexDir ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "运行环境不存在：$Python`n请先执行 .\scripts\bootstrap.ps1。"
}

$env:DUBBER_RUNTIME_ROOT = $RuntimeRoot
$env:INDEXTTS_DIR = $IndexDir
$env:INDEXTTS_MODEL_DIR = $ModelDir
if (-not $env:INDEXTTS_CONFIG) { $env:INDEXTTS_CONFIG = Join-Path $ModelDir "config.yaml" }
if (-not $env:ASR_RUNTIME_ROOT) { $env:ASR_RUNTIME_ROOT = Join-Path $RuntimeRoot "asr-runtimes" }
if (-not $env:TRANSLATION_RUNTIME_ROOT) { $env:TRANSLATION_RUNTIME_ROOT = Join-Path $RuntimeRoot "translation-runtimes" }
if (-not $env:HF_HOME) { $env:HF_HOME = Join-Path $RuntimeRoot "cache\huggingface" }
if (-not $env:HYMT_HF_HOME) { $env:HYMT_HF_HOME = $env:HF_HOME }
if (-not $env:HF_HUB_DISABLE_XET) { $env:HF_HUB_DISABLE_XET = "1" }
$env:PYTHONUNBUFFERED = "1"

& $Python -u (Join-Path $ProjectRoot "app.py") @args
exit $LASTEXITCODE
