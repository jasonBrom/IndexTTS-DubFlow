$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$SetupScript = Join-Path $ProjectRoot "scripts\setup_runtime.py"

$Python = Get-Command py -ErrorAction SilentlyContinue
if ($Python) {
    & $Python.Source -3.11 $SetupScript @args
    exit $LASTEXITCODE
}

$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) {
    throw "找不到 Python。请安装 Python 3.11 x64，并在安装器中勾选 Add Python to PATH。"
}

& $Python.Source $SetupScript @args
exit $LASTEXITCODE
