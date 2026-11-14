param([int]$GpuLayers = 99, [int]$Context = 8192, [int]$Port = 8080)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $projectRoot '.tools/llama-runtime.json'
if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'Run python scripts/setup_local_llm.py --install first.' }
$runtime = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if (-not (Test-Path -LiteralPath $runtime.server)) { throw 'Runtime executable is missing.' }
if (-not (Test-Path -LiteralPath $runtime.model)) { throw 'Model file is missing.' }
# Foreground process so the teammate can see and stop their own server.
# Thinking is disabled by the client's chat_template_kwargs on every request.
& $runtime.server --model $runtime.model --alias avito-qwen3-8b --host 127.0.0.1 --port $Port --ctx-size $Context --parallel 1 --n-gpu-layers $GpuLayers --jinja
exit $LASTEXITCODE
