#!/usr/bin/env bash
# Read-only, fail-closed preflight for the isolated Code Stage-C run.
set -euo pipefail

: "${CODE_TRAIN_MANIFEST:?set public 100-row runtime manifest}"
: "${CODE_EVALUATOR_MANIFEST:?set private evaluator manifest}"
: "${CODE_OUTPUT_DIR:?set dedicated Code output directory}"
: "${CODE_ENV_SERVER_URL:?set dedicated Code pool URL}"
: "${CODE_HF_CHECKPOINT:?set base model checkpoint}"

case "${CODE_OUTPUT_DIR}" in
  *OpenClaw-RL*|*outputs/doc*|"" ) echo "refusing non-Code output path" >&2; exit 2 ;;
esac
case "${CODE_ENV_SERVER_URL}" in
  *:18091|*:18092|*:18093 ) ;;
  * ) echo "CODE_ENV_SERVER_URL must use a dedicated Code port" >&2; exit 2 ;;
esac

python - <<'PY'
import json, os
from pathlib import Path

train = Path(os.environ['CODE_TRAIN_MANIFEST'])
evaluator = Path(os.environ['CODE_EVALUATOR_MANIFEST'])
for path in (train, evaluator):
    if not path.is_file():
        raise SystemExit(f'missing manifest: {path}')
public_rows = [json.loads(line) for line in train.read_text(encoding='utf-8').splitlines() if line.strip()]
private_rows = [json.loads(line) for line in evaluator.read_text(encoding='utf-8').splitlines() if line.strip()]
if len(public_rows) != 100:
    raise SystemExit(f'expected exactly 100 public train rows, got {len(public_rows)}')
if len(private_rows) != 100:
    raise SystemExit(f'expected exactly 100 private evaluator rows, got {len(private_rows)}')
private_by_id = {str(row.get('metadata', {}).get('instance_id') or row.get('metadata', {}).get('public_instance', {}).get('instance_id') or ''): row for row in private_rows}
for row in public_rows:
    metadata = row.get('metadata') or {}
    if 'evaluator_private' in metadata:
        raise SystemExit('public runtime manifest contains evaluator_private')
    public = metadata.get('public_instance') or {}
    instance_id = str(public.get('instance_id') or '')
    if not instance_id or instance_id not in private_by_id:
        raise SystemExit(f'public instance lacks matching private evaluator: {instance_id!r}')
    serialized = json.dumps(row, ensure_ascii=False).lower()
    if any(token in serialized for token in ('test_patch', 'fail_to_pass', 'pass_to_pass', 'gold_patch', 'eval_script')):
        raise SystemExit(f'private evaluator content leaked into public runtime row {instance_id!r}')
print('manifest_preflight=ok rows=100')
PY

curl --fail --silent --show-error --max-time 10 "${CODE_ENV_SERVER_URL%/}/healthz" >/dev/null
echo "pool_health=ok"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is unavailable on this Code host" >&2
  exit 2
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
fi
echo "code_stage_c_preflight=ok"
