#!/usr/bin/env bash
set -euo pipefail
CODE_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${CODE_AGENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
python -m training.stage_b_policy "$@"
