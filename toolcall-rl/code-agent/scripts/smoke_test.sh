#!/usr/bin/env bash
set -euo pipefail
CODE_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${CODE_AGENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
python -m pytest -q "${CODE_AGENT_DIR}/tests"
