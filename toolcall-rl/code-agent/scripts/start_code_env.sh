#!/usr/bin/env bash
set -euo pipefail

# Code-only server bootstrap.  It is intentionally separate from all Doc
# launchers and does not source Doc environment variables.
CODE_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${CODE_AGENT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
python -m env.server.pool_server "$@"
