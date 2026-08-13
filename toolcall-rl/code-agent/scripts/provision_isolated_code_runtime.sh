#!/usr/bin/env bash
# Provision an isolated Code/SWE runtime.  This is deliberately self-contained:
# it never imports from, writes to, or starts services for the Doc project.
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/workspace/data/code-agent-swe-smoke-20260811}"
CODE_RUNTIME_ROOT="${CODE_RUNTIME_ROOT:-${CODE_ROOT}/runtime}"
CODE_REPO_ROOT="${CODE_REPO_ROOT:-${CODE_ROOT}/work/openclaw-code-runtime}"
CODE_VENV="${CODE_VENV:-${CODE_RUNTIME_ROOT}/venv}"
CODE_SLIME_ROOT="${CODE_SLIME_ROOT:-${CODE_REPO_ROOT}/slime}"
CODE_OUTPUT_DIR="${CODE_OUTPUT_DIR:-${CODE_ROOT}/outputs/stage-c-100}"
CODE_REPOSITORY_URL="${CODE_REPOSITORY_URL:-https://github.com/springx04/Doc-agentic-RL.git}"
CODE_REPOSITORY_REF="${CODE_REPOSITORY_REF:-codex/code-agent-swe-env}"

for value in "$CODE_ROOT" "$CODE_RUNTIME_ROOT" "$CODE_REPO_ROOT" "$CODE_VENV" "$CODE_SLIME_ROOT" "$CODE_OUTPUT_DIR"; do
  case "$value" in *OpenClaw-RL*) echo "refusing Doc path: $value" >&2; exit 2;; esac
done

if ! command -v docker >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends docker.io python3-venv git ca-certificates
fi

mkdir -p "$CODE_RUNTIME_ROOT" "$CODE_OUTPUT_DIR" "$(dirname "$CODE_REPO_ROOT")"
CODE_DOCKER_SOCKET="${CODE_DOCKER_SOCKET:-${CODE_RUNTIME_ROOT}/docker.sock}"
CODE_DOCKER_DATA_ROOT="${CODE_DOCKER_DATA_ROOT:-${CODE_RUNTIME_ROOT}/docker-data}"
CODE_DOCKER_EXEC_ROOT="${CODE_DOCKER_EXEC_ROOT:-${CODE_RUNTIME_ROOT}/docker-exec}"
if ! docker -H "unix://${CODE_DOCKER_SOCKET}" info >/dev/null 2>&1; then
  rm -f "$CODE_DOCKER_SOCKET"
  nohup dockerd \
    --host "unix://${CODE_DOCKER_SOCKET}" \
    --data-root "$CODE_DOCKER_DATA_ROOT" \
    --exec-root "$CODE_DOCKER_EXEC_ROOT" \
    --pidfile "${CODE_RUNTIME_ROOT}/dockerd.pid" \
    --iptables=false --ip-masq=false --bridge=none --storage-driver=vfs \
    >"${CODE_RUNTIME_ROOT}/dockerd.log" 2>&1 &
  for _ in $(seq 1 30); do
    if docker -H "unix://${CODE_DOCKER_SOCKET}" info >/dev/null 2>&1; then break; fi
    sleep 1
  done
fi
docker -H "unix://${CODE_DOCKER_SOCKET}" info >/dev/null
if [ ! -d "$CODE_REPO_ROOT/.git" ]; then
  git clone --filter=blob:none --no-checkout "$CODE_REPOSITORY_URL" "$CODE_REPO_ROOT"
fi
git -C "$CODE_REPO_ROOT" fetch --depth=1 origin "$CODE_REPOSITORY_REF"
git -C "$CODE_REPO_ROOT" sparse-checkout init --cone
git -C "$CODE_REPO_ROOT" sparse-checkout set slime toolcall-rl/code-agent
git -C "$CODE_REPO_ROOT" checkout --detach FETCH_HEAD

python3 -m venv "$CODE_VENV"
"$CODE_VENV/bin/python" -m pip install --upgrade pip wheel
# The Code pool/exec services are CPU-only.  Training dependencies are kept in
# this venv rather than the Doc environment; the operator may add a CUDA torch
# wheel appropriate for this host before invoking the Stage-C launcher.
"$CODE_VENV/bin/python" -m pip install flask requests pytest

cat >"$CODE_RUNTIME_ROOT/code-runtime.env" <<EOF
export CODE_ROOT='$CODE_ROOT'
export CODE_REPO_ROOT='$CODE_REPO_ROOT'
export CODE_VENV='$CODE_VENV'
export CODE_SLIME_ROOT='$CODE_SLIME_ROOT'
export CODE_OUTPUT_DIR='$CODE_OUTPUT_DIR'
export CODE_ENV_SERVER_URL='http://127.0.0.1:18091'
export CODE_EXEC_SERVER_URLS='http://127.0.0.1:18092'
export DOCKER_HOST='unix://$CODE_DOCKER_SOCKET'
export PATH='$CODE_VENV/bin':\$PATH
EOF
echo "code_runtime_provisioned=$CODE_RUNTIME_ROOT"
