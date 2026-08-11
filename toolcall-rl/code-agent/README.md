# Code Agent / SWE environment

This directory is an independent text-only Code Agent environment added beside
the existing document agent.  It owns its tools, interaction world, belief
features/checkpoints, repository snapshots, rollout metadata, evaluator, and
Stage A/B/C/D entry points.  It never imports the document `bayestool` or Doc
tools and does not change the Doc entry point or checkpoint format.

The policy protocol is one exact action per assistant turn:

```xml
<tool_call>{"name":"read_file","arguments":{"path":"src/a.py"}}</tool_call>
```

or `<final>...</final>` / `<abstain>...</abstain>`.  Persistent edits are only
accepted through `apply_patch`; `run_command` rejects search/read/test/check
equivalents and restores the repository if a permitted runtime command mutates
tracked files.

The four required worlds are sampled for the same public SWE task and base
repository: `healthy`, `local_degradation`, `shared_family_fault`, and
`change`.  Each decision-level K=4/K=8 sibling gets its own lease and starts
from the same repository checkpoint.  Hidden world labels are trainer-only;
policy-visible prompts contain task state and public posterior summaries only.
World-injected failures are valid RL outcomes, while transport/Docker/evaluator
failures invalidate the whole sibling group.

## Offline local smoke

The unit tests use `LocalCodeEnvClient`, which creates disposable local copies
of a fixture repository.  They do not start Docker or contact a server.  Remote
deployment is explicit and uses the Code-only environment variables
`CODE_ENV_SERVER_URL`, `CODE_TOOL_BUDGET`, `CODE_BELIEF_CHECKPOINT`,
`CODE_Q_CHECKPOINT`, `CODE_RISK_CHECKPOINT`, `CODE_SAVE_TRAJ_DIR`,
`CODE_OUTPUT_DIR`, `CODE_CACHE_DIR`, and `CODE_LOG_DIR`.

From the repository root:

```bash
PYTHONPATH=toolcall-rl/code-agent python -m pytest -q toolcall-rl/code-agent/tests
```

The Stage C first smoke is one instance × four worlds × K=4.  It is launched
with `scripts/train_stage_c.sh`; server upload/remote testing is deliberately a
separate operator decision.

## Remote Docker smoke (isolated from Doc training)

The Code pool is a separate service at `env.server.pool_server`. It proxies to
pre-provisioned SWE-compatible Docker exec nodes, but owns only `code-lease-*`
leases. Do not reuse the Doc/SWE pool port or its output paths.

On the GPU/head host, launch the pool on an unused port with one container per
exec node:

```bash
export PYTHONPATH=/workspace/data/OpenClaw-RL/toolcall-rl/code-agent
export CODE_EXEC_SERVER_URLS=http://<code-exec-node>:5000
export CODE_ENV_SERVER_PORT=18091
export CODE_MAX_CONTAINERS_PER_NODE=1
export CODE_OUTPUT_DIR=/workspace/data/OpenClaw-RL/outputs/code-smoke-<date>
export CODE_SAVE_TRAJ_DIR=${CODE_OUTPUT_DIR}/trajectories
export CODE_LOG_DIR=${CODE_OUTPUT_DIR}/logs
python -c 'import flask, requests'  # dependency gate
bash toolcall-rl/code-agent/scripts/start_code_env.sh \
  --host 127.0.0.1 --port "${CODE_ENV_SERVER_PORT}" \
  --exec-server-urls "${CODE_EXEC_SERVER_URLS}" \
  --max-containers-per-node "${CODE_MAX_CONTAINERS_PER_NODE}"
```

Then use a dedicated one-instance Code manifest (with evaluator-private
`eval_script`) and set `CODE_ENV_SERVER_URL=http://127.0.0.1:18091`. First
verify `/healthz`, `/status`, allocation, `git_diff`, and clean evaluation;
only then run the one-instance / four-world / K=4 Stage C smoke. Keep all Code
data, logs, checkpoints, Ray temp paths, ports, and container limits distinct
from the active Doc run.
