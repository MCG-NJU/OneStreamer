#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${EVAL_PYTHON:-python}" "$ROOT/run_script/eval_queue.py" --bench odvbench "$@"
