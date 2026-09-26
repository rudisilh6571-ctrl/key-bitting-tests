#!/usr/bin/env bash
# Launch KeyForge3D from its virtualenv, creating it on first run.
# uv's managed Python ships with Tk, so no system Tk package is needed.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if [ ! -x .venv/bin/python ]; then
  uv venv --managed-python --python 3.12 .venv
  uv pip install --python .venv -r requirements.txt
fi
exec .venv/bin/python keyforge3d.py "$@"
