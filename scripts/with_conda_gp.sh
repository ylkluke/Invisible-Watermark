#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="gp"

usage() {
  cat <<EOF
Run a command using the conda environment: ${ENV_NAME}

Usage:
  scripts/with_conda_gp.sh <command> [args...]

Examples:
  scripts/with_conda_gp.sh python -V
  scripts/with_conda_gp.sh python experiments/edge_wm_train.py --help
  scripts/with_conda_gp.sh python experiments/edge_wm_infer.py --help
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -eq 0 ]]; then
  usage
  exit 0
fi

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    return 0
  fi

  # In many WSL installs, conda is only available after sourcing conda.sh.
  local candidates=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "$HOME/mambaforge/etc/profile.d/conda.sh"
    "$HOME/miniforge3/etc/profile.d/conda.sh"
  )
  for p in "${candidates[@]}"; do
    if [[ -f "$p" ]]; then
      # shellcheck disable=SC1090
      source "$p"
      break
    fi
  done

  command -v conda >/dev/null 2>&1
}

if ! ensure_conda; then
  echo "conda not found in PATH. Open a shell where conda is initialized, or install conda." >&2
  exit 2
fi

# Robustly initialize conda for non-interactive shells without requiring `conda init`.
# Prefer the hook when available; fallback to conda.sh already sourced above.
if conda shell.bash hook >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  eval "$(conda shell.bash hook)"
fi

conda activate "$ENV_NAME" >/dev/null 2>&1 || {
  echo "failed to activate conda env: ${ENV_NAME}" >&2
  echo "Available envs:" >&2
  conda env list >&2 || true
  exit 2
}

export PYTHONDONTWRITEBYTECODE=1

exec "$@"
