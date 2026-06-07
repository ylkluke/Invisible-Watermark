from __future__ import annotations

import shutil
from pathlib import Path


def maybe_clean_outputs(*, repo_root: Path, outputs_dir: str | Path, enabled: bool) -> Path:
    """
    Delete the outputs directory before generating new outputs.

    Safety:
    - Only deletes paths inside repo_root.
    - Refuses to delete repo_root itself.
    """
    out = Path(outputs_dir)
    if not out.is_absolute():
        out = (repo_root / out).resolve()
    else:
        out = out.resolve()

    repo = repo_root.resolve()
    if out == repo:
        raise ValueError("refuse to delete repo root as outputs_dir")
    if repo not in out.parents:
        raise ValueError(f"refuse to delete outputs_dir outside repo root: {out}")

    if not enabled:
        return out

    if out.exists():
        shutil.rmtree(out)
    return out

