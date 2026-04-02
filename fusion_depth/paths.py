from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VGGT_ROOT = REPO_ROOT / "vggt"
DA3_SRC_ROOT = REPO_ROOT / "Depth-Anything-3" / "src"


def ensure_repo_paths() -> None:
    for path in (VGGT_ROOT, DA3_SRC_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


ensure_repo_paths()

