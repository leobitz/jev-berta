from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .jevberta import JevBerta, _resolve_checkpoint_dir
from .jevdecider import ARCHITECTURES, JevDecider


def load(model_source: str | Path | None = None, **kwargs: Any) -> JevBerta | JevDecider:
    """Load any supported checkpoint, choosing the class from its config.json.

    Checkpoints with `architecture` set to "hierarchical" or "bev" load as `JevDecider`;
    everything else loads as `JevBerta`. `kwargs` go to the chosen class.
    """
    path = _resolve_checkpoint_dir(
        model_source,
        cache_dir=kwargs.get("cache_dir"),
        local_files_only=kwargs.get("local_files_only", False),
        token=kwargs.get("token"),
    )
    config_path = path / "config.json"
    if not config_path.exists() and (path / "best_val" / "config.json").exists():
        path = path / "best_val"
        config_path = path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    cls = JevDecider if config.get("architecture") in ARCHITECTURES else JevBerta
    return cls(path, **kwargs)
