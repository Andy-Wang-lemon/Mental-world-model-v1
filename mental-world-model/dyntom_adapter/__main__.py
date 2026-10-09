"""python3 dyntom_adapter/__main__.py run --input ... --output ..."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parents[1]
    for entry in (root, root / "src"):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)


if __name__ == "__main__":
    _bootstrap()
    from dyntom_adapter.cli import main

    raise SystemExit(main())
