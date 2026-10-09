"""python3 run_benchmark.py --dataset dyntom --method cot --limit 1"""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parent
    for entry in (root, root / "src"):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)


if __name__ == "__main__":
    _bootstrap()
    from inference_baselines.cli import main

    raise SystemExit(main())
