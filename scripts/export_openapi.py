"""Write the service's OpenAPI contract to docs/openapi.json (no model needed).

    python scripts/export_openapi.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "service"))


def main() -> None:
    from app.main import app  # the lifespan (model loading) does not run for openapi()

    spec = app.openapi()
    out = REPO / "docs" / "openapi.json"
    out.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(f"{len(spec['paths'])} paths -> {out}")


if __name__ == "__main__":
    main()
