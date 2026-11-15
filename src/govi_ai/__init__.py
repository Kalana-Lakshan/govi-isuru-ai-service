"""Govi Isuru AI service: shared code for data, training, evaluation and serving."""

from pathlib import Path

__version__ = "0.1.0"

#: Repository root (…/govi-isuru-ai-service). Configs live in ``REPO_ROOT / "configs"``.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
