"""Standalone locations. QCNN_DATA_ROOT contains test/<id>/<id>.jpg."""
from pathlib import Path
import os

ROOT = str(Path(__file__).resolve().parents[1])
DATA_ROOT = str(Path(os.environ.get("QCNN_DATA_ROOT", str(Path(ROOT) / "data"))).expanduser().resolve())
