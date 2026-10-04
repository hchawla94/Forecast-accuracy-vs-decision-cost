"""Shared file locations. Run every script from the repository root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"          # raw M5 CSV files go here (not committed)
WORK = ROOT / "data" / "work"  # intermediate files (not committed)
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
WORK.mkdir(parents=True, exist_ok=True)
