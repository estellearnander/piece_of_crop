"""Read-only checklist before presenting or pushing the prototype."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import MODEL_PREFIXES, check_local_resources, checkpoint_path
from analytics import read_verified_predictions, read_test_split


def main():
    resources = check_local_resources(ROOT)
    print("=== PIECE OF CROP / PROTOTYPE PREFLIGHT ===")
    print("Project:", ROOT)
    for name in ("split", "band_stats", "raw_dataset"):
        print(f"{name:18s}", "OK" if resources[name] else "MISSING")
    split = read_test_split(ROOT)
    print("Current test patches:", len(split))
    runs, warnings = read_verified_predictions(ROOT)
    print("Verified prediction CSVs:", f"{len(runs)}/18")
    print("Local model checkpoints:")
    for model in MODEL_PREFIXES:
        for input_type in ("rgb", "13band"):
            cp = checkpoint_path(ROOT, model, input_type, 0)
            print(f"  {model:14s} {input_type:6s} seed0", "FOUND" if cp.exists() else "not available")
    if warnings:
        print("Data caveats:")
        for warning in warnings[:8]:
            print(" -", warning)
        if len(warnings) > 8:
            print(" - ...", len(warnings)-8, "more")
    print("No files changed. The demo can show live predictions only when matching checkpoint files are available.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
