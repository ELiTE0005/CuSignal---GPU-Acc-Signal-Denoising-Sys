"""Backward-compatible alias — use scripts/run_radar_pipeline.py."""
import pathlib
import runpy

if __name__ == "__main__":
    runpy.run_path(str(pathlib.Path(__file__).resolve().parent / "run_radar_pipeline.py"), run_name="__main__")
