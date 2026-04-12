"""Backward-compatible alias — use scripts/run_sonar_pipeline.py --wav ..."""
import pathlib
import runpy

if __name__ == "__main__":
    runpy.run_path(str(pathlib.Path(__file__).resolve().parent / "run_sonar_pipeline.py"), run_name="__main__")
