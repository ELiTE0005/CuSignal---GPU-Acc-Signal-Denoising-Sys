"""
Flask backend for mstar_pipeline.html — serves the page and processes real MSTAR chips.

Run:
    pip install flask pillow numpy
    python backend.py
Then open http://localhost:5000
"""
from __future__ import annotations

import base64
import io
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader.mstar_phoenix import load_mstar_chip

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder=".")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_ROOT = os.path.join(BASE_DIR, "archive")

_ALLOWED_EXTS = {".001", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _mag_to_uint8(mag: np.ndarray) -> np.ndarray:
    """Log-compress and min-max normalise magnitude to uint8 grayscale."""
    x = np.log1p(mag.astype(np.float64))
    mn, mx = x.min(), x.max()
    if mx > mn:
        x = (x - mn) / (mx - mn)
    else:
        x = np.zeros_like(x)
    return (x * 255).astype(np.uint8)


def _to_png_b64(arr: np.ndarray) -> str:
    """Convert uint8 2-D array to base64 PNG string."""
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required: pip install pillow")
    img = PILImage.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _class_from_path(path: str) -> str:
    parts = path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part in ("train", "test") and i + 1 < len(parts):
            return parts[i + 1]
        if part == "Padded_imgs" and i + 1 < len(parts):
            return parts[i + 1]
    return "UNKNOWN"


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "mstar_pipeline.html")


@app.route("/api/chips")
def list_chips():
    """Return all chip files found under archive/."""
    chips = []
    for dirpath, _dirs, files in os.walk(ARCHIVE_ROOT):
        for fn in sorted(files):
            if fn.startswith("."):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext not in _ALLOWED_EXTS:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, BASE_DIR).replace("\\", "/")
            chips.append({
                "path": rel,
                "name": fn,
                "class": _class_from_path(full),
                "type": "phoenix" if ext == ".001" else "image",
            })
    return jsonify(chips)


@app.route("/api/process-raw", methods=["POST"])
def process_raw():
    """
    Load a chip and return its log-compressed grayscale image as base64 PNG.

    Accepts either:
      - multipart/form-data with field 'file' (any supported format)
      - form field 'path' (relative path inside archive/)
    """
    tmp_path = None
    try:
        if "file" in request.files:
            uploaded = request.files["file"]
            suffix = os.path.splitext(uploaded.filename)[1] or ".bin"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            uploaded.save(tmp_path)
            chip_path = tmp_path
        elif "path" in request.form:
            rel = request.form["path"]
            abs_path = os.path.realpath(os.path.join(BASE_DIR, rel))
            archive_abs = os.path.realpath(ARCHIVE_ROOT)
            # Security: only allow files inside archive/
            if not abs_path.startswith(archive_abs + os.sep) and abs_path != archive_abs:
                return jsonify({"error": "Access denied"}), 403
            ext = os.path.splitext(abs_path)[1].lower()
            if ext not in _ALLOWED_EXTS:
                return jsonify({"error": "Unsupported file type"}), 400
            chip_path = abs_path
        else:
            return jsonify({"error": "Provide 'file' or 'path'"}), 400

        chip = load_mstar_chip(chip_path)
        raw_u8 = _mag_to_uint8(chip.magnitude)
        h, w = raw_u8.shape

        return jsonify({
            "raw_b64": _to_png_b64(raw_u8),
            "shape": {"h": h, "w": w},
            "target_class": _class_from_path(chip_path),
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    print("Starting MSTAR backend — open http://localhost:5000")
    app.run(debug=True, port=5000)
