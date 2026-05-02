CuSignal — GPU signal processing with an **MSTAR SAR ATR** track (classification on real SAR chips).

### MSTAR layout

Put the public MSTAR release under **`archive/MSTAR/`** (see **`archive/MSTAR/README.txt`**):

- `train/<CLASS>/...` — Phoenix `.001` / `.cphd` chips or PNG/TIF
- `test/<CLASS>/...`

Docker mounts **`./archive/MSTAR` → `/data/MSTAR`**; **`MSTAR_ROOT=/data/MSTAR`** is set in `docker-compose.yml`.

### Commands (inside the GPU container)

```bash
# Summarize chip counts
python scripts/run_mstar_pipeline.py --summarize --mstar-root /data/MSTAR

# Visualize one chip (spatial + optional FFT branch)
python scripts/run_mstar_pipeline.py --chip /path/to/chip.001 --fft --out mstar_view.png

# Train CNN classifier
python scripts/train_mstar_cnn.py --mstar-root /data/MSTAR --epochs 40 --out checkpoints/mstar_cnn.pt

# Optional: 2-channel input (spatial + FFT log-magnitude), CuPy FFT in training
python scripts/train_mstar_cnn.py --mstar-root /data/MSTAR --fft-branch --gpu-fft
```

### Pipeline

1. **`mstar_phoenix.py`** — Phoenix ASCII header + binary magnitude (or interleaved mag/phase), or **Pillow** for images.
2. **`sar_pipeline.py`** — log scaling, resize (default **88×88**), optional **FFT** branch (still GPU-friendly via CuPy when enabled).
3. **`models/mstar_cnn.py`** — PyTorch CNN (**torch** installed in the Dockerfile).

Automotive FMCW / sonar demos remain under **`scripts/`** (radar/sonar) but are separate from this SAR benchmark.

