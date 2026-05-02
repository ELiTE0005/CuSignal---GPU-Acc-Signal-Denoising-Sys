# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

CuSignal is a GPU-accelerated signal processing system for automotive radar and underwater sonar. It has two tracks:

1. **MSTAR SAR ATR** — SAR image classification using a PyTorch CNN trained on MSTAR Phoenix binary datasets
2. **Synthetic Radar/Sonar** — FMCW radar and phased-array sonar simulation with GPU-based Range-Doppler processing, CA-CFAR detection, DBSCAN clustering, and multi-object tracking

The entire stack runs inside a Docker container based on NVIDIA RAPIDS 24.04 (CUDA 12.2). There is no `requirements.txt` or `setup.py` — dependencies are managed via the `Dockerfile` and `docker-compose.yml`.

## Running the Project

```bash
# Start the GPU container
docker compose up -d

# Shell into the container
docker compose exec cusignal bash

# Run MSTAR pipeline (summarize dataset)
python scripts/run_mstar_pipeline.py --summarize --mstar-root /data/MSTAR

# Train MSTAR CNN classifier
python scripts/train_mstar_cnn.py --mstar-root /data/MSTAR --epochs 40 --out checkpoints/mstar_cnn.pt
# With FFT branch (2-channel input):
python scripts/train_mstar_cnn.py --mstar-root /data/MSTAR --epochs 40 --fft-branch --gpu-fft

# Radar and sonar demos
python scripts/run_radar_pipeline.py
python scripts/run_sonar_pipeline.py
python scripts/run_radar_synthetic_demo.py
python scripts/run_sonar_synthetic_demo.py

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_mstar_phoenix.py
```

## Architecture

```
src/
├── data_loader/
│   ├── mstar_phoenix.py          # Parse MSTAR Phoenix binary format (ASCII header + raw mag/phase)
│   ├── radarscenes_loader.py     # Load RadarScenes HDF5 datasets (158 sequences, 77 GHz FMCW)
│   ├── synthetic_adc_generator.py # GPU FMCW IF signal synthesis from target parameters
│   └── synthetic_sonar.py        # Synthetic phased-array sonar signals
├── signal_processing/
│   ├── radar_pipeline.py         # Range-Doppler 2D FFT → CA-CFAR → DBSCAN clustering
│   ├── sar_pipeline.py           # SAR chip preprocessing + optional FFT log-magnitude branch
│   ├── sonar_pipeline.py         # Spatial FFT beamforming for array sonar
│   └── ego_motion.py             # Ego-velocity compensation for RadarScenes
├── models/
│   └── mstar_cnn.py              # PyTorch CNN: 3 conv blocks (32→64→128), adaptive pool, 256-unit FC
├── tracking/
│   └── cv_kalman_tracker.py      # Constant-velocity Kalman + Hungarian assignment for MOT
└── optimization/
    └── ga_optimizer.py           # Genetic algorithm running entirely on CuPy (tournament, crossover, mutation)
```

**Processing flows:**

- **Radar**: `SyntheticADCGenerator` (or `RadarScenesLoader`) → `RadarPipeline.process()` → range FFT → Doppler FFT → CA-CFAR → `cuml.DBSCAN` → visualize
- **Sonar**: Multi-element array input → `SonarPipeline.process()` → spatial window FFT → beam power map → threshold detect
- **MSTAR**: Phoenix binary/PNG chip → log compression + normalization → 88×88 resize → `MSTAR_CNN` → class prediction
- **Tracking**: Per-frame detections → `ConstantVelocityKalman2D.update()` → Hungarian assignment → track lifecycle management

## Key Configuration

- `pytest.ini` sets `pythonpath = src` — all `src/` subpackages are importable without install
- `docker-compose.yml` mounts `MSTAR_ROOT=/data/MSTAR` inside the container; the host path is set via `MSTAR_ROOT` env var
- MSTAR dataset layout expected: `<root>/train/<CLASS>/*.001` and `<root>/test/<CLASS>/*.001`
- Model checkpoints saved to `checkpoints/` directory

## Technology Stack

| Layer | Libraries |
|-------|-----------|
| GPU compute | CuPy, CUDA 12.2 |
| Signal processing | CuSignal (GPU FFT/CFAR), cuML (DBSCAN) |
| Data frames | cuDF, pandas |
| ML | PyTorch (CNN training), scikit-learn |
| Data formats | MSTAR Phoenix binary, RadarScenes HDF5, PNG/TIF |
| Container | Docker, NVIDIA RAPIDS 24.04 base image |
