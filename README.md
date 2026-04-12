building a high-efficiency signal processing pipeline for Radar and Sonar systems, focusing on clutter/noise suppression, Doppler filtering, and target detection. The system will leverage Genetic Algorithms (GA) to dynamically optimize processing parameters and curate/modify datasets for better downstream modeling.


Radar Pipeline :

Loads **RadarScenes** sequences (`radar_data.h5`); builds IF from real detector points → 2D Range–Doppler GPU FFTs → CA-CFAR → cuML DBSCAN.  
Run: `python scripts/run_radar_pipeline.py` with `RADARSCENES_ROOT` set (see Docker compose).  
Optional offline demo: `scripts/run_radar_synthetic_demo.py`

Sonar Pipeline :

**Real audio:** `python scripts/run_sonar_pipeline.py --wav your.wav` → multi-beam spatial FFT → thresholding → `sonar_output.png`.  
Optional synthetic phased-array demo: `scripts/run_sonar_synthetic_demo.py`

Both stacks:

Run on GPU via CuPy (and RAPIDS cuML/cuDF where used)  
Execute inside the Docker container with CUDA 12.2 and the RAPIDS suite  
