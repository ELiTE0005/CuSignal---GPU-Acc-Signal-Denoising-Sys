building a high-efficiency signal processing pipeline for Radar and Sonar systems, focusing on clutter/noise suppression, Doppler filtering, and target detection. The system will leverage Genetic Algorithms (GA) to dynamically optimize processing parameters and curate/modify datasets for better downstream modeling.


Radar Pipeline :

Generated 128 chirp frames with 100 ADC samples each
Ran 2D Range-Doppler GPU FFTs
Applied CA-CFAR denoising on GPU
Detected 31 target points, clustered into 5 unique targets using RAPIDS cuML
Generated radar_output.png visualization

Sonar Pipeline :

Generated 32-element phased array sonar data (100k samples)
Ran multi-beam spatial FFT beamforming
Extracted 34,936 detection cells via thresholding
Generated sonar_output.png visualization

Both scripts now:

Run entirely on GPU via CuPy
Use RAPIDS libraries (cuML, cuDF)
Execute inside the Docker container with CUDA 12.2 and full RAPIDS suite
Successfully process real automotive signal data
