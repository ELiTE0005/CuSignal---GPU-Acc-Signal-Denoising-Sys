import sys
import os

# Initialize Numba CUDA context first to avoid CuPy and cuML/cuDF context conflict
from numba import cuda
cuda.current_context()

import cupy as cp
import cusignal
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader.synthetic_adc_generator import SyntheticADCGenerator
from signal_processing.radar_pipeline import RadarPipeline

def main():
    print("Initializing Radar Pipeline components...")
    adc_gen = SyntheticADCGenerator(num_chirps=128, pulse_duration=10e-6, pulse_bw=1e9)
    pipeline = RadarPipeline(num_chirps=128, num_samples=adc_gen.num_adc_samples)
    
    # 1. Synthesize target points (simulating outputs from RadarScenes load)
    print("Generating synthetic ADC data on GPU...")
    # Two targets
    points = {
        'range': cp.array([50.0, 120.0]),
        'velocity': cp.array([15.0, -10.0]), # m/s
        'rcs': cp.array([10.0, 5.0]) # dBsm
    }
    
    adc_data = adc_gen.generate_adc_data(points)
    print(f"Generated ADC Data shape: {adc_data.shape}")
    
    # 2. Process
    print("Running 2D Range-Doppler GPU FFTs...")
    range_doppler = pipeline.process_range_doppler(adc_data)
    
    # 3. GPU CA-CFAR Denoising
    print("Applying CuPy-Accelerated CA-CFAR Denoising...")
    detections, power_map, threshold = pipeline.ca_cfar_2d(range_doppler, 
                                                              train_cells=(8, 4), 
                                                              guard_cells=(4, 2), 
                                                              pfa=1e-5)
    
    num_det = cp.sum(detections)
    print(f"Algorithm detected {num_det} target points after noise suppression.")
    
    # 4. Clustering (DBSCAN via RAPIDS cuML and returned as cuDF DataFrame)
    print("Applying RAPIDS cuML clustering (DBSCAN) to detections...")
    df_targets = pipeline.cluster_detections(detections, power_map, eps=5.0, min_samples=1)
    
    print("\n--- Aggregated Radar Targets ---")
    if len(df_targets) > 0:
        print(df_targets.to_string(index=False))
        print(f"Total Unique Targets (Clusters): {df_targets['cluster_id'].nunique()}")
    else:
        print("No targets found.")
    print("--------------------------------\n")
    
    # Visualization
    print("Visualizing results (saving to radar_output.png)...")
    pm_np = cp.asnumpy(power_map)
    det_np = cp.asnumpy(detections)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    # Convert to dB for visualization
    db_map = 10 * cp.asnumpy(cp.log10(power_map + 1e-12))
    plt.imshow(db_map, aspect='auto', cmap='jet')
    plt.title("Range-Doppler Power Map (dB)")
    plt.xlabel("Fast Time (Range Bins)")
    plt.ylabel("Slow Time (Doppler Bins)")
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(det_np, aspect='auto', cmap='gray')
    plt.title("CA-CFAR Detections")
    plt.xlabel("Fast Time (Range Bins)")
    plt.ylabel("Slow Time (Doppler Bins)")
    
    plt.tight_layout()
    plt.savefig('radar_output.png')
    print("Done! See radar_output.png")

if __name__ == "__main__":
    main()