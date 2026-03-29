import sys
import os
import cupy as cp
import cusignal
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_loader.synthetic_sonar import SyntheticSonarGenerator
from signal_processing.sonar_pipeline import SonarPipeline

def main():
    print("Initializing Sonar Multi-Beam Pipeline components...")
    sonar_gen = SyntheticSonarGenerator(num_elements=32, fc=100e3, max_range=150.0)
    pipeline = SonarPipeline(num_elements=32, fc=100e3)
    
    print("Generating synthetic Sonar Array data on GPU...")
    # Targets (range in m, angle in radians, SNR linear)
    targets = [
        {'range': 60.0, 'angle': 0.2, 'snr': 50.0},
        {'range': 110.0, 'angle': -0.4, 'snr': 30.0}
    ]
    
    array_data = sonar_gen.generate_array_data(targets)
    print(f"Generated Array Data shape: {array_data.shape}")
    
    print("Running Multi-Beam Spatial FFT Processing...")
    power_map = pipeline.process_spatial_fft(array_data)
    
    print("Extracting targets via Thresholding...")
    detections, _ = pipeline.extract_targets(power_map, threshold=10.0)
    print(f"Found {int(cp.sum(detections))} detection cells.")
    
    print("Saving visualization as sonar_output.png...")
    plt.figure(figsize=(8, 6))
    
    pm_np = cp.asnumpy(power_map)
    # We transpose for better visualization (Range on X, Angle on Y)
    db_map = 10 * cp.asnumpy(cp.log10(power_map.T + 1e-12))
    
    plt.imshow(db_map, aspect='auto', cmap='viridis')
    plt.title("Sonar Multi-Beam Range-Angle Power Map (dB)")
    plt.xlabel("Spatial Frequency (Angle Bins)")
    plt.ylabel("Time (Range Bins)")
    plt.colorbar(label='Power (dB)')
    
    plt.tight_layout()
    plt.savefig('sonar_output.png')
    print("Done!")

if __name__ == "__main__":
    main()
