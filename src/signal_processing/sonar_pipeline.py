import cupy as cp
import numpy as np
from scipy.signal.windows import chebwin

class SonarPipeline:
    def __init__(self, 
                 fs=500e3, 
                 num_elements=32,
                 fc=100e3,
                 c=1500.0):
        
        self.fs = fs
        self.num_elements = num_elements
        self.fc = fc
        self.c = c
        self.lambda_c = self.c / self.fc
        self.d = self.lambda_c / 2.0
        
        # Create Chebyshev window on CPU and transfer to GPU
        window_cpu = chebwin(self.num_elements, at=60)  # 60dB attenuation
        self.window_spatial = cp.asarray(window_cpu)
        
    def process_spatial_fft(self, array_data):
        """
        Multi-beam spatial processing using Spatial FFT.
        array_data: (num_elements x num_samples) complex cupy array
        
        Returns Range-Angle (Azimuth) detection map.
        """
        num_samples = array_data.shape[1]
        
        # Apply spatial windowing to reduce sidelobes
        array_windowed = array_data * self.window_spatial[:, cp.newaxis]
        
        # Zero-pad for higher angular resolution
        n_fft_spatial = 128
        
        # Perform 1D FFT along the elements array (axis=0)
        # This converts spatial domain across sensors to angular frequency (Azimuth)
        beam_map = cp.fft.fftshift(cp.fft.fft(array_windowed, n=n_fft_spatial, axis=0), axes=0)
        
        # Compute magnitude squared for power
        power_map = cp.abs(beam_map)**2
        
        # The x-axis is discrete time (range): R = t * c / 2
        # The y-axis is discrete angular frequency: sin(theta) = k * lambda_c / (N * d)
        
        return power_map
        
    def extract_targets(self, power_map, threshold=5.0):
        """
        Simple thresholding for target extraction from the beam map.
        threshold: multiplier wrt median background noise level
        """
        # Estimate background noise (median)
        noise_floor = cp.median(power_map)
        
        detection_mask = power_map > (noise_floor * threshold)
        
        return detection_mask, power_map
