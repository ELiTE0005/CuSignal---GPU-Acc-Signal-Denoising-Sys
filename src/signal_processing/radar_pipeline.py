import cupy as cp
import cupyx.scipy.ndimage as ndimage
import math

class RadarPipeline:
    def __init__(self, 
                 num_chirps=128, 
                 num_samples=256,
                 fs=10e6,
                 slope=1e14):
        
        self.num_chirps = num_chirps
        self.num_samples = num_samples
        self.fs = fs
        self.slope = slope
        
        # Hann windows for FFTs
        self.window_fast = cp.hanning(self.num_samples)
        self.window_slow = cp.hanning(self.num_chirps)
        
    def process_range_doppler(self, adc_data):
        """
        Computes 2D Range-Doppler map from Raw ADC Data.
        adc_data: (num_chirps x num_samples) complex cupy array
        """
        # 1. Range FFT (Fast-Time)
        # Apply window along fast-time
        adc_windowed = adc_data * self.window_fast
        # FFT along axis 1
        range_fft = cp.fft.fft(adc_windowed, axis=1)
        
        # 2. Doppler FFT (Slow-Time)
        # Apply window along slow-time
        range_fft_windowed = range_fft * self.window_slow[:, cp.newaxis]
        # FFT along axis 0, and shift zero frequency to center
        range_doppler = cp.fft.fftshift(cp.fft.fft(range_fft_windowed, axis=0), axes=0)
        
        return range_doppler
        
    def ca_cfar_2d(self, range_doppler_map, 
                   train_cells=(8, 4),    # (Range, Doppler)
                   guard_cells=(4, 2),    # (Range, Doppler)
                   pfa=1e-5):             # Probability of False Alarm
        """
        2D Cell-Averaging CA-CFAR implemented natively in CuPy.
        """
        # Get Power magnitude
        power_map = cp.abs(range_doppler_map)**2
        
        N_train = train_cells[0]
        N_guard = guard_cells[0]
        M_train = train_cells[1]
        M_guard = guard_cells[1]
        
        # Create convolution kernel mask for background averaging
        kernel_size_r = 2 * (N_train + N_guard) + 1
        kernel_size_d = 2 * (M_train + M_guard) + 1
        
        kernel = cp.ones((kernel_size_d, kernel_size_r), dtype=cp.float32)
        
        # Zero-out the guard cells and CUT (Cell Under Test)
        center_d = kernel_size_d // 2
        center_r = kernel_size_r // 2
        
        kernel[center_d - M_guard : center_d + M_guard + 1, 
               center_r - N_guard : center_r + N_guard + 1] = 0.0
               
        num_train_cells = cp.sum(kernel)
        kernel = kernel / num_train_cells
        
        # Apply 2D convolution for fast parallel cell averaging
        noise_floor = ndimage.convolve(power_map, kernel, mode='constant', cval=0.0)
        
        # Calculate Threshold multiplier (alpha)
        # alpha = num_train_cells * (pfa^(-1/num_train_cells) - 1)
        alpha = num_train_cells * (power(pfa, -1.0 / num_train_cells) - 1)
        
        threshold_map = noise_floor * alpha
        
        # Detection Mask
        detections = power_map > threshold_map
        
        return detections, power_map, threshold_map
        
    def cluster_detections(self, detections, power_map, eps=3.0, min_samples=2):
        """
        Uses cuML DBSCAN to cluster raw CA-CFAR detections.
        Returns a plain pandas DataFrame to avoid cuDF/Numba CUDA context conflicts.
        """
        import cuml
        import pandas as pd

        # Extract indices where detections are True
        doppler_idx, range_idx = cp.where(detections)

        if len(range_idx) == 0:
            return pd.DataFrame({'range_bin': [], 'doppler_bin': [], 'power': [], 'cluster_id': []})

        # Build (N, 2) CuPy float32 array — cuML accepts this natively,
        # no cuDF/Numba bridge involved, so no CUDA_ERROR_INVALID_CONTEXT
        X = cp.stack([range_idx, doppler_idx], axis=1).astype(cp.float32)
        power_vals = power_map[detections]

        dbscan = cuml.DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)

        # Pull everything to CPU — pandas DataFrame has no CUDA context dependency
        df = pd.DataFrame({
            'range_bin': cp.asnumpy(range_idx),
            'doppler_bin': cp.asnumpy(doppler_idx),
            'power': cp.asnumpy(power_vals),
            'cluster_id': cp.asnumpy(labels)
        })

        return df

def power(base, exp):
    return math.pow(base, float(exp))