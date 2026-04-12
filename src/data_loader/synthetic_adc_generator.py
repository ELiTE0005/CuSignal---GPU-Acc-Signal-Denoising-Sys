import cupy as cp
import math

class SyntheticADCGenerator:
    """
    Simulates raw FMCW radar IF (Intermediate Frequency) signals (ADC data)
    based on the point cloud targets (range, velocity, RCS) loaded from RadarScenes.
    """
    def __init__(self, 
                 fs=10e6,         # Sampling frequency (10 MHz)
                 pulse_bw=1e9,    # Pulse Bandwidth (1 GHz)
                 pulse_duration=10e-6, # Pulse Duration (10 us)
                 num_chirps=128,  # Number of chirps per frame
                 fc=77e9):        # Carrier Frequency (77 GHz)
        
        self.fs = fs
        self.B = pulse_bw
        self.T = pulse_duration
        self.num_chirps = num_chirps
        self.fc = fc
        self.c = 3e8 # Speed of light
        
        self.slope = self.B / self.T
        self.num_adc_samples = int(self.fs * self.T)
        self.lambda_c = self.c / self.fc
        
    def generate_adc_data(self, points, max_targets: int | None = None):
        """
        Generates 2D complex ADC Data (Slow Time x Fast Time) on the GPU
        from given targets.
        
        points is a dict with 'range', 'velocity', 'rcs' (CuPy arrays or numpy;
        numpy is converted on GPU). If max_targets is set, keeps strongest
        returns by RCS (same as RadarScenes loader cap).
        """
        ranges = points["range"]
        velocities = points["velocity"]
        rcs = points["rcs"]
        if ranges is None or len(ranges) == 0:
            return cp.zeros((self.num_chirps, self.num_adc_samples), dtype=cp.complex64)

        ranges = cp.asarray(ranges, dtype=cp.float32)
        velocities = cp.asarray(velocities, dtype=cp.float32)
        rcs = cp.asarray(rcs, dtype=cp.float32)

        if max_targets is not None and ranges.shape[0] > max_targets:
            order = cp.argsort(-rcs)[: int(max_targets)]
            ranges = ranges[order]
            velocities = velocities[order]
            rcs = rcs[order]

        # Time arrays on GPU
        t_fast = cp.arange(self.num_adc_samples, dtype=cp.float32) / self.fs
        t_slow = cp.arange(self.num_chirps, dtype=cp.float32) * self.T
        
        # Initialize empty ADC data block (num_chirps x num_samples)
        adc_data = cp.zeros((self.num_chirps, self.num_adc_samples), dtype=cp.complex64)
        
        # Broadcaster arrays
        T_fast_mesh, T_slow_mesh = cp.meshgrid(t_fast, t_slow)

        # RCS is typically in dBsm, convert to linear amplitude
        linear_rcs = cp.power(10.0, rcs / 10.0)
        amplitudes = cp.sqrt(linear_rcs)
        
        for i in range(len(ranges)):
            R0 = ranges[i]
            v = velocities[i]
            A = amplitudes[i]
            
            # Note: For FMCW, the IF signal phase is approximately:
            # phase = 2 * pi * ( (S * 2 * R0 / c) * t_fast + (2 * v / lambda_c) * t_slow )
            # We ignore some smaller quadratic terms for simple synthetic simulation.
            
            f_IF = self.slope * 2 * R0 / self.c
            f_doppler = 2 * v / self.lambda_c
            
            phase = 2.0 * cp.pi * (f_IF * T_fast_mesh + f_doppler * T_slow_mesh)
            
            # Superpose onto ADC data
            adc_data += A * cp.exp(1j * phase)
            
        # Add thermal noise
        noise = (cp.random.randn(*adc_data.shape) + 1j * cp.random.randn(*adc_data.shape)) * 1.0 # noise floor
        adc_data += noise
            
        return adc_data
