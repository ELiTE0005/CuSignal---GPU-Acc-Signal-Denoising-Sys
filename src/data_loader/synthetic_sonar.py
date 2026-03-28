import cupy as cp

class SyntheticSonarGenerator:
    """
    Simulates raw sonar multi-beam data (time-domain signals)
    across an M-element linear array.
    """
    def __init__(self, 
                 fs=500e3,       # Sampling frequency (500 kHz)
                 num_elements=32, # M-element uniform linear array (ULA)
                 fc=100e3,        # Carrier Frequency (100 kHz)
                 c=1500.0,        # Speed of sound in water (1500 m/s)
                 pulse_duration=1e-3, # 1 ms pulse
                 max_range=150.0): # Max detection range
        
        self.fs = fs
        self.num_elements = num_elements
        self.fc = fc
        self.c = c
        self.T = pulse_duration
        self.max_range = max_range
        
        # Element spacing (half wavelength)
        self.lambda_c = self.c / self.fc
        self.d = self.lambda_c / 2.0
        
        self.num_samples = int((2.0 * self.max_range / self.c) * self.fs)
        
    def generate_array_data(self, targets):
        """
        Generates 2D complex Baseband Data (Elements x Time) on the GPU.
        
        targets is a list of dicts: [{'range': float, 'angle': float (radians), 'snr': float (linear)}]
        """
        # Time array on GPU
        t = cp.arange(self.num_samples, dtype=cp.float32) / self.fs
        
        # Initialize empty array data (num_elements x num_samples)
        array_data = cp.zeros((self.num_elements, self.num_samples), dtype=cp.complex64)
        
        # Element positions
        element_positions = cp.arange(self.num_elements) * self.d
        
        for target in targets:
            R0 = target.get('range', 50.0)
            theta = target.get('angle', 0.0)
            snr_linear = target.get('snr', 10.0)
            
            # Time of flight for center element
            tau_0 = 2.0 * R0 / self.c
            
            # Phase shifts and delays across elements
            # Arrival time difference: Delta_t = d * sin(theta) / c
            # Phase difference: Delta_phi = 2 * pi * f_c * Delta_t
            
            delta_tau = element_positions * cp.sin(theta) / self.c
            
            for m in range(self.num_elements):
                # Total delay for element m
                tau_m = tau_0 + delta_tau[m]
                
                # Pulse gating (rect function)
                pulse_mask = (t >= tau_m) & (t <= tau_m + self.T)
                
                # Baseband signal phase
                phase = -2.0 * cp.pi * self.fc * tau_m
                
                # Superpose
                array_data[m] += snr_linear * cp.exp(1j * phase) * pulse_mask
                
        # Add thermal noise
        noise = (cp.random.randn(*array_data.shape) + 1j * cp.random.randn(*array_data.shape)) * 1.0 # noise floor
        array_data += noise
            
        return array_data
