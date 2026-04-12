"""Write a short sine WAV for testing scripts/run_sonar_pipeline.py --wav (not for distribution)."""
import numpy as np
from scipy.io import wavfile

def main():
    rate = 48_000
    t = np.linspace(0, 0.5, rate // 2, endpoint=False)
    x = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wavfile.write("sample.wav", rate, x)
    print("Wrote sample.wav", x.shape, "samples")

if __name__ == "__main__":
    main()
