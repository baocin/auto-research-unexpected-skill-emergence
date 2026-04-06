import numpy as np
import time

def run_experiment():
    """
    Investigates 'Multi-Scale Drift Detection':
    Tests if a fixed z-score threshold can detect drifts of varying magnitudes.
    We measure success as the ability to detect the drift within a small window error.
    """
    stream_length = 500
    drift_point = 250
    window_size = 30
    
    # Test different drift magnitudes (from subtle to obvious)
    drift_magnitudes = [0.1, 0.5, 1.0, 2.0, 5.0]
    results = []

    for magnitude in drift_magnitudes:
        t = np.linspace(0, 10, stream_length)
        signal = np.sin(t)
        # Apply the drift
        signal[drift_point:] += magnitude
        
        # Add noise
        stream = signal + np.random.normal(0, 0.2, stream_length)
        
        detections = []
        for i in range(window_size, stream_length):
            window = stream[i-window_size : i]
            mu = np.mean(window)
            sigma = np.std(window) + 1e-6
            z_score = abs(stream[i] - mu) / sigma
            if z_score > 5.0:
                detections.append(i)

        # Evaluation
        detection_error = float('inf')
        for d in detections:
            error = abs(d - drift_point)
            if error < detection_error:
                detection_error = error
        
        success = 1.0 if detection_error <= 5 else 0.0
        results.append((magnitude, success, detection_error))

    print(f"experiment: multi_scale_drift_detection")
    for mag, success, err in results:
        print(f"mag_{mag:.1f}_success_{success:.4f}_err_{err:.2f}")

if __name__ == "__main__":
    run_experiment()