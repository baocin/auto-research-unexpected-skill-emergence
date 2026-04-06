import numpy as np
import time

def run_experiment():
    """
    Investigates 'Streaming Drift Detection':
    A stream of data follows a sine wave, but at some point, the mean shifts.
    We use a sliding window z-score to detect this sudden change in distribution.
    """
    stream_length = 500
    drift_point = 250
    window_size = 30
    
    # 1. Generate stream: Sine wave + Drift
    t = np.linspace(0, 10, stream_length)
    signal = np.sin(t)
    
    # Apply drift (sudden shift in mean at drift_point)
    signal[drift_point:] += 2.0 
    
    # Add noise
    stream = signal + np.random.normal(0, 0.1, stream_length)
    
    # 2. Detection logic (Sliding Window Z-Score)
    detections = []
    for i in range(window_size, stream_length):
        window = stream[i-window_size : i]
        mu = np.mean(window)
        sigma = np.std(window) + 1e-6
        
        # Check if the current point is a massive outlier relative to window
        z_score = abs(stream[i] - mu) / sigma
        if z_score > 5.0: # Threshold for detection
            detections.append(i)

    # 3. Evaluation
    # Success is defined as detecting the drift point within a small error window
    true_drift = drift_point
    detection_error = float('inf')
    for d in detections:
        error = abs(d - true_drift)
        if error < detection_error:
            detection_error = error
            
    # Success metric: 1.0 if detected within +/- 5 steps, else 0.0
    success_metric = 1.0 if detection_error <= 5 else 0.0

    print(f"experiment: streaming_drift_detection")
    print(f"drift_point: {true_drift}")
    print(f"detected_at: {detections[0] if detections else 'None'}")
    print(f"detection_error: {detection_error:.2f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {time.time() - time.time()}") # Placeholder

if __name__ == "__main__":
    run_experiment()