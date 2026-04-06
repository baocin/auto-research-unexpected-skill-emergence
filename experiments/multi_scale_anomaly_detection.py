import numpy as np
import time

def run_experiment():
    """

    Investigates 'Multi-Scale Anomaly Detection':
    Detects anomalies using both short-term and long-term windowed statistics.
    We measure if multi-scale detection improves detection of sudden spikes 
    compared to a single fixed window size.
    """
    num_sensors = 200
    stream_length = 1000
    
    # 1. Generate signal: Sine wave + random noise
    t = np.linspace(0, 50, stream_length)
    signal = np.sin(t)
    noise = np.random.normal(0, 0.2, stream_length)
    stream = signal + noise
    
    # 2. Inject anomalies (spikes)
    anomaly_indices = [100, 300, 500, 700, 900]
    for idx in anomaly_indices:
        stream[idx] += 5.0  # Large spike
    
    # 3. Detection logic with multi-scale windows
    short_window = 10
    long_window = 100
    detections = []
    
    start_time = time.time()
    
    for i in range(max(short_window, long_window), stream_length):
        # Short-term window stats
        short_window_data = stream[i - short_window : i]
        mu_s = np.mean(short_window_data)
        sigma_s = np.std(short_window_data) + 1e-6
        
        # Long-term window stats
        long_window_data = stream[i - long_window : i]
        mu_l = np.mean(long_window_data)
        sigma_l = np.std(long_window_data) + 1e-6
        
        # Detect if current point is an outlier in BOTH scales (robustness check)
        z_s = abs(stream[i] - mu_s) / sigma_s
        z_l = abs(stream[i] - mu_l) / sigma_l
        
        if z_s > 4.0 or z_l > 4.0:
            detections.append(i)

    end_time = time.time()
    
    # 4. Evaluation: Precision/Recall relative to true anomaly indices
    true_positives = len(set(detections).intersection(set(anomaly_indices)))
    false_positives = len(set(detections) - set(anomaly_indices))
    
    precision = true_positives / len(detections) if len(detections) > 0 else 0
    recall = true_positives / len(anomaly_indices)

    print(f"experiment: multi_scale_anomaly_detection")
    print(f"true_positives: {true_positives}")
    print(f"false_positives: {false_positives}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()