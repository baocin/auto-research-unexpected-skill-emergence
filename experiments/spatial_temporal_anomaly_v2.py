import numpy as np
import time

def run_experiment():
    """
    Investigates 'Spatial-Temporal Anomaly Detection':
    The detector scans a 1D stream and identifies spikes using a sliding window z-score.
    We measure precision and recall relative to known anomaly indices.
    """
    num_sensors = 100
    stream_length = 1000
    
    # 1. Generate signal: Sine wave + noise
    t = np.linspace(0, 50, stream_length)
    signal = np.sin(t)
    noise = np.random.normal(0, 0.2, stream_length)
    stream = signal + noise
    
    # 2. Inject anomalies (spikes at specific indices)
    anomaly_indices = [150, 350, 550, 750]
    for idx in anomaly_indices:
        stream[idx] += 5.0
    
    # 3. Detection logic (Scanning the whole stream)
    window_size = 20
    detections = []
    
    start_time = time.time()
    for i in range(window_int := window_size, stream_length):
        window_data = stream[i - window_size : i]
        mu = np.mean(window_data)
        sigma = np.std(window_data) + 1e-6
        
        # Check if the current point is an outlier
        z_score = abs(stream[i] - mu) / sigma
        if z_score > 5.0:
            detections.append(i)
    end_time = time.time()
    
    # 4. Evaluation
    true_positives = len(set(detections).intersection(set(anomaly_indices)))
    false_positives = len(set(dete_indices := [d for d in detections if d not in anomaly_indices]))
    
    precision = true_positives / len(detections) if len(detections) > 0 else 0
    recall = true_positives / len(anomaly_indices)

    print(f"experiment: spatial_temporal_anomaly_v2")
    print(f"true_positives: {true_positives}")
    print(f"false_positives: {len(dete_indices)}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()