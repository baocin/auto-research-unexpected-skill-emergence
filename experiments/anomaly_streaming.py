import numpy as np

def generate_stream(n, noise_prob=0.0, anomaly_idx=None, burst_size=20):
    """Generates a stream of bits, with an optional transient burst anomaly."""
    stream = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    noise_mask = np.random.rand(n) < noise_prob
    stream[noise_mask] = 1 - stream[noise_mask]

    if anomaly_idx is not None and anomaly_idx < n:
        end_idx = min(anomaly_idx + burst_size, n)
        stream[anomaly_idx:end_idx] = 1
    return stream

def detect_anomalies_streaming(stream, window_size=20):
    """Detects anomalies in a streaming fashion using a sliding window."""
    detections = []
    window_densities = []
    
    for i in range(0, len(stream) - window_size + 1, window_size):
        window = stream[i:i+window_size]
        current_density = np.mean(window)
        
        if len(window_dens_list := window_densities) >= 2:
            mu = np.mean(window_dens_list)
            sigma = np.std(window_dens_list) + 1e-6
            z_score = abs(current_density - mu) / sigma
            # Increased threshold to reduce sensitivity to zero-noise fluctuations
            if z_score > 5.0:
                detections.append(i)
        
        window_densities.append(current_density)
    return detections

def run_experiment(n, noise_levels, anomaly_positions):
    results = []
    for p in noise_levels:
        for a_pos in anomaly_positions:
            stream = generate_stream(n, noise_prob=p, anomaly_idx=a_pos)
            detections = detect_anomalies_streaming(stream)
            # Check if any detection index falls within the burst range
            detected = any(a_pos <= d <= a_pos + 20 for d in detections)
            results.append({
                'noise_prob': p,
                'anomaly_pos': a_pos,
                'detected': detected
            })
    return results

if __name__ == "__main__":
    n = 1000
    noise_levels = [0.0, 0.05, 0.1]
    anomaly_positions = [200, 500, 800]
    
    print(f"experiment: streaming_burst_detection")
    results = run_experiment(n, noise_levels, anomaly_positions)
    
    for res in results:
        print(f"p:{res['noise_prob']} pos:{res['anomaly_pos']} detected:{res['detected']}")