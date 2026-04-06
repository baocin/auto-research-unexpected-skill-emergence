import numpy as np

def generate_sensor_stream(n, noise_prob=0.0, sensor_failure_prob=0.0):
    """
    Generates a stream of sensor readings (e.g., temperature).
    Some sensors are 'damaged' and always report a fixed value (0.0).
    """
    t = np.linspace(0, 10, n)
    signal = np.sin(t)
    
    # Add Gaussian noise to the signal
    noise = np.random.normal(0, 0.1, n)
    stream = signal + noise
    
    # Introduce sensor failures: some sensors are 'stuck' at 0.0
    failure_mask = np.random.rand(n) < sensor_failure_prob
    stream[failure_mask] = 0.0
    
    return stream

def detect_anomalies_with_damage(stream, threshold=3.0):
    """Detects anomalies (spikes) in the sensor stream using a simple z-score."""
    detections = []
    mu = np.mean(stream)
    sigma = np.std(stream) + 1e-6
    
    for i, val in enumerate(stream):
        z_score = abs(val - mu) / sigma
        if z_score > threshold:
            detections.append(i)
    return detections

def run_experiment(n_samples, noise_levels, failure_levels):
    """
    Tests how sensor failures affect our ability to detect signal spikes.
    The spike magnitude decreases as damage increases.
    """
    results = []
    spike_idx = n_samples // 2
    
    for p_noise in noise_levels:
        for p_fail in failure_levels:
            # The spike becomes harder to detect as damage increases
            # We scale magnitude inversely with the total 'damage' (p_noise + p_fail)
            spike_magnitude = 5.0 / (1.0 + p_noise + p_fail)
            
            stream = generate_sensor_stream(n_samples, noise_prob=p_noise, sensor_failure_prob=p_fail)
            stream[spike_idx] += spike_magnitude 
            
            detections = detect_anomalies_with_damage(stream)
            detected = spike_idx in detections
            
            results.append({
                'noise': p_noise,
                'failure': p_fail,
                'mag': round(spike_magnitude, 2),
                'detected': detected
            })
    return results

if __name__ == "__main__":
    n_samples = 500
    noise_levels = [0.0, 0.3, 0.6]
    failure_levels = [0.0, 0.3, 0.6]
    
    print(f"experiment: anomaly_detection_sensitivity")
    results = run_experiment(n_samples, noise_levels, failure_levels)
    
    for res in results:
        print(f"noise:{res['noise']} fail:{res['failure']} mag:{res['mag']} detected:{res['detected']}")