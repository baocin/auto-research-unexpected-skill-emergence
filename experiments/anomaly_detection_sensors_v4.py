import numpy as np
import time

def generate_sensor_data(n, noise_std=0.1, failure_prob=0.0):
    """
    Generates a continuous sine wave signal with Gaussian noise 
    and random sensor 'dead' zones (permanent zeroing).
    """
    t = np.linspace(0, 4 * np.pi, n)
    signal = np.sin(t)
    
    # Add Gaussian measurement noise
    noise = np.random.normal(0, noise_std, n)
    stream = signal + noise
    
    # Introduce sensor failures (dead sensors reporting 0.0)
    if failure_prob > 0:
        failure_mask = np.random.rand(n) < failure_prob
        stream[failure_mask] = 0.0
    return stream, t

def detect_anomalies_with_dynamic_threshold(stream, threshold_multiplier=3.0):
    """
    Detects spikes using a rolling window Z-score to handle non-stationary signal.
    """
    detections = []
    window_size = 20
    for i in range(len(stream) - window_size):
        window = stream[i:i+window_size]
        mu = np.mean(window)
        sigma = npstd = np.std(window) + 1e-6
        
        # Check the point immediately following the window
        val = stream[i + window_size]
        if abs(val - mu) / sigma > threshold_multiplier:
            det_idx = i + window_size
            if not detections or det_idx != detections[-1]:
                detections.append(det_idx)
    return detections

def run_experiment(n_samples, noise_levels, failure_levels):
    """
    Tests the detectability of a spike under increasing levels of 
    sensor degradation and measurement noise.
    """
    results = []
    spike_idx_rel = 0.5
    
    for p_noise in noise_levels:
        for p_fail in failure_levels:
            trials = 30
            successes = 0
            start_time = time.time()
            
            # The spike magnitude decays as damage increases
            spike_magnitude_base = 5.0
            
            for _ in range(trials):
                stream, _ = generate_sensor_data(n_samples, p_noise, p_fail)
                s_idx = int(n_samples * spike_idx_rel)
                # Magnitude scales inversely with total corruption
                mag = spike_magnitude_base / (1.0 + p_noise + p_fail)
                stream[s_idx] += mag
                
                if s_idx in detect_anomalies_with_dynamic_threshold(stream):
                    successes += 1
            
            duration = time.time() - start_time
            results.append({
                'noise': p_noise,
                'failure': p_fail,
                'success_rate': successes/trials,
                'avg_time': duration/trials
            })
    return results

if __name__ == "__main__":
    n_samples = 500
    noises = [0.1, 0.5, 1.0]
    failures = [0.0, 0.2, 0.4]
    print(f"experiment: sensor_anomaly_sensitivity_v4")
    res = run_experiment(n_samples, noises, failures)
    for r in res:
        print(f"noise:{r['noise']} fail:{r['failure']} success:{r['success_rate']:.2f} time:{r['avg_time']:.4f}")