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
    # We use a sliding window to calculate local mean and std
    for i in range(len(stream) - window_size):
        window = stream[i:i+window_size]
        mu = np.mean(window)
        sigma = np.std(window) + 1e-6
        
        # Check the point immediately following the window
        val = stream[i + window_size]
        if abs(val - mu) / sigma > threshold_multiplier:
            detections.append(i + window_size)
    return detections

def run_experiment(n_samples, noise_levels, failure_levels):
    """
    Tests the detectability of a spike under increasing levels of 
    sensor degradation and measurement noise.
    """
    results = []
    spike_idx_relative = 0.5 # Spike at midpoint
    
    for p_noise in noise_levels:
        for p_fail in failure_levels:
            trials = 30
            successes = 0
            start_time = time.time()
            
            for _ in range(trials):
                # Generate base stream
                stream, t = generate_sensor_int_helper(n_samples, p_noise, p_fail)
                
                # Inject a spike at the 50% mark
                spike_idx = int(n_samples * spike_idx_relative)
                # Magnitude is scaled to be detectable but vulnerable
                spike_mag = 5.0 / (1.0 + p_noise + p_fail)
                stream[spike_idx] += spike_mag
                
                # Detect
                detections = detect_anomalies_with_dynamic_threshold(stream)
                if spike_idx in detections:
                    successes += 1
            
            duration = time.time() - start_time
            results.append({
                'noise': p_noise,
                'failure': p_fail,
                'mag': 0, # Placeholder for simplicity in output
                'success_rate': successes/trials,
                'avg_time': duration/trials
            })
    return results

def generate_sensor_int_helper(n, noise_std, failure_prob):
    t = np.linspace(0, 4 * np.pi, n)
    signal = np.sin(t)
    noise = np.random.normal(0, noise_std, n)
    stream = signal + noise
    if failure_prob > 0:
        mask = np.random.rand(n) < failure_prob
        stream[mask] = 0.0
    return stream, t

# Redefining run_experiment to use the helper correctly
def run_experiment_final(n_samples, noise_levels, failure_levels):
    results = []
    spike_idx_rel = 0.5
    for p_noise in noise_levels:
        for p_int_fail in failure_levels:
            trials = 30
            successes = 0
            start_time = time.time()
            for _ in range(trials):
                stream, _ = generate_sensor_int_helper(n_samples, p_noise, p_int_fail)
                s_idx = int(n_samples * spike_idx_rel)
                mag = 5.0 / (1.0 + p_noise + p_int_fail)
                stream[s_idx] += mag
                if s_idx in detect_anomalies_with_dynamic_threshold(stream):
                    successes += 1
            duration = time.time() - start_time
            results.append({
                'noise': p_noise, 'failure': p_int_fail,
                'success_rate': successes/trials, 'avg_time': duration/trials
            })
    return results

if __name__ == "__main__":
    n_samples = 500
    noises = [0.1, 0.5, 1.0]
    failures = [0.0, 0.2, 0.4]
    print(f"experiment: sensor_anomaly_dynamic_window")
    res = run_experiment_final(n_samples, noises, failures)
    for r in res:
        print(f"noise:{r['noise']} fail:{r['failure']} success:{r['success_rate']:.2f} time:{r['avg_time']:.4f}")