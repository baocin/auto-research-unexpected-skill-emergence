import numpy as np
import time

def generate_sensor_data(n, noise_std=0.1, failure_prob=0.0):
    t = np.linspace(0, 4 * np.pi, n)
    signal = np.sin(t)
    noise = np.random.normal(0, noise_std, n)
    stream = signal + noise
    if failure_prob > 0:
        failure_mask = np.random.rand(n) < failure_prob
        stream[failure_mask] = 0.0
    return stream, t

def detect_anomalies(stream, threshold=3.0):
    detections = []
    mu = np.mean(stream)
    sigma = np.std(stream) + 1e-6
    for i, val in enumerate(stream):
        if abs(val - mu) / sigma > threshold:
            detections.append(i)
    return detections

def run_experiment(n_samples, noise_levels, failure_levels):
    results = []
    spike_idx = n_samples // 2
    for p_noise in noise_levels:
        for p_fail in failure_levels:
            trials = 20
            successes = 0
            start_time = time.time()
            
            # The spike magnitude decays as damage increases
            spike_magnitude = 5.0 / (1.0 + p_noise + p_fail)
            
            for _ in range(trials):
                stream, _ = generate_sensor_data(n_samples, p_noise, p_fail)
                stream[spike_idx] += spike_magnitude 
                if spike_idx in detect_anomalies(stream):
                    successes += 1
            
            duration = time.time() - start_time
            results.append({
                'noise': p_noise, 'failure': p_fail,
                'mag': round(spike_magnitude, 2),
                'success_rate': successes/trials,
                'avg_time': duration/trials
            })
    return results

if __name__ == "__main__":
    n_samples = 500
    noises = [0.1, 0.5, 1.0]
    failures = [0.0, 0.2, 0.4]
    print(f"experiment: sensor_anomaly_sensitivity")
    res = run_experiment(n_samples, noises, failures)
    for r in res:
        print(f"noise:{r['noise']} fail:{r['failure']} mag:{r['mag']} success:{r['success_rate']:.2f} time:{r['avg_time']:.4f}")