import numpy as np
import time

def run_experiment():
    stream_length = 1000
    t = np.linspace(0, 50, stream_length)
    signal = np.sin(t)
    noise_std = 0.2
    stream = signal + np.random.normal(0, noise_std, stream_length)
    
    anomaly_indices = [150, 3_50, 550, 750] # typo in 3_50
    for idx in anomaly_indices:
        stream[idx] += 5.0
    
    short_window = 10
    long_window = 100
    detections = []
    
    start_time = time.time()
    for i in range(max(short_window, long_window), stream_length):
        sw = stream[i-short_window:i]
        mu_s = np.mean(sw)
        sigma_s = np.std(sw) + 1e-6
        lw = stream[i-long_window:i]
        mu_l = np.mean(lw)
        sigma_l = np.std(lw) + 1e-6
        if abs(stream[i]-mu_s)/sigma_s > 5.0 or abs(stream[i]-mu_l)/sigma_l > 5.0:
            detections.append(i)
    end_time = time.time()
    
    tp = len(set(detections).intersection(set(anomaly_indices)))
    fp = len([d for d in detections if d not in anomaly_indices])
    precision = tp / len(detections) if len(detections) > 0 else 0
    recall = tp / len(anomaly_indices)

    print(f"experiment: multi_scale_anomaly_v2")
    print(f"true_positives: {tp}")
    print(f"false_positives: {fp}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()