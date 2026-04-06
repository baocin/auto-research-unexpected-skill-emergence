import numpy as np
import time

def run_experiment():
    """
    Investigates 'Anomaly Detection SNR Breakdown':
    As the noise level increases, we measure when a fixed-magnitude spike 
    becomes indistinguishable from random fluctuations.
    """
    stream_length = 1000
    t = np.linspace(0, 50, stream_length)
    base_signal = np.sin(t)
    anomaly_indices = [200, 400, 600, 800]
    spike_magnitude = 3.0
    
    noise_levels = [0.1, 0.5, 1.0, 2.0, 4.0]
    results = []

    for sigma in noise_levels:
        noise = np.random.normal(0, sigma, stream_length)
        stream = base_signal + noise
        # Apply spikes to the noisy signal
        for idx in anomaly_indices:
            stream[idx] += spike_magnitude
        
        window_size = 30
        detections = []
        start_time = time.time()
        
        for i in range(window_size, stream_length):
            window_data = stream[i - window_size : i]
            mu = np.mean(window_data)
            std = np.std(window_data) + 1e-6
            if abs(stream[i] - mu) / std > 5.0:
                detections.append(i)
        
        end_time = time.time()
        
        true_positives = len(set(detections).intersection(set(anomaly_indices)))
        false_positives = len([d for d in detections if d not in anomaly_indices])
        
        precision = true_positives / len(detections) if len(detections) > 0 else 0
        recall = true_positives / len(anomaly_indices)

        results.append((sigma, precision, recall, end_time - start_time))

    print(f"experiment: anomaly_snr_breakdown")
    for sigma, precision, recall, duration in results:
        print(f"sigma_{sigma:.1f}_precision_{precision:.4f}_recall_{recall:.4f}_time_{duration:.4f}")

if __name__ == "__main__":
    run_experiment()