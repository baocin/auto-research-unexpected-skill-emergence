import numpy as np
import time

def run_experiment():
    """
    Investigates if structural sensor damage (zeroing data) 
    impacts the ability to detect true anomalies in a noisy stream.
    """
    n_samples = 500
    p_damage = 0.2  # 20% of sensors are 'dead' (return 0)
    anomaly_magnitude = 10.0
    noise_std = 1.0
    num_trials = 20

    results = []

    for trial in range(num_trials):
        # Generate ground truth: a stream of Gaussian noise
        true_stream = np.random.normal(0, noise_std, n_samples)
        
        # Inject an anomaly at index 250
        true_stream[250] = anomaly_magnitude
        
        # Create the 'damaged' stream (zeroing out some sensors)
        damaged_stream = true_stream.copy()
        damage_mask = np.random.rand(n_samples) < p_damage
        damaged_stream[damage_mask] = 0.0
        
        # Detection logic: Z-score based anomaly detection
        # We use a rolling window to avoid the anomaly itself biasing the mean too much
        window_size = 20
        detections = []
        
        # Calculate global stats from 'undamaged' parts (simulating sensor-agnostic monitoring)
        # In reality, we only see the damaged stream.
        running_mean = np.zeros(n_samples)
        running_std = np.zeros(n_samples)
        
        for i in range(window_size, n_samples):
            window = damaged_stream[i-window_size:i]
            running_mean[i] = np.mean(window)
            running_std[i] = np.std(window)

        # Detection threshold (Z > 3)
        for i in range(n_samples):
            if running_std[i] > 0:
                z_score = abs(damaged_stream[i] - running_mean[i]) / running_std[i]
                if z_score > 3.0:
                    detections.append(i)

        # Evaluate performance
        # True Positive: Anomaly was detected (index 250)
        # False Positive: A zeroed-out sensor or noise spike was detected
        tp = 1 if 250 in detections else 0
        fp = len([d for d in detections if d != 250])
        
        # Precision/Recall calculation
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp # Since there is only one anomaly
        
        results.append({
            'tp': tp,
            'fp': fp,
            'precision': precision,
            'recall': recall
        })

    avg_prec = np.mean([r['precision'] for r in results])
    avg_rec = np.mean([r['recall'] for r in results])
    
    print(f"experiment: anomaly_detection_sensor_damage_v1")
    print(f"n:{n_samples} p_damage:{p_damage} avg_precision:{avg_prec:.4f} avg_recall:{avg_rec:.4f}")

if __name__ == "__main__":
    run_experiment()