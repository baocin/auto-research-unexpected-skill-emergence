import numpy as np
import time

def run_experiment():
    n_samples = 1000
    anomaly_rate = 0.05
    d = 20
    # Instead of noise, we sweep separation to find the breakdown point
    separations = [2.0, 3.0, 4.0, 5.0, 6.0]
    noise_level = 1.0 # Fixed noise
    num_trials = 10
    results = []

    for sep in separations:
        f1_scores = []
        t_scores = []
        
        for _ in range(num_trials):
            n_anomalies = int(n_samples * anomaly_rate)
            n_normal = n_samples - n_anomalies
            
            # Generate data: Normal at 0, Anomalies at [sep, 0, 0...]
            data_normal = np.random.normal(0, 1, (n_normal, d))
            centers_anomaly = np.zeros(d)
            centers_anomaly[0] = sep
            data_anomalies = np.random.normal(centers_anomaly, 1, (n_anomalies, d))
            
            data = np.vstack([data_normal, data_anomalies])
            labels = np.array([0] * n_normal + [1] * n_anomalies)

            # Add fixed noise to simulate 'damaged' cells
            perturbation = np.random.normal(0, noise_level, data.shape)
            data += perturbation

            s_time = time.time()
            
            # Simple Anomaly Detection: Distance from origin threshold
            distances = np.linalg.norm(data, axis=1)
            threshold = np.percentile(distances, 95)
            predictions = (distances > threshold).astype(int)
            
            t_scores.append(time.time() - s_time)
            
            # Evaluation
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)

        results.append({
            'sep': sep,
            'avg_f1': np.mean(f1_scores),
            'avg_time': np.mean(t_scores)
        })

    print("experiment: anomaly_detection_separation_sweep")
    for r in results:
        print(f"sep:{r['sep']} f1:{r['avg_f1']:.4f} t:{r['avg_time']:.6f}")

if __name__ == "__main__":
    run_experiment()