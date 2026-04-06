import numpy as np
import time

def run_experiment():
    """
    Investigates the impact of 'sensor damage' (randomly zeroing out features)
    on the effectiveness of a distance-based anomaly detector.
    We measure how much damage is required to collapse the Precision/Recall.
    """
    n_samples = 1000
    d = 50  # High dimensionality to allow for significant feature loss
    anomaly_rate = 0.05
    n_anomalies = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anomalies
    
    damage_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
    num_trials = 10
    results = []

    for damage in damage_rates:
        p_scores = []
        r_scores = []
        t_scores = []
        
        for _ in range(num_trials):
            # Generate clusters
            # Normal cluster at origin
            data_normal = np.random.normal(0, 1, (n_normal, d))
            # Anomaly cluster shifted by 5 units on the first axis
            data_anomalies = np.random.normal(0, 1, (n_anomalies, d))
            data_anomalies[:, 0] += 5.0
            
            data = np.vstack([data_normal, data_anomalies])
            labels = np.array([0] * n_normal + [1] * n_anomalies)

            # Apply sensor damage (zero out features)
            if damage > 0:
                mask = np.random.rand(n_samples, d) < damage
                data[mask] = 0.0

            s_time = time.time()
            
            # Detection Algorithm: Distance from origin thresholding
            # We use the top 5% of distances as our anomaly candidates
            distances = np.linalg.norm(data, axis=1)
            threshold = np.percentile(distances, 95)
            predictions = (distances > threshold).astype(int)
            
            t_scores.append(time.time() - s_time)

            # Evaluation: Precision and Recall
            tp = np.sum((predictions == 1) & (labels == 1))
            fp = np.sum((predictions == 1) & (labels == 0))
            fn = np.sum((predictions == 0) & (labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            p_scores.append(precision)
            r_scores.append(recall)

        results.append({
            'damage': damage,
            'avg_p': np.mean(p_scores),
            'avg_r': np.mean(r_scores),
            'avg_t': np.mean(t_scores)
        })

    print("experiment: anomaly_detection_sensor_damage_v2")
    for r in results:
        print(f"damage:{r['damage']:.1f} precision:{r['avg_p']:.4f} recall:{r['avg_r']:.4f} time:{r['avg_t']:.6f}")

if __name__ == "__main__":
    run_experiment()