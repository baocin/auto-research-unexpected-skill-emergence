import numpy as np
import time

def run_experiment():
    num_points = 100
    num_anomalies = 5
    radius = 2.0
    
    # 1. Generate normal points (Gaussian)
    data = np.random.randn(num_points, 2)
    
    # 2. Inject anomalies (far away)
    anomaly_indices = []
    for _ in range(num_anomalies):
        idx = np.random.randint(0, num_points)
        data[idx] += np.array([15.0, 15.0]) # Move far away
        anomaly_indices.append(idx)

    start_time = time.time()
    
    # 3. Local computation: Each point looks at neighbors and calculates local Z-score
    detected_anomalies = []
    for i in range(num_points):
        p = data[i]
        distances = np.linalg.norm(data - p, axis=1)
        neighbors_mask = (distances <= radius) & (distances > 0)
        neighbors = data[neighbors_mask]
        
        if len(neighbors) > 0:
            # Local mean and std of neighbors
            local_mean = np.mean(neighbors, axis=0)
            local_std = np.std(neighbors, axis=0) + 1e-6
            
            # Z-score of the current point relative to its neighborhood
            z_scores = np.abs((p - local_mean) / local_std)
            if np.any(z_scores > 3.0): # Threshold for anomaly
                detected_anomalies.append(i)
        else:
            # If no neighbors, it's an isolated point (could be an anomaly)
            pass

    end_time = time.time()
    
    # 4. Evaluate
    true_positives = len(set(detected_anomalies).intersection(set(anomaly_indices)))
    false_positives = len(set(detected_anomalies) - set(anomaly_indices))
    recall = true_positives / num_anomalies if num_anomalies > 0 else 0
    precision = true_positives / len(detected_anomalies) if len(detected_anomalies) > 0 else 1.0
    
    # Success metric: F1 score of anomaly detection
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"experiment: local_anomaly")
    print(f"true_positives: {true_positives}")
    print(f"false_positives: {false_positives}")
    print(f"recall: {recall:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"success_rate: {f1:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()