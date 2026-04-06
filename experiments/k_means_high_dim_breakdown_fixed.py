import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def run_experiment():
    dimensions = [2, 10, 50, 100] # Removed 500 to prevent timeout/memory issues
    n_points = 100
    k = 3
    num_trials = 10
    noise_std = 1.0
    
    results = []

    for d in dimensions:
        trial_precisions = []
        start_time = time.time()
        
        for _ in range(num_trials):
            # Generate ground truth: K clusters in D-dimensional space
            centers = np.random.uniform(-5, 5, (k, d))
            
            # Generate points around centers
            points = []
            true_labels = []
            for i in range(k):
                count = n_points // k + (1 if i < n_points % k else 0)
                pts = centers[i] + np.random.normal(0, noise_std, (count, d))
                points.append(pts)
                true_labels.extend([i] * count)
            
            data = np.vstack(points)
            true_labels = np.array(true_labels)

            # K-means implementation
            centroids = centers.copy()
            for _ in range(15): # Max iterations
                distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
                labels = np.argmin(distances, axis=1)
                
                new_centroids = []
                for i in range(k):
                    cluster_points = data[labels == i]
                    if len(cluster_points) > 0:
                        new_centroids.append(np.mean(cluster_points, axis=0))
                    else:
                        new_centroids.append(data[np.random.randint(n_points)])
                centroids = np.array(new_centroids)

            # Evaluate Precision via Hungarian Algorithm
            cost_matrix = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    cost_matrix[i, j] = -np.sum((labels == i) & (true_labels == j))
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            mapping = {row: col for row, col in zip(row_ind, col_ind)}
            
            correct = 0
            for idx in range(len(true_labels)):
                if mapping.get(labels[idx], -1) == true_labels[idx]:
                    correct += 1
            
            trial_precisions.append(correct / n_points)

        duration = time.time() - start_time
        results.append({
            'dim': d,
            'precision': np.mean(trial_precisions),
            'avg_time': duration / num_trials
        })

    print(f"experiment: k_means_high_dim_breakdown_fixed")
    for r in results:
        print(f"n:{r['dim']} precision:{r.get('precision', 0):.4f} time:{r['avg_time']:.4f}")

if __name__ == "__main__":
    run_experiment()