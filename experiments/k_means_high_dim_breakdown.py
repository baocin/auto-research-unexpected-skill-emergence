import numpy as np
import time

def run_experiment():
    """
    Investigates the breakdown of Local K-Means clustering as dimensionality 
    increases, causing 'distance concentration' where all points become 
    equally far from centroids (the Curse of Dimensionality).
    """
    dimensions = [2, 10, 50, 100, 500]
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
                pts = centers[i] + np.random.normal(0, noise_std, (n_points // k, d))
                points.append(pts)
                true_labels.extend([i] * (n_points // k))
            
            data = np.vstack(points)
            true_labels = np.array(true_labels)

            # K-means implementation
            centroids = centers.copy()
            for _ in range(15): # Max iterations
                # Compute distances (N, K)
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

            # Evaluate Precision/Recall via Label Mapping (Hungarian Algorithm simplified)
            # We check how many points are assigned to the 'correct' cluster 
            # after optimal permutation of labels.
            from scipy.optimize import linear_sum_assignment
            cost_matrix = np.zeros((k, k))
            for i in range(k):
                for j in range(k):
                    # Cost is negative count of matches
                    cost_matrix[i, j] = -np.sum((labels == i) & (true_labels == j))
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            mapping = {row: col for row, col in zip(row_ind, col_ind)}
            
            correct = 0
            for idx in range(n_points):
                if mapping.get(labels[idx], -1) == true_labels[idx]:
                    correct += 1
            
            trial_precisions.append(correct / n_points)

        duration = time.time() - start_time
        results.append({
            'dim': d,
            'precision': np.mean(trial_precisions),
            'avg_time': duration / num_trials
        })

    print(f"experiment: k_means_high_dim_breakdown")
    for r in results:
        print(f"n:{r['dim']} precision:{r['precision']:.4f} time:{r['avg_time']:.4f}")

if __name__ == "__main__":
    run_experiment()