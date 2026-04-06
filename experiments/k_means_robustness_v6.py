import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def local_k_means(data, k, iterations=10):
    """
    Performs K-means clustering where each point is assigned to 
    the nearest centroid based on a noisy distance metric.
    """
    n = data.shape[0]
    if n < k:
        return np.zeros(n, dtype=int), data
        
    # Initialize centroids randomly from the dataset
    indices = np.random.choice(n, k, replace=False)
    centroids = data[indices].copy()
    
    for _ in range(iterations):
        # Calculate distances (N, K)
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = []
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                # If a cluster is empty, re-initialize it to a random point from the dataset
                new_centroids.append(data[np.random.randint(n)])
        centroids = np.array(new_centroids)
        
    return labels, centroids

def run_experiment(n_points, k_values, noise_levels):
    """
    Tests K-means cluster purity under Gaussian noise and structural uncertainty.
    """
    results = []
    # Ground truth: 4 well-separated clusters in 2D space
    true_centers = [np.array([0, 0]), np.array([10, 0]), np.array([0, 10]), np.array([10, 10])]
    
    for n in n_points:
        # Generate data points around true centers
        data_list = []
        true_labels = []
        for i, center in enumerate(true_centers):
            pts = np.random.normal(loc=center, scale=1.0, size=(n // 4, 2))
            data_list.append(pts)
            true_labels.extend([i] * len(pts))
        
        data = np.vstack(data_list)[:n]
        true_labels = np.array(true_labels)[:n]

        for k in k_values:
            for p_noise in noise_levels:
                trials = 15
                purity_scores = []
                start_time = time.time()
                
                for _ in range(trials):
                    # Apply Gaussian noise
                    noisy_data = data + np.random.normal(0, p_noise, size=data.shape)
                    labels, _ = local_k_means(noisy_data, k=k)
                    
                    # Calculate Purity using Hungarian Algorithm for optimal label mapping
                    cost_matrix = np.zeros((k, 4))
                    for i in range(k):
                        for j in range(4):
                            mask = (labels == i) & (true_labels == j)
                            cost_int = np.sum(mask)
                            cost_matrix[i, j] = -float(cost_int)
                    
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    mapping = {pred: true for pred, true in zip(row_ind, col_ind)}
                    
                    correct_count = 0
                    for idx in range(n):
                        if true_labels[idx] == mapping.get(labels[idx], -1):
                            correct_count += 1
                    purity_scores.append(correct_count / n)
                
                duration = time.time() - start_time
                results.append({
                    'n': n, 'k': k, 'noise': p_noise,
                    'purity': np.mean(purity_scores),
                    'avg_time': duration / trials
                })
    return results

if __name__ == "__main__":
    n_points = [40]
    k_values = [2, 4]
    noise_levels = [0.1, 1.0, 3.0]
    print(f"experiment: k_means_robustness_v6")
    res = run_experiment(n_points, k_values, noise_levels)
    for r in res:
        print(f"n:{r['n']} k:{r['k']} noise:{r['noise']} purity:{r['purity']:.4f} time:{r['avg_time']:.4f}")