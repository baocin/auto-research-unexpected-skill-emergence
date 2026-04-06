import numpy as np
import time

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
    centroids = data[indices]
    
    for _ in range(iterations):
        # Calculate distances to all centroids (N, K)
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        
        # Assignment step
        labels = np.argmin(distances, axis=1)
        
        # Update step: move centroids to the mean of assigned points
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
    Tests the stability of K-means clustering under Gaussian noise 
    applied to the data points. We measure 'Pairwise Accuracy' (PA).
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
                pairwise_accuracies = []
                start_time = time.time()
                
                for _ in range(trials):
                    # Apply Gaussian noise to the data
                    noisy_data = data + np.random.normal(0, p_noise, size=data.shape)
                    labels, _ = local_k_means(noisy_data, k=k)
                    
                    # Calculate Pairwise Accuracy (PA)
                    # How often do two points share the same cluster in both true and predicted?
                    # We use a subset of pairs for computational efficiency (O(N^2) is slow)
                    pairs = np.random.choice(n * (n - 1) // 2, min(1000, n * (n - 1) // 2), replace=False)
                    # To simplify implementation, we'll use a simpler metric: 
                    # The proportion of points whose predicted label matches their true label.
                    # Note: This requires a label mapping since cluster indices are arbitrary.
                    
                    # Correct approach for K-means evaluation without sklearn:
                    # Find the best permutation of labels (Hungarian algorithm is too heavy)
                    # Instead, we use the 'Cluster Purity' or simple overlap.
                    
                    # For this experiment, let's use a simpler metric: 
                    # The fraction of points whose cluster center is closer to their true center.
                    # Actually, let's just check if the prediction matches the ground truth directly 
                    # after finding the best mapping for each trial.
                    
                    from scipy.optimize import linear_sum_assignment
                    
                    # Cost matrix: distance between predicted cluster centers and true clusters
                    # But we don't have centroids here, only labels.
                    # Let's use a simpler metric: The accuracy of the 'true_labels' 
                    # being preserved in the 'labels' via one-to-one mapping.
                    
                    # We calculate how many true_labels match the predicted labels 
                    # under the best possible permutation of k labels.
                    cost_matrix = np.zeros((k, 4)) # k clusters vs 4 true classes
                    for i in range(k):
                        for j in range(4):
                            # Cost is number of points with true label j that were assigned to cluster i
                            mask = (true_labels == j) & (labels == i)
                            cost_matrix[i, j] = -np.sum(mask) 
                    
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    
                    # Now count how many points are correctly labeled under this mapping
                    mapping = {pred: true: for pred, true in zip(row_ind, col_ind)}
                    
                    correct_count = 0
                    for idx in range(n):
                        if true_labels[idx] == mapping.get(labels[idx], -1):
                            correct_count += 1
                    
                    pairwise_accuracies.append(correct_count / n)
                
                duration = time.time() - start_time
                results.append({
                    'n': n, 'k': k, 'noise': p_noise,
                    'accuracy': np.mean(pairwise_accuracies),
                    'avg_time': duration / trials
                })
    return results

if __name__ == "__main__":
    # Small N to keep the O(N^2) mapping calculation fast
    n_points = [40]
    k_values = [2, 4]
    noise_levels = [0.1, 1.0, 3.0]
    
    print(f"experiment: k_means_robustness")
    # We need to import scipy for the assignment in the function
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        print("Error: scipy required for label mapping.")
        exit(1)

    results = run_experiment(n_int_list := n_points, k_values, noise_levels)
    
    for res in results:
        print(f"n:{res['n']} k:{res['k']} noise:{res['noise']} acc:{res['accuracy']:.4f} time:{res['avg_time']:.4f}")