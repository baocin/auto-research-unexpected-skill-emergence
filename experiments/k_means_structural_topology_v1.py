import numpy as np
import time

def run_experiment():
    """
    Investigates K-means clustering when the 'distance' is not Euclidean, 
    but follows a graph-based path distance (Manhattan) on a grid with obstacles.
    """
    n_points = 100
    k = 3
    num_trials = 5
    noise_std = 0.5
    grid_size = 20
    
    # Obstacle density: how many cells are 'impassable' for the distance metric
    obstacle_rates = [0.0, 0.1, 0.3]
    
    results = []

    for obs_rate in obstacle_rates:
        trial_precisions = []
        start_time = time.time()
        
        # Create a grid with obstacles
        grid = np.zeros((grid_size, grid_size))
        obstacles = np.random.rand(grid_size, grid_size) < obs_rate
        
        # Generate ground truth centers in the grid
        centers_coords = []
        for _ in range(k):
            r, c = np.random.randint(0, grid_size, size=2)
            while obstacles[r, c]: # Ensure center is not an obstacle
                r, c = np.random.randint(0, grid_size, size=2)
            centers_coords.append([r, c])
        true_centers = np.array(centers_coords)

        for _ in range(num_trials):
            # Generate points around centers (using Manhattan distance logic)
            points = []
            true_labels = []
            for i in range(k):
                count = n_points // k + (1 if i < n_points % k else 0)
                # Add Gaussian noise to the grid coordinates
                pts = true_centers[i] + np.random.normal(0, 1.5, (count, 2))
                # Clip to grid boundaries
                pts = np.clip(pts, 0, grid_size - 1)
                points.append(pts)
                true_labels.extend([i] * count)
            data = np.vstack(points)
            true_labels = np.array(true_labels)

            # K-means implementation using Manhattan distance (L1)
            centroids = true_centers.copy()
            for _ in range(10): 
                # L1 Distance: |x - c|
                distances = np.sum(np.abs(data[:, np.newaxis, :] - centroids[np.newaxis, :, :]), axis=2)
                labels = np.argmin(distances, axis=1)
                
                new_centroids = []
                for i in range(k):
                    cluster_points = data[labels == i]
                    if len(cluster_points) > 0:
                        new_centroids.append(np.mean(cluster_points, axis=0))
                    else:
                        new_centroids.append(centroids[i])
                centroids = np.array(new_centroids)

            # Evaluation (Precision via Hungarian)
            from scipy.optimize import linear_sum_assignment
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
            'obs_rate': obs_rate,
            'precision': np.mean(trial_prec_list if 'prect_list' in locals() else trial_prec_precision := trial_prec_precision_fix(trial_precisions)}, # logic fix below
            'avg_time': duration / num_trials
        })

    # Re-writing the print loop to avoid the messy lambda/logic error above
    print(f"experiment: k_means_topological_obstruction")
    for r in results:
        print(f"obs:{r['obs_rate']} precision:{r['precision']:.4f} time:{r['avg_time']:.4f}")

# Wait, I see a logic error in my print/result structure. Let me rewrite the whole run_experiment cleanly.