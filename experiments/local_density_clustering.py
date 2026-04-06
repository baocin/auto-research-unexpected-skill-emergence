import numpy as np
import time

def run_experiment():
    num_points = 100
    radius = 2.5
    iterations = 10
    
    # 1. Generate two overlapping clusters
    c1 = np.random.randn(num_points // 2, 2) + np.array([0, 0])
    c2 = np.random.randn(num_points // 2, 2) + np.array([3, 3])
    data = np.vstack([c1, c2])
    
    initial_var = np.var(data)
    start_time = time.time()
    
    # 2. Iterative Local Density-Based Movement
    current_points = data.copy()
    for _ in range(iterations):
        new_points = []
        densities = []
        
        # First pass: Calculate local densities
        for i in range(len(current_points)):
            p = current_points[i]
            dists = np.linalg.norm(current_points - p, axis=1)
            density = np.sum(dists <= radius)
            densities.append(density)
        
        densities = np.array(densities)
        # Normalize densities to use as weights
        weights = densities / (np.max(densities) + 1e-6)
        
        # Second pass: Move points towards weighted center of neighbors
        for i in range(len(current_points)):
            p = current_points[i]
            dists = np.linalg.norm(current_points - p, axis=1)
            neighbors_mask = (dists <= radius)
            
            neighbor_weights = weights[neighbors_mask]
            neighbor_positions = current_points[neighbors_mask]
            
            if len(neighbor_positions) > 0:
                # Weighted centroid calculation
                weighted_centroid = np.sum(neighbor_positions * neighbor_weights[:, np.newaxis], axis=0) / np.sum(neighbor_weights)
                # Move towards the weighted centroid
                new_p = p + 0.5 * (weighted_centroid - p)
                new_points.append(new_p)
            else:
                new_points.append(p)
        
        current_points = np.array(new_points)

    end_time = time.time()
    final_var = np.var(current_points)
    
    # Success is the reduction in variance (contraction of clusters) 
    # combined with maintaining separation (we'll use variance reduction as a proxy for convergence)
    success_metric = 1.0 - (final_var / initial_var) if initial_var > 0 else 0

    print(f"experiment: local_density_clustering")
    print(f"initial_variance: {initial_var:.4f}")
    print(f"final_variance: {final_var:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()