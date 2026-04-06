import numpy as np
import time

def run_weighted_kmeans(points, weights, iterations=15, radius=2.5):
    """
    An experiment investigating if weighting points by their local density 
    impros the convergence of k-means in a local regime.
    """
    current_points = points.copy()
    current_weights = weights.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot_points = current_points.copy()
        snapshot_weights = current_weights.copy()
        
        for i in range(len(current_points)):
            p = current_points[i]
            distances = np.linalg.norm(snapshot_points - p, axis=1)
            neighbors_mask = (distances <= radius) & (distances > 0)
            
            if np.any(neighbors_mask):
                neighbor_weights = snapshot_weights[neighbors_mask]
                neighbor_positions = snapshot_points[neighbors_mask]
                
                # Weighted centroid calculation
                weighted_centroid = np.sum(neighbor_positions * neighbor_weights[:, np.newaxis], axis=0) / np.sum(neighbor_weights)
                
                direction = weighted_centroid - p
                new_p = p + 0.5 * direction
                new_points.append(new_p)
            else:
                new_points.append(p)
        
        current_points = np.array(new_points)
        # Update weights based on new local density (simplified)
        for i in range(len(current_points)):
            dists = np.linalg.norm(current_points - current_points[i], axis=1)
            current_weights[i] = np.sum(dists <= radius)

    return current_points

def run_experiment():
    np.random.seed(42)
    # Two clusters
    c1 = np.random.randn(30, 2) + np.array([0, 0])
    c2 = np.random.randn(30, 2) + np.array([5, 5])
    data = np.vstack([c1, c2])
    
    # Initial uniform weights
    weights = np.ones(len(data))
    
    initial_variance = np.var(data)
    
    start_time = time.time()
    final_points = run_weighted_kmeans(data, weights=weights, iterations=20, radius=3.0)
    end_time = time.time()
    
    final_variance = np.var(final_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0

    print(f"experiment: weighted_local_kmeans")
    print(f"initial_variance: {initial_variance:.4f}")
    print(f"final_variance: {final_variance:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()