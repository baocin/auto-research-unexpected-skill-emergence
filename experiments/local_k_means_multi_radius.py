import numpy as np
import time

def run_multi_radius_kmeans(points, radius):
    """
    Runs local k-means with a fixed radius to observe how 
    the scale of interaction affects convergence.
    """
    current_points = points.copy()
    iterations = 20
    
    for _ in range(iterations):
        new_points = []
        snapshot_points = current_points.copy()
        
        for i in range(len(current_points)):
            p = current_points[i]
            distances = np.linalg.norm(snapshot_points - p, axis=1)
            neighbors_mask = (distances <= radius) & (distances > 0)
            neighbors = snapshot_points[neighbors_mask]
            
            if len(neighbors) > 0:
                centroid = np.mean(neighbors, axis=0)
                direction = centroid - p
                new_p = p + 0.5 * direction
                new_points.append(new_p)
            else:
                new_points.append(p)
        current_points = np.array(new_points)
        
    return current_points

def run_experiment():
    # Setup two distinct clusters with a gap
    np.random.seed(123)
    c1 = np.random.randn(40, 2) + np.array([0, 0])
    c2 = np.random.randn(40, 2) + np.array([6, 6])
    data = np.vstack([c1, c2])
    initial_variance = np.var(data)
    
    # Test a range of radii: from very small (local) to very large (global)
    radii = [0.5, 2.0, 4.0, 8.0]
    
    print(f"experiment: local_k_means_multi_radius")
    for r in radii:
        start_time = time.time()
        final_points = run_multi_radius_kmeans(data, radius=r)
        end_time = time.time()
        
        final_variance = np.var(final_points)
        success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0

        print(f"radius_{r:.1f}_success_{success_metric:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()