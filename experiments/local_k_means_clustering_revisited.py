import numpy as np
import time

def run_k_means_cluster(n_samples=100, radius=2.0, iterations=30):
    """
    Tests the ability of local updates to form distinct clusters from 
    a random distribution using a fixed interaction radius.
    """
    np.random.seed(42)
    # Generate random points in [0, 10] x [0, 10]
    points = np.random.rand(n_samples, 2) * 10
    initial_variance = np.var(points)
    
    current_points = points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            # Find neighbors within radius
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= radius) & (dists > 0)
            
            if np.any(neighbors_mask):
                # Move towards the centroid of local neighbors
                centroid = np.mean(snapshot[neighbors_mask], axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    # Success is the reduction of variance (bringing points together into clusters)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: local_k_means_revisited")
    radii = [0.5, 2.0, 5.0]
    n_samples = 150
    
    for r in radii:
        start_time = time.time()
        success = run_k_means_cluster(n_samples=n_samples, radius=r)
        end_time = time.time()
        print(f"radius_{r:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()