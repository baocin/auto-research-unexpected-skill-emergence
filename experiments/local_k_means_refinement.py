import numpy as np
import time

def run_local_kmeans_refined(points, iterations=15, radius=2.0):
    """
    An improved version of local k-means with a momentum term 
    to see if it helps escape local minima or speeds up convergence.
    """
    current_points = points.copy()
    velocities = np.zeros_like(points)
    momentum = 0.9
    
    for _ in range(iterations):
        new_points = []
        # We must use a fixed set of points for the neighbor calculation during this iteration
        # to avoid using partially updated points within the same step (synchronous update).
        snapshot_points = current_points.copy()
        
        for i in range(len(current_points)):
            p = current_points[i]
            distances = np.linalg.norm(snapshot_points - p, axis=1)
            neighbors_mask = (distances <= radius) & (distances > 0)
            neighbors = snapshot_points[neighbors_mask]
            
            if len(neighbors) > 0:
                centroid = np.mean(neighbors, axis=0)
                direction = centroid - p
                velocities[i] = (momentum * velocities[i]) + (1 - momentum) * direction
                new_p = p + velocities[i]
                new_points.append(new_p)
            else:
                new_points.append(p)
        current_points = np.array(new_points)
        
    return current_points

def run_experiment():
    # Generate two distinct clusters
    np.random.seed(40)
    cluster1 = np.random.randn(30, 2) + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) + np.array([5, 5])
    data = np.vstack([cluster1, cluster2])
    
    initial_variance = np.var(data)
    
    start_time = time.time()
    # Run the refined algorithm
    final_points = run_local_kmeans_refined(data, iterations=20, radius=3.0)
    end_time = time.time()
    
    final_variance = np.var(final_points)
    
    # Success is the reduction in variance (clustering effect)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0

    print(f"experiment: local_k_means_momentum")
    print(f"initial_variance: {initial_variance:.4f}")
    print(f"final_variance: {final_variance:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()