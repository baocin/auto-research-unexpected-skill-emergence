import numpy as np
import time

def local_kmeans(points, iterations=10, radius=1.0):
    """
    A locally-executed algorithm where each point moves toward 
    the centroid of its neighbors within a certain radius.
    """
    new_points = points.copy()
    for _ in range(iterations):
        updates = []
        for i, p in enumerate(points):
            # Find neighbors within radius
            distances = np.linalg.norm(points - p, axis=1)
            neighbors_mask = distances <= radius
            neighbors = points[neighbors_mask]
            
            if len(neighbors) > 1:
                centroid = np.mean(neighbors, axis=0)
                # Move towards centroid
                new_p = p + 0.5 * (centroid - p)
                updates.append(new_p)
            else:
                updates.append(p)
        new_points = np.array(updates)
        points = new_points.copy()
    return points

def run_experiment():
    # Create clusters of data
    cluster1 = np.random.randn(20, 2) + np.array([0, 0])
    cluster2 = np.random.randn(20, 2) + np.array([5, 5])
    cluster3 = np.random.randn(20, 2) + np.array([0, 5])
    data = np.vstack([cluster1, cluster2, cluster3])
    
    initial_variance = np.var(data)
    
    start_time = time.time()
    # Run the local algorithm with a radius that captures neighbors
    final_points = local_kmeans(data, iterations=15, radius=2.0)
    end_time = time.time()
    
    final_variance = np.var(final_points)
    
    # Measure "success" as the reduction in variance (convergence towards clusters/centroid)
    # In a local k-means sense, we want points to group together.
    # Here we use the ratio of final variance to initial variance.
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0

    print(f"experiment: local_kmeans")
    print(f"initial_variance: {initial_variance:.4f}")
    print(f"final_variance: {final_variance:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()