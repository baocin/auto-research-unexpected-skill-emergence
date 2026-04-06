import numpy as np
import time

def run_adaptive_kmeans(points, iterations=15):
    current_points = points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot_points = current_points.copy()
        
        for i in range(len(current_points)):
            p = current_points[i]
            radius = 1.0
            distances = np.linalg.norm(snapshot_points - p, axis=1)
            neighbors_mask = (distances <= radius) & (distances > 0)
            
            # Adaptive step: if too few neighbors, expand search radius
            if np.sum(neighbors_mask) < 3:
                radius = 5.0
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
    np.random.seed(42)
    c1 = np.random.randn(30, 2) + np.array([0, 0])
    c2 = np.random.randn(30, 2) + np.array([5, 5])
    data = np.vstack([c1, c2])
    
    initial_variance = np.var(data)
    
    start_time = time.time()
    final_points = run_adaptive_kmeans(data, iterations=15)
    end_time = time.time()
    
    final_variance = np.var(final_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0

    print(f"experiment: adaptive_radius_kmeans")
    print(f"initial_variance: {initial_variance:.4f}")
    print(f"final_variance: {final_variance:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()