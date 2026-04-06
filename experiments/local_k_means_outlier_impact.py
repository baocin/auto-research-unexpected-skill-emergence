import numpy as np
import time

def run_outlier_kmeans(points, iterations=20, radius=2.5, outlier_magnitude=10.0):
    """
    Investigates how the presence of a single extreme outlier 
    affast the convergence and structural stability of local k-means.
    """
    current_points = points.copy()
    
    # Inject an extreme outlier
    outlier = np.array([outlier_magnitude, outlier_magnitude])
    current_points = np.vstack([current_points, outlier])
    
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
    np.random.seed(42)
    # Base clusters
    c1 = np.random.randn(30, 2) + np.array([0, 0])
    c2 = np.random.randn(30, 2) + np.array([5, 5])
    data = np.vstack([c1, c2])
    initial_variance = np.var(data)
    
    # Test different outlier magnitudes
    magnitudes = [5.0, 10.0, 50.0]
    
    print(f"experiment: outlier_impact_k_means")
    for m in magnitudes:
        start_time = time.time()
        final_points = run_outlier_kmeans(data, iterations=20, radius=3.0, outlier_magnitude=m)
        end_time = time.time()
        
        final_variance = np.var(final_points)
        success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0

        print(f"magnitude_{m:.1f}_success_{success_metric:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()