import numpy as np
import time

def run_extreme_outlier_kmeans(points, iterations=20, radius=2.5, outlier_count=1):
    """
    Investigates how a very small number of extreme outliers 
    affect the stability of local k-means clustering.
    """
    current_points = points.copy()
    
    # Inject extreme outliers far away from the main distribution
    for _ in range(outlier_count):
        new_outlier = np.random.uniform(20, 30, size=2)
        current_points = np.vstack([current_points, new_outlier])
    
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
    
    # Test different numbers of extreme outliers
    outlier_counts = [1, 3, 10]
    
    print(f"experiment: extreme_outlier_k_means")
    for count in outlier_counts:
        start_time = time.time()
        final_points = run_extreme_outlier_kmeans(data, iterations=20, radius=3.0, outlier_count=count)
        end_time = time.time()
        
        final_variance = np.var(final_points)
        # Success is the reduction in variance relative to initial cluster variance
        success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0

        print(f"outliers_{count}_success_{success_metric:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()