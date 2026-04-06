import numpy as np
import time

def run_cluster_density_ratio_kmeans(n_points, cluster_ratio, radius=2.5, iterations=20):
    """
    Investigates k-means when the ratio of points between two clusters is extreme.
    This tests if a very small 'target' cluster can be swallowed or if it remains 
    distinct under local updates.
    """
    # Create two clusters with different sizes (imbalance)
    # Cluster 1: Large population
    c1_size = int(n_points * (1 - cluster_ratio))
    # Cluster 2: Small population
    c2_size = n_points - c1_size
    
    c1 = np.random.randn(c1_size, 2) + np.array([0, 0])
    c2 = np.random.randn(c2_size, 2) + np.array([6, 6])
    data = np.vstack([c1, c2])
    initial_variance = np.var(data)
    
    current_points = data.copy()
    
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
        
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric, final_variance

def run_experiment():
    # Test different imbalance ratios: from balanced to extreme
    # Ratios represent the proportion of points in the second cluster
    ratios = [0.5, 0.1, 0.05, 0.01]
    n_total = 100
    
    print(f"experiment: k_means_imbalance_sensitivity")
    for ratio in ratios:
        start_time = time.time()
        success, final_var = run_cluster_density_ratio_kmeans(n_total, cluster_ratio=ratio)
        end_time = time.time()
        
        print(f"ratio_{ratio:.2f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()