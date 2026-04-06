import numpy as np
import time

def run_dynamic_separation_experiment(n_samples=100, separation=2.0, iterations=30):
    """
    Investigates how the initial spatial separation between two clusters 
    affects the convergence success of local k-means updates.
    As separation increases, the 'success' (variance reduction) should theoretically 
    decrease because the global variance is higher and the local interaction 
    is less likely to bridge the gap.
    """
    np.random.seed(42)
    # Cluster 1 at origin
    c1 = np.random.normal(0, 1, (n_samples // 2, 2))
    # Cluster 2 shifted by 'separation' along the x-axis
    c2 = np.random.normal(separation, 1, (n_samples // 2, 2))
    points = np.vstack([c1, c2])
    
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 2.5) & (dists > 0)
            
            if np.any(neighbors_mask):
                centroid = np.mean(snapshot[neighbors_mask], axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    # Success is the reduction in global variance relative to initial
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: dynamic_separation_k_means")
    # Test different separation distances between the two clusters
    separations = [1.0, 3.0, 5.0, 10.0, 20.0]
    
    for s in separations:
        start_time = time.time()
        success = run_dynamic_separation_experiment(separation=s)
        end_time = time.time()
        print(f"sep_{s:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()