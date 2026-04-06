import numpy as np
import time

def run_entropy_experiment(n_samples=100, entropy_level=0.5, iterations=30):
    """
    Investigates how 'structural entropy' (the randomness of the cluster 
    placement) affects convergence success. We use a mixture of two clusters 
    where one is fixed and the other is distributed with increasing entropy.
    """
    np.random.seed(42)
    # Cluster 1: Stable, dense center
    c1 = np.random.normal(0, 0.5, (n_samples // 2, 2))
    
    # Cluster 2: Entropy-driven distribution
    # High entropy = Uniformly distributed in [0, 10]
    # Low entropy = Concentrated near a point
    scale = 10.0 * (1.0 - entropy_level)
    c2 = np.random.rand(n_samples // 2, 2) * scale
    
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_samples) != i)
            
            if np.any(neighbors_mask):
                centroid = np.mean(snapshot[neighbors_mask], axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: k_means_structural_entropy")
    # Entropy levels from highly ordered to highly disordered
    entropy_levels = [0.0, 0.2, 0.5, 0.8]
    
    for e in entropy_levels:
        start_time = time.time()
        success = run_entropy_experiment(entropy_level=e)
        end_time = time.time()
        print(f"entropy_{e:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()