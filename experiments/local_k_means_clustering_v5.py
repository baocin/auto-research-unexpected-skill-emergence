import numpy as np
import time

def run_k_means_experiment(n_agents=100, n_clusters=2, iterations=30, noise_level=0.5):
    """
    Investigates 'Local k-means clustering': 
    Can agents using only local distance-based updates converge to cluster centroids?
    Focusing on the effect of noise as a driver for stochastic resonance.
    """
    np.random.seed(42)
    
    # Generate clusters with distinct centers
    centroids = np.array([[-2.0, -2.0], [2.0, 2.0]])[:n_clusters]
    points = []
    for i in range(n_agents):
        c = centroids[i % n_clusters]
        # Inject noise around the centroid
        points.append(c + np.random.normal(0, noise_level, 2))
    current_points = np.array(points)
    initial_variance = np.var(current_points)
    
    for _ in range(iterations):
        new_points = []
        for i in range(n_agents):
            p_i = current_points[i]
            # Local interaction: only look at neighbors within radius 3.0
            dists = np.linalg.norm(current_points - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_agents) != i)
            
            if np.any(neighbors_mask):
                # Move towards the local mean of neighbors
                local_mean = np.mean(current_points[neighbors_mask], axis=0)
                new_p = p_i + 0.5 * (local_mean - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    # Success is defined as the reduction in variance (convergence toward centroids)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: local_k_means_clustering_v5")
    noise_levels = [0.1, 0.5, 1.0, 2.0, 4.0]
    for nl in noise_levels:
        start_time = time.time()
        try:
            success = run_k_means_experiment(noise_level=nl)
            duration = time.time() - start_time
            print(f"noise_{nl:.1f}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{nl}_{e}")

if __name__ == "__main__":
    run_experiment()