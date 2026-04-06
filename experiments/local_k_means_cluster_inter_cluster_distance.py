import numpy as np
import time

def run_distance_experiment(initial_separation, n_agents=100, iterations=30):
    """
    Investigates 'Inter-Cluster Distance': How the initial separation 
    between two clusters affects the convergence to a single global centroid.
    """
    np.random.seed(42)
    # Cluster 1 centered at [0, 0]
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    # Cluster 2 centered at [initial_separation, initial_separation]
    c2 = np.append(np.random.normal(0, 1, (n_agents // 2, 2)), 
                   [[initial_separation, initial_separation]], axis=0)
    # Wait, the above logic is flawed for creating a second cluster at distance.
    # Let's redo:
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(initial_separation, 1, (n_agents // 2, 2))
    points = np.vstack([c1, c2])
    
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        for i in range(n_agents):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            # Interaction radius is fixed at 3.0
            neighbors_mask = (dists <= 3.0) & (np.arange(n_agents) != i)
            
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
    print(f"experiment: k_means_inter_cluster_distance")
    # Test different initial separations between cluster centers
    distances = [1.0, 3.0, 5.0, 10.0]
    for d in distances:
        start_time = time.time()
        success = run_distance_experiment(initial_separation=d)
        end_time = time.time()
        print(f"dist_{d:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()