import numpy as np
import time

def run_asymmetry_experiment(n_agents=100, imbalance_ratio=0.5, iterations=30):
    """
    Investigates 'Initial Imbalance': How an asymmetric distribution of agents 
    between two clusters affects the convergence stability.
    We vary the number of agents in Cluster A vs Cluster B.
    """
    np.random.seed(42)
    
    # Define cluster sizes based on imbalance ratio (e.g., 0.5 means 50/50 split)
    n_a = int(n_agents * (1 - imbalance_ratio))
    n_b = n_agents - n_a
    
    # Generate clusters at [0,0] and [5,5]
    c1 = np.random.normal(0, 1, (n_a, 2))
    c2 = np.random.normal(5, 1, (n_b, 2))
    points = np.vstack([c1, c2])
    
    initial_variance = np.var(points)
    current_points = points.int_copy() if hasattr(points, 'int_copy') else points.copy()
    
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        for i in range(n_agents):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
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
    print(f"experiment: k_means_asymmetric_initialization")
    # Test different imbalance ratios (from 0.5/0.5 split to 0.1/0.9 split)
    ratios = [0.5, 0.3, 0.1]
    for r in ratios:
        start_time = time.time()
        try:
            success = run_asymmetry_experiment(imbalance_ratio=r)
            duration = time.time() - start_time
            print(f"ratio_{r:.1f}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{r}_{e}")

if __name__ == "__main__":
    run_experiment()