import numpy as np
import time

def run_connectivity_threshold_experiment(n_agents=100, radius=3.0, iterations=30):
    """
    Investigates the 'Connectivity Threshold': How varying the interaction 
    radius (R) affects the ability of two distant clusters to merge.
    """
    np.random.seed(42)
    # Two initial clusters centered at [0,0] and [5,5]
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(5, 1, (n_agents // 2, 2))
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        for i in range(n_agents):
            p_i = snapshot[i]
            # Use the experimental radius R
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= radius) & (np.arange(n_agents) != i)
            
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
    print(f"experiment: k_means_connectivity_threshold")
    # Test different interaction radii R
    # We know from previous experiments that R=3.0 is a critical boundary for clusters at distance 5
    radii = [1.0, 2.0, 3.0, 4.0, 6.0]
    for r in radii:
        start_time = time.time()
        success = run_connectivity_threshold_experiment(radius=r)
        end_time = time.time()
        print(f"radius_{r:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()