import numpy as np
import time

def run_scaling_experiment(n_agents=100, scale_factor=1.0, iterations=30):
    """
    Investigates 'Dynamic Scaling': How scaling the entire universe 
    (positions and radius) affects convergence stability.
    If truly scale-invariant, success should be constant.
    """
    np.random.seed(42)
    # Base clusters [0,0] and [5,5]
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(5, 1, (n_agents // 2, 2))
    points = np.vstack([c_array for c_array in [c1, c2]])
    
    # Scale the positions and the interaction radius
    scaled_points = points * scale_factor
    interaction_radius = 3.0 * scale_factor
    
    initial_variance = np.var(scaled_points)
    current_points = scaled_points.copy()
    
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        for i in range(n_agents):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= interaction_radius) & (np.arange(n_agents) != i)
            
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
    print(f"experiment: k_means_dynamic_scaling")
    # Test different scale factors for the universe
    scales = [0.1, 1.0, 10.0, 100.0]
    for s in scales:
        start_time = time.time()
        success = run_scaling_experiment(scale_factor=s)
        duration = time.time() - start_time
        print(f"scale_{s:.1f}_success_{success:.4f}_time_{duration:.4_f}")

if __name__ == "__main__":
    run_experiment()