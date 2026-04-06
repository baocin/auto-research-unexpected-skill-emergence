import numpy as np
import time

def run_heterogeneous_radius_experiment(n_agents=100, radius_range=(1.0, 5.0), iterations=30):
    """
    Investigates 'Heterogeneous Interaction Radius': How varying the 
    interaction range for different agents affects global convergence.
    This simulates a swarm with heterogeneous sensing capabilities.
    """
    np.random.seed(42)
    # Two clusters [0,0] and [5,5]
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(5, 1, (n_agents // 2, 2))
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    current_points = points.copy()
    
    # Assign a random interaction radius to each agent within the range
    radii = np.random.uniform(radius_range[0], radius_range[1], n_agents)
    
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        for i in range(n_agents):
            p_i = snapshot[i]
            r_i = radii[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            # Each agent uses its own unique sensing radius
            neighbors_mask = (dists <= r_i) & (np.arange(n_agents) != i)
            
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
    print(f"experiment: k_means_heterogeneous_radius")
    # Test different ranges of radii (from small/local to large/global)
    ranges = [(1.0, 2.0), (1.0, 4.0), (1.0, 6.0)]
    for r_min, r_max in ranges:
        start_time = time.time()
        success = run_heterogeneous_radius_experiment(radius_range=(r_min, r_max))
        end_time = time.time()
        print(f"range_{r_min}-{r_max}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()