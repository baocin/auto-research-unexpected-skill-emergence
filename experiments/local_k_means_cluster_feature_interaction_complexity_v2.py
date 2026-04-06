import numpy as np
import time

def run_complexity_experiment(n_agents=100, interaction_radius=3.0, iterations=30):
    """
    Investigates 'Interaction Complexity': How the number of neighbors 
    per agent (driven by density) affects the convergence stability.
    We vary the density of agents within a fixed bounding box [0, 10].
    """
    np.random.seed(42)
    # Fixed bounding box [0, 10] for all experiments
    box_size = 10.0
    
    # Generate random points in the box
    points = np.random.uniform(0, box_size, (n_agents, 2))
    
    # To create two distinct clusters within this density context:
    # We shift half the points to a second cluster center at [10, 10]
    c1_indices = np.arange(0, n_agents // 2)
    c2_indices = np.arange(n_agents // 2, n_agents)
    points[c1_indices] = np.random.uniform(0, 5, (n_agents // 2, 2))
    points[c2_indices] = np.random.uniform(5, 10, (n_agents // 2, 2))
    
    initial_variance = np.var(points)
    current_points = points.copy()
    
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
    print(f"experiment: k_means_interaction_complexity")
    # We vary the number of agents in the same box to change connectivity density
    agent_counts = [50, 100, 200, 400]
    for n in agent_counts:
        start_time = time.time()
        try:
            success = run_complexity_experiment(n_agents=n)
            duration = time.time() - start_time
            print(f"n_agents_{n}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{n}_{e}")

if __name__ == "__main__":
    run_experiment()