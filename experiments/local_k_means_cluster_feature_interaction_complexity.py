import numpy as np
import time

def run_complexity_experiment(n_agents=100, interaction_radius=3.0, iterations=30):
    """
    Investigates 'Interaction Complexity': How the number of neighbors 
    per agent (driven by density) affects the convergence stability.
    We vary the density of agents within a fixed bounding box [0, 10].
    """
    np.random.seed(42)
    # Fixed bounding box [0, 10] for all experiments to maintain scale
    points = np.random.uniform(0, 10, (n_agents, 2))
    
    # Two initial clusters within the box
    c1_idx = np.random.choice(n_agents, n_agents // 2, replace=False)
    points[c1_idx] = np.random.normal(0, 1, (n_agents // 2, 2))
    c2_idx = np.random.choice(n_agents, n_agents // 2, replace=False)
    points[c2_idx] = np.random.normal(10, 1, (n_agents // 2, 2))
    
    # Ensure points stay within bounds for the experiment
    points = np.clip(points, 0, 10)
    
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
            # We use a function that takes n_agents as parameter
            def run_with_n(num):
                # Re-implementing to ensure local scope is clean
                np.random.seed(4_2)
                pts_c1 = np.random.normal(0, 1, (num // 2, 2))
                pts_c2 = np.random.normal(10, 1, (num // 2, 2))
                pts = np.vstack([pts_c1, pts_c2])
                iv = np.var(pts)
                curr = pts.copy()
                for _ in range(30):
                    snap = curr.copy()
                    new_p_list = []
                    for i in range(num):
                        d = np.linalg.norm(snap - snap[i], axis=1)
                        m = (d <= 3.0) & (np.arange(num) != i)
                        if np.any(m):
                            new_p_list.append(snap[i] + 0.5 * (np.mean(snap[m], axis=0) - snap[i]))
                        else:
                            new_p_list.append(snap[i])
                    curr = np.array(new_p_list)
                return 1.0 - (np.var(curr)/iv)

            success = run_with_n(n)
            duration = time.time() - start_time
            print(f"n_agents_{n}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{n}_{e}")

if __name__ == "__main__":
    run_experiment()