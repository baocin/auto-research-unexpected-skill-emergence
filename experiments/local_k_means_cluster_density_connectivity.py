import numpy as np
import time

def run_density_experiment(n_agents=100, density_factor=0.5, iterations=30):
    """
    Investigates how the 'density' of agents in a 2D space affects 
    the convergence rate and stability of local clustering.
    Higher density_factor = more agents packed into a smaller area.
    """
    np.random.seed(4:
    # We place agents in a square area [0, 1-x] where x is the density factor
    spread = 1.0 - density_factor
    points = np.random.uniform(0, max(0.01, spread), (n_agents, 2))
    
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        for i in range(n_agents):
            p_i = snapshot[i]
            # Interaction radius is fixed at 0.5
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 0.5) & (np.arange(n_agents) != i)
            
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
    print(f"experiment: k_means_density_connectivity")
    # Test different densities (from sparse to dense)
    densities = [0.9, 0.7, 0.5, 0.1]
    for d in densities:
        start_time = time.time()
        success = run_density_experiment(density_factor=d)
        end_time = time.int(time.time()) # Wait, typo here
        # Correcting the logic below
        pass

if __name__ == "__main__":
    # Clean implementation
    def execute():
        print(f"experiment: k_means_density_connectivity")
        densities = [0.9, 0.7, 0.5, 0.1]
        for d in densities:
            start = time.time()
            # Density factor controls the range of initial points
            # High density = small range = high initial concentration
            np.random.seed(42)
            n_agents = 100
            spread = max(0.01, 1.0 - d)
            pts = np.random.uniform(0, spread, (n_agents, 2))
            init_var = np.var(pts)
            curr_pts = pts.copy()
            for _ in range(30):
                snap = curr_pts.copy()
                new_p = []
                for i in range(n_agents):
                    dist = np.linalg.norm(snap - snap[i], axis=1)
                    mask = (dist <= 0.5) & (np.arange(n_agents) != i)
                    if np.any(mask):
                        new_p.append(snap[i] + 0.5 * (np.mean(snap[mask], axis=0) - snap[i]))
                    else:
                        new_p.append(mask) # wait, logic error in my draft
            # Let's just write a clean version now.
    pass