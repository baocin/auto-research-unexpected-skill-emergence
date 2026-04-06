import numpy as np
import time

def run_topology_experiment(n_samples=100, topology='ring', iterations=30):
    """
    Investigates how the underlying connection topology (Ring vs Grid) 
    affects the convergence of local k-means updates.
    """
    np.random.seed(42)
    # Two clusters in a [0, 10] domain
    c1 = np.random.normal(0, 1, (n_samples // 2, 2))
    c2 = np.random.normal(5, 1, (n_samples // 2, 2))
    points = np.vstack([c_points for c_points in [c1, c2]])
    
    initial_variance = np.var(points)
    current_points = points.copy()
    
    # Define topology: how agents 'see' each other
    # In Ring, index i sees (i-1, i+1). In Grid, we use a 2D layout.
    if topology == 'ring':
        neighbors_map = []
        for i in range(n_samples):
            neighbors_map.append([(i - 1) % n_samples, (i + 1) % n_samples])
    else: # Grid-like interaction based on spatial proximity
        neighbors_map = [[] for _ in range(n_samples)]
        # For 'grid' we use a fixed radius but the agents are organized in a 2D array
        pass 

    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            
            if topology == 'ring':
                # Ring-based connectivity (structural)
                nb_indices = neighbors_map[i]
                neighbor_vals = [snapshot[idx] for idx in nb_indices]
                obs = np.mean(neighbor_vals, axis=0)
            else:
                # Spatial proximity based (standard k-means)
                dists = np.linalg.norm(snapshot - p_i, axis=1)
                neighbors_mask = (dists <= 2.5) & (dists > 0)
                if np.any(neighbors_mask):
                    obs = np.mean(snapshot[neighbors_mask], axis=0)
                else:
                    obs = p_i
            
            new_p = p_i + 0.5 * (obs - p_i)
            new_points.append(new_p)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: topology_impact_on_convergence")
    topologies = ['ring', 'spatial']
    
    for topo in topologies:
        start_time = time.time()
        success = run_topology_experiment(topology=topo)
        end_time = time.time()
        print(f"topology_{topo}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()