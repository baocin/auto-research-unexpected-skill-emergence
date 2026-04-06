import numpy as np
import time

def run_topology_experiment(n_samples=100, topology='random', iterations=30):
    """
    Investigates how the underlying connection topology (Random vs Ring) 
    affects the convergence of local k-means updates.
    """
    np.random.seed(42)
    # Create two clusters: C1 at [0,0], C2 at [5,5]
    c1 = np.random.normal(0, 1, (n_samples // 2, 2))
    c2 = np.random.normal(5, 1, (n_samples // 2, 2))
    points = np.vstack([c_p for c_p in [c1, c2]])
    
    initial_variance = np.var(points)
    current_points = points.copy()
    
    # Pre-compute adjacency based on topology
    if topology == 'ring':
        # Each agent is connected to its neighbors in a ring
        adj_indices = []
        for i in range(n_samples):
            adj_indices.append([(i - 1) % n_samples, (i + 1) % n_samples])
    else:
        # 'random' means purely spatial proximity (standard k-means)
        adj_indices = [None] * n_samples

    for _ in range(iterations):
        new_points = []
        snapshot = current_points.copy()
        for i in range(n_samples):
            p_i = snapshot[i]
            
            if topology == 'ring':
                neighbors = [snapshot[idx] for idx in adj_indices[i]]
                # Include self to ensure stability if no neighbors found
                neighbors = np.vstack([p_i, neighbors]) 
                centroid = np.mean(neighbors, axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
            else:
                # Spatial proximity mode
                dists = np.linalg.norm(snapshot - p_i, axis=1)
                mask = (dists <= 3.0) & (np.arange(n_samples) != i)
                if np.any(mask):
                    centroid = np.mean(snapshot[mask], axis=0)
                    new_p = p_i + 0.5 * (centroid - p_i)
                else:
                    new_p = p_i
            new_points.append(new_p)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: k_means_topology_comparison")
    topologies = ['ring', 'spatial']
    
    for topo in topologies:
        start_time = time.time()
        success = run_topology_experiment(topology=topo)
        end_time = time.time()
        print(f"topology_{topo}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()