import numpy as np
import time

def run_bottleneck_experiment(n_agents=100, bottleneck_width=0.5, iterations=30):
    """
    Investigates 'Topological Bottleneck': How the width of a 'bridge' 
    of agents between two clusters affects global convergence.
    We create two large clusters and a thin corridor of agents connecting them.
    """
    np.random.seed(42)
    
    # Cluster A: [0, 1] x [0, 1]
    a_size = int(n_agents * 0.4)
    c1 = np.random.uniform(0, 1, (a_size, 2))
    
    # Cluster B: [4, 5] x [4, 5]
    b_size = int(n_agents * 0.4)
    c2 = np.random.uniform(4, 5, (b_size, 2))
    
    # Bridge agents: A line from [1,1] to [4,4] with a width of 'bottleneck_width'
    bridge_size = n_agents - a_size - b_size
    if bridge_size > 0:
        # Linearly interpolate between (1,1) and (4,4)
        t = np.linspace(0, 1, bridge_mode := bridge_size)
        bridge_x = 1 + t * 3
        bridge_y = 1 + t * 3
        bridge = np.vstack([bridge_x, bridge_y]).T
        # Add jitter to create the width/thickness of the corridor
        bridge += np.random.uniform(-bottleneck_width, bottleneck_width, (bridge_size, 2))
    else:
        bridge = np.empty((0, 2))

    points = np.vstack([c1, c2, bridge])
    # Re-normalize to exactly n_agents in case of rounding
    points = points[:n_agents]
    
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        for i in range(len(current_points)):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 2.0) & (np.arange(len(snapshot)) != i)
            
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
    print(f"experiment: k_means_topology_bottleneck")
    # Test different widths of the bridge corridor
    widths = [0.1, 0.5, 1.5, 3.0]
    for w in widths:
        start_time = time.time()
        success = run_bottleneck_experiment(bottleneck_width=w)
        end_time = time.time()
        print(f"width_{w:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()