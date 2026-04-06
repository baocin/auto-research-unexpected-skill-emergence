import numpy as np
import time

def run_decay_experiment(n_agents=100, decay_rate=0.5, iterations=30):
    """
    Investigates 'Temporal Decay': How the fading of information over time 
    (simulating message loss or aging) affects convergence stability.
    We simulate this by reducing the influence of historical positions in each step.
    """
    np.random.seed(42)
    # Two clusters [0,0] and [5,5]
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(5, 1, (n_agents // 2, 2))
    points = np.vstack([c_array for c_array in [c1, c2]])
    initial_variance = np.var(points)
    current_points = points.copy()
    
    # Memory of previous states (to simulate temporal decay/forgetting)
    memory_buffer = [points.copy()]
    
    for step in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        
        # Decay factor: weight of the 'old' memory decreases over time
        decay_weight = np.exp(-decay_rate * step)
        
        for i in range(n_agents):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_agents) != i)
            
            if np.any(neighbors_mask):
                # Weighted average of current neighbors AND a decayed historical state
                centroid = np.mean(snapshot[neighbors_mask], axis=0)
                historical_state = memory_buffer[0][i] 
                
                # The new position is a blend of current consensus and decaying history
                target = (decay_weight * centroid) + ((1 - decay_weight) * historical_state)
                new_p = p_i + 0.5 * (target - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        
        current_points = np.array(new_points)
        memory_buffer.insert(0, current_points.copy()) # Add new state to memory

    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: k_means_temporal_decay")
    # Test different decay rates (how fast the system 'forgets' its history)
    decay_rates = [0.0, 0.1, 0.5, 2.0]
    for dr in decay_rates:
        start_time = time.time()
        try:
            success = run_decay_experiment(decay_rate=dr)
            duration = time.time() - start_time
            print(f"decay_{dr:.1f}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{dr}_{e}")

if __name__ == "__main__":
    run_experiment()