import numpy as np
import time

def run_collision_jitter_experiment(n_agents=100, jitter_magnitude=0.1, iterations=30):
    """
    Investigates 'Collision Jitter': The effect of stochastic positional noise 
    applied at every step to the agents' coordinates.
    This simulates sensor error in spatial perception.
    """
    np.random.seed(42)
    # Initial positions: Two clusters [0,0] and [5,5]
    c1 = np.random.normal(0, 1, (n_agents // 2, 2))
    c2 = np.random.normal(5, 1, (n_agents // 2, 2))
    points = np.vstack([c_array for c_array in [c1, c2]])
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        # Apply jitter (positional noise) to all agents
        jitter = np.random.normal(0, jitter_magnitude, current_points.shape)
        current_points += jitter
        
        snapshot = current_points.copy()
        new_points = []
        for i in range(n_agents):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            # Interaction radius of 3.0
            neighbors_mask = (dists <= 3.0) & (np.arange(n_agents) != i)
            
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
    print(f"experiment: k_means_collision_jitter")
    # Test different magnitudes of positional jitter
    jitter_levels = [0.0, 0.1, 0.5, 2.0]
    for jl in jitter_levels:
        start_time = time.time()
        success = run_collision_jump_experiment(jitter_magnitude=jl)
        end_time = time.time()
        print(f"jitter_{jl:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

# Wait, I have a naming error in my call: 'run_collision_jump_experiment' 
# should be 'run_collision_jitter_experiment'. Let me fix it.

def run_experiment_fixed():
    print(f"experiment: k_means_collision_jitter")
    jitter_levels = [0.0, 0.1, 0.5, 2.0]
    for jl in jitter_levels:
        start_time = time.time()
        success = run_collision_jitter_experiment(jitter_magnitude=jl)
        end_time = time.time()
        print(f"jitter_{jl:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment_fixed()