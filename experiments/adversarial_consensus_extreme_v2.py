import numpy as np
import time

def run_extreme_consensus(n_agents=100, frozen_ratio=0.5, iterations=50):
    """
    Investigates consensus formation in a ring topology where a percentage of agents 
    are 'adversarial' (fixed at 0 or 1) and cannot be moved by local updates.
    The adversarial agents are split equally between the 0 and 1 poles.
    """
    np.random.seed(42)
    # Initial random values [0, 1] for all agents
    values = np.random.rand(n_agents)
    
    # Identify indices of agents to be frozen
    num_frozen = int(n_agents * frozen_ratio)
    all_indices = np.arange(n_agents)
    np.random.shuffle(all_indices)
    frozen_indices = all_indices[:num_frozen]
    
    # Create a mask to prevent updates on frozen agents
    fixed_mask = np.zeros(n_agents, dtype=bool)
    
    # Assign adversarial values: half of the frozen agents go to 0.0, half to 1.0
    for i, idx in enumerate(frozen_indices):
        values[idx] = 1.0 if i % 2 == 0 else 0.0
        fixed_mask[idx] = True

    # Define ring topology: each agent interacts with immediate neighbors (i-1, i+1)
    neighbors_map = []
    for i in range(n_agents):
        nb = [(i - 1) % n_agents, (i + 1) % n_agents]
        neighbors_map.append(nb)

    initial_variance = np.var(values)
    
    # Simulation loop
    for _ in range(iterations):
        new_values = values.copy()
        for i in range(n_agents):
            if not fixed_mask[i]:
                # Update rule: average of neighbors
                nb_indices = neighbors_map[i]
                neighbor_vals = [values[idx] for idx in nb_indices]
                new_values[i] = np.mean(neighbor_vals)
        values = new_values

    final_variance = np.var(values)
    # Success metric: 1 - (relative change in variance). 
    # High success means the system converged toward a single value (low variance).
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: adversarial_consensus_extreme_v2")
    # Test a range of frozen agent densities
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for ratio in ratios:
        start_time = time.time()
        success = run_extreme_consensus(frozen_ratio=ratio)
        end_time = time.time()
        print(f"frozen_ratio_{ratio:.2f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()