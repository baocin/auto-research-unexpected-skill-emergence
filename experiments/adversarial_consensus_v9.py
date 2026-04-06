import numpy as np
import time

def run_adversarial_consensus_v9(n_agents=100, frozen_ratio=0.5, iterations=50):
    """
    Investigates 'Adversarial Consensus': 
    A population where a subset of agents is 'frozen' at extreme values (0 and 1),
    testing if the remaining free agents can reach a consensus or if the system
    splits into polarized clusters.
    """
    np.random.seed(42)
    # Initialize agents with random values in [0, 1]
    values = np.random.rand(n_agents)
    
    # Determine indices of frozen agents
    num_frozen = int(n_agents * frozen_ratio)
    indices = np.arange(n_agents)
    np.random.shuffle(indices)
    
    frozen_indices = indices[:num_frozen]
    free_indices = indices[num_frozen:]
    
    # Assign frozen agents to extreme values (half 0, half 1)
    for i, idx in enumerate(frozen_indices):
        values[idx] = 0.0 if i < num_frozen // 2 else 1.0
    
    initial_variance = np.var(values)
    current_values = values.copy()

    for _ in range(iterations):
        new_values = current_values.copy()
        # Local interaction: agents look at neighbors within a value-range window
        for i in free_indices:
            diffs = np.abs(current_values - current_values[i])
            neighbors_mask = diffs < 0.3 
            neighbors = current_values[neighbors_mask]
            
            if len(neighbors) > 0:
                target = np.mean(neighbors)
                new_values[i] = current_values[i] + 0.5 * (target - current_values[i])
        current_values = new_values

    final_variance = np.var(current_values)
    # Success is defined as variance reduction relative to the polarized state
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: adversarial_consensus_v9")
    ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    for r in ratios:
        start_time = time.time()
        try:
            success = run_adversarial_consensus_v9(frozen_ratio=r)
            duration = time.time() - start_time
            print(f"frozen_ratio_{r:.1f}_success_{success:.4f}_time_{duration:.4f}")
        except Exception as e:
            print(f"error_at_{r}_{e}")

if __name__ == "__main__":
    run_experiment()