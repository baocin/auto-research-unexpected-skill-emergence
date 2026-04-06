import numpy as np
import time

def run_consensus_correct(n_agents=100, frozen_ratio=0.5, iterations=50):
    np.random.seed(42)
    values = np.random.rand(n_agents)
    
    num_frozen = int(n_agents * frozen_ratio)
    indices = np.arange(n_agents).copy()
    np.random.shuffle(indices)
    
    frozen_indices = indices[:num_frozen]
    fixed_mask = np.zeros(n_agents, dtype=bool)
    
    # Half of frozen agents at 0, half at 1
    for i, idx in enumerate(frozen_indices):
        values[idx] = 1.0 if i % 2 == 0 else 0.0
        fixed_mask[idx] = True

    neighbors_map = []
    for i in range(n_agents):
        nb = [(i - 1) % n_agents, (i + 1) % n_agents]
        neighbors_map.append(nb)

    initial_variance = np.var(values)
    
    for _ in range(iterations):
        new_values = values.copy()
        for i in range(n_agents):
            if not fixed_mask[i]:
                nb_indices = neighbors_map[i]
                neighbor_vals = [values[idx] for idx in nb_indices]
                new_values[i] = np.mean(neighbor_vals)
        values = new_values

    final_variance = np.var(values)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: adversarial_consensus_extreme")
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    for ratio in ratios:
        start_time = time.time()
        success = run_consensus_correct(frozen_ratio=ratio)
        end_time = time.time()
        print(f"frozen_ratio_{ratio:.2f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()