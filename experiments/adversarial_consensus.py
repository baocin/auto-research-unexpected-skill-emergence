import numpy as np
import time

def run_consensus_experiment(n_agents=100, adversarial_ratio=0.1, iterations=50):
    """
    Investigates consensus formation in a network where a percentage of agents 
    are 'adversarial' (fixed at 0 or 1) and cannot be moved by local updates.
    """
    np.random.seed(42)
    values = np.random.rand(n_agents)
    
    # Identify adversarial agents (fixed at 0 or 1)
    num_adv = int(n_agents * adversarial_ratio)
    all_indices = np.arange(n_agents)
    adv_indices = np.random.choice(all_indices, num_adv, replace=False)
    
    # Set fixed values for adversaries
    fixed_mask = np.zeros(n_agents, dtype=bool)
    for i, idx in enumerate(adv_indices):
        values[idx] = 1.0 if i % 2 == 0 else 0.0
        fixed_mask[idx] = True

    # Create a simple network: each agent connected to neighbors in a ring
    neighbors_map = []
    for i in range(n_agents):
        nb = [(i - 1) % n_agents, (i + 1) % n_agents, (i - 2) % n_agents, (i + 2) % n_agents]
        neighbors_map.append(nb)

    initial_variance = np.var(values)
    
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
    # Success is defined as how much the variance decreased (approaching consensus)
    # Since adversaries prevent zero variance, we look at relative reduction.
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    adversarial_ratios = [0.0, 0.05, 0.1, 0.2, 0.3]
    print(f"experiment: adversarial_consensus")
    
    for ratio in adversarial_ratios:
        start_time = time.time()
        success = run_consensus_experiment(adversarial_ratio=ratio)
        end_time = time.time()
        print(f"adv_ratio_{ratio:.2f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()