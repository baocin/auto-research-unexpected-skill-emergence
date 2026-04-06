import numpy as np
import time

def run_influence_ratio_experiment(n_agents=100, anchor_ratio=0.5, noise_level=0.05, iterations=30):
    """
    Investigates 'Influence Ratio': How the proportion of fixed anchors 
    relative to mobile agents affects the system's ability to reach consensus.
    """
    np.random.seed(42)
    # Initialize values: Half at 0, half at 1
    values = np.concatenate([np.zeros(n_agents // 2), np.ones(n_agents // 2)])
    
    # Define anchors based on the ratio (e.g., if ratio is 0.5, 50% of agents are fixed)
    num_anchors = int(n_agents * anchor_ratio)
    fixed_mask = np.zeros(n_agents, dtype=bool)
    # Anchors are placed at the extreme ends (0 and 1) to create tension
    anchor_indices = np.random.choice(n_agents, num_anchors, replace=False)
    fixed_mask[anchor_indices] = True
    
    # Ensure anchors are fixed at their initial extreme values
    for idx in anchor_indices:
        if values[idx] > 0.5:
            values[idx] = 1.0
        else:
            values[idx] = 0.0

    initial_variance = np.var(values)
    current_values = values.copy()
    
    for _ in range(iterations):
        snapshot = current_values.copy()
        new_values = current_values.copy()
        
        # Calculate global mean of all agents (including anchors)
        global_mean = np.mean(snapshot)
        
        for i in range(n_agents):
            if fixed_mask[i]:
                continue
            
            # Mobile agents move towards the global mean with noise
            noise = np.random.normal(0, noise_level)
            new_values[i] = snapshot[i] + 0.5 * (global_mean - snapshot[i]) + noise
            
        current_values = new_values

    final_variance = np.var(current_values)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: adversarial_consensus_influence_ratio")
    # Test different ratios of fixed anchors to total population
    ratios = [0.1, 0.3, 0.5, 0.8]
    for r in ratios:
        start_time = time.time()
        success = run_influence_ratio_experiment(anchor_ratio=r)
        end_time = time.time()
        print(f"anchor_ratio_{r:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()