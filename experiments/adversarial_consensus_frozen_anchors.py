import numpy as np
import time

def run_adversarial_consensus(n_agents=100, noise_level=0.05, iterations=40):
    """
    Investigates 'Adversarial Consensus': 
    Half the agents are frozen at value 0, half are frozen at value 1.
    We measure if the mobile agents can reach a stable intermediate consensus.
    """
    np.random.seed(40)
    # Initialize values: First half at 0, second half at 1
    values = np.concatenate([np.zeros(n_agents // 2), np.ones(n_agents // 2)])
    
    # Define which agents are 'frozen' (the anchors)
    # We will make the first 25% frozen at 0 and the last 25% frozen at 1
    # The middle 50% are mobile.
    fixed_mask = np.zeros(n_agents, dtype=bool)
    fixed_mask[:int(n_agents * 0.25)] = True
    fixed_mask[int(n_agents * 0.75):] = True
    
    # Initial variance of the population
    initial_variance = np.var(values)
    current_values = values.copy()
    
    for _ in range(iterations):
        new_values = current_values.copy()
        for i in range(n_agents):
            if fixed_mask[i]:
                continue
            
            # Mobile agents look at the global state (Consensus)
            # We add noise to the perception of the mean
            perceived_mean = np.mean(current_values) + np.random.normal(0, noise_level)
            
            # Update rule: Move towards the perceived mean
            new_values[i] = current_values[i] + 0.5 * (perceived_mean - current_values[i])
            
        current_values = new_values

    # Success metric: How much did the variance reduce?
    # Perfect consensus would lead to a single value (variance -> 0)
    final_variance = np.var(current_values)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: adversarial_consensus_frozen_anchors")
    # Test different noise levels in the consensus process
    noise_levels = [0.0, 0.05, 0.2, 0.5]
    for nl in noise_levels:
        start_time = time.time()
        success = run_adversarial_consensus(noise_level=nl)
        end_time = time.time()
        print(f"noise_{nl:.2f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()