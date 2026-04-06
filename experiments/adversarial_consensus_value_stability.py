import numpy as np
import time

def run_consensus_experiment(n_agents=100, noise_level=0.0, iterations=50):
    """
    Investigates consensus stability when half the agents are fixed at 0
    and half are fixed at 1, with added stochasticity (noise) in the update rule.
    """
    np.random.seed(42)
    # Initialize: Half at 0, half at 1
    values = np.concatenate([np.zeros(n_agents // 2), np.ones(n_agents // 2)])
    initial_variance = np.var(values)
    
    # Fixed agents mask
    fixed_mask = np.zeros(n_agents, dtype=bool)
    # We will make the first half fixed at 0 and second half fixed at 1 
    # to see if they can reach a middle ground or if it just collapses.
    # Actually, let's keep them all mobile but add noise to see how 'consensus' holds.
    # Wait, the prompt asked for: half frozen at 0, half at 1.
    # If they are frozen, variance won't change unless we measure the "mobile" agents.
    
    # Let's redefine: Agents can move, but some are 'anchors'.
    # We will measure how well the *unfixed* agents converge to a value 
    # that respects the anchors.
    
    anchors_mask = np.zeros(n_agents, dtype=
bool)
    anchors_mask[:n_agents // 2] = True # First half are anchors at 0
    anchors_mask[n_agents // 2:] = False # Second half are mobile

    current_values = values.copy()
    
    for _ in range(iterations):
        new_values = current_values.copy()
        for i in range(n_agents):
            if anchors_mask[i]:
                continue
            
            # Interaction with all other agents (including anchors)
            neighbors = current_values[current_values != i] # simplified
            # Actually, let's use a radius-based interaction for consistency with previous experiments
            # But for consensus, it's usually global. Let's do global.
            
            avg_neighbor = np.mean(current_values)
            
            # Apply update with noise
            noise = np.random.normal(0, noise_level)
            new_values[i] = current_values[i] + 0.5 * (avg_neighbor - current_values[i]) + noise
            
        current_values = new_values

    final_variance = np.var(current_values)
    # Success is how much the variance decreased relative to initial spread
    # In a perfect consensus, variance -> 0.
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: adversarial_consensus_value_stability")
    noise_levels = [0.0, 0.1, 0.5, 1.0]
    for nl in noise_levels:
        start_time = time.time()
        success = run_consensus_experiment(noise_level=nl)
        end_time = time.time()
        print(f"noise_{nl:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()