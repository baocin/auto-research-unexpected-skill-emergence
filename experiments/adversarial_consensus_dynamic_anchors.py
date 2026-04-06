import numpy as np
import time

def run_dynamic_anchor_experiment(n_agents=100, anchor_drift=0.5, noise_level=0.05, iterations=30):
    """
    Investigates 'Dynamic Anchors': The anchors themselves move over time.
    We measure if the mobile agents can track the moving consensus 
    target or if the system falls into divergence.
    """
    np.random.seed(42)
    # Initial values: First half at 0, second half at 1
    values = np.concatenate([np.zeros(n_agents // 2), np.ones(n_agents // 2)])
    
    # Define which agents are 'frozen' (the anchors)
    fixed_mask = np.zeros(n_agents, dtype=bool)
    fixed_mask[:int(n_agents * 0.25)] = True
    fixed_mask[int(n_agents * 0.75):] = True
    
    # The anchors' values will drift over time
    anchor_values = [0.0, 1.0]
    
    initial_variance = np.var(values)
    current_values = values.copy()
    
    for step in range(iterations):
        # Update anchor positions (drift)
        anchor_values[0] += anchor_drift
        anchor_values[1] -= anchor_drift
        
        # Re-apply anchors to the fixed indices
        new_values = current_values.copy()
        for i in range(n_agents):
            if fixed_mask[i]:
                if i < n_agents // 2:
                    new_values[i] = anchor_values[0]
                else:
                    new_values[i] = anchor_values[1]
        
        # Mobile agents update based on the current (moving) population mean
        snapshot = new_values.copy()
        for i in range(n_agents):
            if not fixed_mask[i]:
                perceived_mean = np.mean(snapshot) + np.random.normal(0, noise_level)
                new_values[i] = snapshot[i] + 0.5 * (perceived_mean - snapshot[i])
        
        current_values = new_values

    final_variance = np.var(current_values)
    # Success is defined as maintaining a low variance despite the drift
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: adversarial_consensus_dynamic_anchors")
    # Test different drift rates
    drift_rates = [0.01, 0.05, 0.1, 0.2]
    for dr in drift_rates:
        start_time = time.time()
        success = run_dynamic_anchor_experiment(anchor_drift=dr)
        end_time = time.time()
        print(f"drift_{dr:.2f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()