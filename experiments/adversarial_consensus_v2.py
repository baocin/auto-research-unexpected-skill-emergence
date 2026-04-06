import numpy as np
import time

def run_experiment():
    """
    Investigates 'Adversarial Consensus' with a focus on the 
    influence of 'frozen' nodes on the convergence of active nodes.
    We measure the residual variance in the active node population.
    """
    num_nodes = 100
    # Split: 30 frozen at 0, 30 frozen at 1, 40 active
    num_frozen_zero = 30
    num_frozen_one = 30
    num_active = 40
    iterations = 50
    
    # Initialize values: Active nodes start with random noise [0, 1]
    values = np.zeros(num_nodes)
    values[0:num_frozen_zero] = 0.0
    values[num_frozen_zero : num_frozen_zero + num_frozen_one] = 1.0
    values[num_frozen_zero + num_frozen_one:] = np.random.rand(num_active)
    
    # Define connectivity: Active nodes connect to neighbors in a ring topology,
    # but they can also 'sense' the frozen boundary nodes.
    adj = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        if i >= (num_frozen_zero + num_frozen_one): # Only active nodes update
            # Ring neighbors within a window of 5
            for offset in [-2, -1, 1, 2]:
                neighbor = (i + offset) % num_nodes
                adj[i].append(neighbor)

    initial_std = np.std(values[num_frozen_zero + num_frozen_one:])
    start_time = time.time()
    
    # Simulation loop
    for _ in range(iterations):
        new_values = np.copy(values)
        for i in range(num_frozen_zero + num_frozen_one, num_nodes): # Only update active nodes
            if adj[i]:
                neighbor_indices = adj[i]
                neighbor_values = values[neighbor_indices]
                # Local averaging (consensus step)
                new_values[i] = np.mean(neighbor_values)
        values = new_values

    end_time = time.time()
    
    # Final metric: Standard deviation of the active nodes only. 
    # If they reached consensus, std should be low.
    final_std_active = np.std(values[num_frozen_zero + num_frozen_one:])
    
    # Success is the reduction in standard deviation within the active group
    success_metric = 1.0 - (final_std_active / initial_std) if initial_std > 0 else 0

    print(f"experiment: adversarial_consensus_v2")
    print(f"initial_std_active: {initial_std:.4f}")
    print(f"final_std_active: {final_std_active:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()