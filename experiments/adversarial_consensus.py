import numpy as np
import time

def run_experiment():
    """
    Investigates 'Adversarial Consensus': 
    Half the nodes are frozen at value 0, half are frozen at value 1.
    The remaining 'active' nodes attempt to reach a consensus via local averaging.
    We measure if the active nodes can escape the pull of the frozen anchors 
    to find a stable intermediate value or if they are trapped in a tug-of-war.
    """
    num_nodes = 100
    num_active = 40
    num_frozen_zero = 30
    num_frozen_one = 30
    iterations = 50
    
    # Initialize values
    values = np.zeros(num_nodes)
    
    # Set frozen nodes
    # Indices 0 to 29 are frozen at 0
    # Indices 30 to 59 are frozen at 1
    # Indices 60 to 99 are active (start at random [0, 1])
    values[0:30] = 0.0
    values[30:60] = 1.0
    values[60:] = np.random.rand(num_active)
    
    # Define connectivity: Active nodes connect to a random subset of all nodes
    # We'll use a simple ring/local topology for active nodes, but they can see frozen nodes.
    adj = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        if i >= 60: # Only active nodes update
            # Active nodes look at neighbors in a local window (indices around them)
            window = 5
            start = max(0, i - window)
            end = min(num_nodes, i + window + 1)
            for neighbor in range(start, end):
                if i != neighbor:
                    adj[i].append(neighbor)

    initial_std = np.std(values)
    start_time = time.time()
    
    # Simulation loop
    for _ in range(iterations):
        new_values = np.copy(values)
        for i in range(60, num_nodes): # Only update active nodes
            if adj[i]:
                neighbor_indices = adj[i]
                neighbor_values = values[neighbor_indices]
                # Local averaging (consensus step)
                new_values[i] = np.mean(neighbor_values)
        values = new_values

    end_time = time.time()
    final_std = np.std(values)
    
    # Success metric: Can the active nodes achieve a state of low variance 
    # (meaning they have converged to a stable value, even if it's stuck between 0 and 1)?
    # Or does the tension keep them in high-variance flux?
    success_metric = 1.0 - (final_std / initial_std) if initial_std > 0 else 0

    print(f"experiment: adversarial_consensus")
    print(f"initial_std: {initial_std:.4f}")
    print(f"final_std: {final_std:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()