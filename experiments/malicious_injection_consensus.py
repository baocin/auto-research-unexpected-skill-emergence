import numpy as np
import time

def run_experiment():
    """
    Investigates 'Malicious Injection Consensus':
    A single node (the attacker) injects a random value [0, 1] into its neighbors every step.
    We measure if the network can still reach consensus despite this continuous disturbance.
    """
    num_nodes = 50
    iterations = 40
    
    # Initialize values [0, 1]
    initial_values = np.random.rand(num_int_nodes := num_nodes)
    initial_std = np.std(initial_values)
    
    # Connectivity (Erdos-Renyi)
    p_edge = 0.2
    adj = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < p_edge:
                adj[i].append(j)
                adj[j].append(i)

    # The attacker is node 0
    attacker_node = 0
    
    start_time = time.time()
    values = initial_values.copy()
    
    for _ in range(iterations):
        new_values = np.copy(values)
        
        # Step 1: The attacker injects noise into its neighbors
        for neighbor in adj[attacker_node]:
            # We simulate the "injection" by perturbing the neighbor's value towards a random target
            target = np.random.rand()
            values[neighbor] = (0.5 * values[neighbor]) + (0.5 * target)

        # Step 2: Standard consensus step for all nodes
        for i in range(num_nodes):
            if adj[i]:
                neighbor_mean = np.mean(values[adj[i]])
                new_values[i] = neighbor_mean
        values = new_values
    
    end_time = time.time()
    final_std = np.std(values)
    success_metric = 1.0 - (final_std / initial_std) if initial_std > 0 else 0

    print(f"experiment: malicious_injection_consensus")
    print(f"initial_std: {initial_std:.4f}")
    print(f"final_std: {final_std:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()