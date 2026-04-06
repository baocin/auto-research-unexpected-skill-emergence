import numpy as np
import time

def run_experiment():
    """
    Investigates 'Malicious Node Consensus':
    A single node (the attacker) attempts to pull its neighbors towards 1.0,
    while the rest of the network tries to reach a consensus on their initial values.
    We measure if the network can resist the influence of the malicious agent.
    """
    num_nodes = 50
    iterations = 40
    
    # Initialize values [0, 1]
    values = np.random.rand(num_nodes)
    initial_std = np.std(values)
    
    # Connectivity (Erdos-Renyi)
    p_edge = 0.2
    adj = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < p_edge:
                adj[i].append(j)
                adj[j].append(i)

    # The attacker is node 0
    attacker_target = 1.0
    
    start_time = time.time()
    for _ in range(iterations):
        new_values = np.copy(values)
        for i in range(num_nodes):
            if i == 0:
                # The attacker node doesn't follow consensus; it just pushes its value to 1.0
                new_values[i] = attacker_target
                continue
                
            if adj[i]:
                neighbor_mean = np.mean(values[adj[i]])
                
                # If the attacker is a neighbor, they pull the node towards 1.0
                if 0 in adj[i]:
                    # The influence of the attacker is blended with the consensus
                    new_values[i] = (0.5 * neighbor_mean) + (0.5 * attacker_target)
                else:
                    new_values[i] = neighbor_mean
        values = new_values
    
    end_time = time.time()
    final_std = np.std(values)
    success_metric = 1.0 - (final_std / initial_std) if initial_std > 0 else 0

    print(f"experiment: malicious_node_consensus")
    print(f"initial_std: {initial_std:.4f}")
    print(f"final_std: {final_std:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()