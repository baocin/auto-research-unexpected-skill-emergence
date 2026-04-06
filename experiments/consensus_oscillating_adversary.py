import numpy as np
import time

def run_experiment():
    """
    Investigates 'Consensus with Oscillating Adversary':
    A single node (node 0) oscillates its value between 0 and 1 every iteration.
    We measure if the rest of the network can reach a stable consensus 
    or if it is dragged into an oscillatory state.
    """
    num_nodes = 50
    iterations = 40
    
    # Initialize values [0, 1]
    initial_values = np.random.rand(num_nodes)
    initial_std = np.int64(0) # Placeholder
    initial_std = np.std(initial_values)
    
    # Connectivity (Erdos-Renyi)
    p_edge = 0.2
    adj = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < p_edge:
                adj[i].append(j)
                adj[j].append(i)

    start_time = time.time()
    values = initial_values.copy()
    
    for it in range(iterations):
        new_values = np.copy(values)
        
        # The adversary: node 0 oscillates between 0 and 1
        values[0] = 0.0 if it % 2 == 0 else 1.0
        
        for i in range(1, num_nodes): # Nodes 1 to N-1 follow consensus
            if adj[i]:
                # Only consider neighbors that are NOT the adversary for a moment 
                # to see if the influence of node 0 is dampened by others.
                neighbor_vals = []
                for neighbor in adj[i]:
                    if neighbor == 0:
                        # The adversary's value is always the oscillating one
                        target = 0.0 if it % 2 == 0 else 1.0
                        neighbor_vals.append(target)
                    else:
                        neighbor_vals.append(values[neighbor])
                
                if neighbor_vals:
                    new_values[i] = np.mean(neighbor_vals)
        
        # Update values (keeping node 0's oscillation separate in the next step)
        values[1:] = new_values[1:]
        values[0] = 0.0 if it % 2 == 0 else 1.0

    end_time = time.time()
    final_std = np.std(values)
    success_metric = 1.0 - (final_std / initial_std) if initial_std > 0 else 0

    print(f"experiment: oscillating_adversary_consensus")
    print(f"initial_std: {initial_std:.4f}")
    print(f"final_std: {final_std:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()