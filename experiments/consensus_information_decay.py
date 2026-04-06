import numpy as np
import time

def run_experiment():
    """
    Investigates 'Consensus with Information Decay':
    Each node's contribution to the average is weighted by a decay factor 
    based on how long ago that information was updated.
    We measure if this prevents long-term consensus.
    """
    num_nodes = 50
    iterations = 40
    # Decay rates: 0 (no decay/standard) to 0.9 (extreme decay/forgetting)
    decay_rates = [0.0, 0.3, 0.7]
    
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

    results = []

    for decay in decay_rates:
        values = initial_values.copy()
        # Track 'age' of information for each node
        ages = np.zeros(num_nodes)
        start_time = time.time()
        
        for _ in range(iterations):
            new_values = np.copy(values)
            for i in range(num_nodes):
                if adj[i]:
                    # Weight neighbors by their 'age' (newer is better)
                    weights = np.exp(-decay * ages[adj[i]])
                    neighbor_vals = values[adj[i]]
                    
                    # Weighted average
                    new_values[i] = np.sum(neighbor_vals * weights) / np.sum(weights)
                
            values = new_values
            ages += 1.0 # Increment age of all information

        end_time = time.time()
        final_std = np.std(values)
        success_metric = 1.0 - (final_std / initial_std) if initial_std > 0 else 0
        results.append((decay, success_metric, end_time - start_time))

    print(f"experiment: consensus_information_decay")
    for decay, success, duration in results:
        print(f"decay_{decay:.1f}_success_{success:.4f}_time_{duration:.4f}")

if __name__ == "__main__":
    run_experiment()