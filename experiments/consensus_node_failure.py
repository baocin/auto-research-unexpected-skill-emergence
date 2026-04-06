import numpy as np
import time

def run_experiment():
    """

    Investigates 'Consensus with Node Failure':
    Nodes are randomly removed (stop updating) during the iterations.
    We measure if the remaining active nodes can still reach consensus.
    """
    num_nodes = 60
    iterations = 40
    failure_probability = 0.1 # Probability of a node failing each step
    
    initial_values = np.random.rand(num_nodes)
    initial_std = np.std(initial_values)
    
    # Connectivity (Erdos-Renyi)
    p_edge = 0.2
    adj = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < p_edge:
                adj[i].append(j)
                adj[j].append(i)

    # Track which nodes are still 'alive' (updating)
    is_alive = np.ones(num_nodes, dtype=bool)
    values = initial_values.copy()
    
    start_time = time.time()
    for _ in range(iterations):
        new_values = np.copy(values)
        # Randomly kill nodes
        deaths = np.random.rand(num_nodes) < failure_probability
        is_alive &= ~deaths
        
        for i in range(num_nodes):
            if is_alive[i] and adj[i]:
                # Only consider neighbors that are also alive
                live_neighbors = [n for n in adj[i] if is_alive[n]]
                if live_neighbors:
                    neighbor_mean = np.mean(values[live_neighbors])
                    new_values[i] = neighbor_mean
        values = new_values
    
    end_time = time.time()
    
    # We measure the std of all nodes, but note that dead nodes are frozen
    final_std = np.std(values)
    success_metric = 1.0 - (final_std / initial_std) if initial_std > 0 else 0

    print(f"experiment: consensus_node_failure")
    print(f"initial_std: {initial_std:.4f}")
    print(f"final_std: {final_std:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()