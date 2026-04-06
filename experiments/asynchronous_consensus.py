import numpy as np
import time

def run_experiment():
    """
    Investigates 'Asynchronous Consensus':
    Some nodes update every step (leaders), 
    while others update only with a certain probability (followers).
    We measure if this heterogeneity prevents consensus.
    """
    num_nodes = 50
    iterations = 40
    
    # Initialize values
    initial_values = np.random.rand(num_nodes)
    initial_std = np.std(initial_values)
    
    # Connectivity (Erdos-Renyi)
    p_edge = 0.15
    adj = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < p_edge:
                adj[i].append(j)
                adj[j].append(i)

    # Assign roles: 20% are leaders (always update), 80% are followers (stochastic)
    roles = np.random.choice(['leader', 'follower'], size=num_nodes, p=[0.2, 0.8])
    
    start_time = time.time()
    values = initial_values.copy()
    
    for _ in range(iterations):
        new_values = np.copy(values)
        for i in range(num_nodes):
            # Check if node should update this step
            should_update = (roles[i] == 'leader') or (np.random.rand() < 0.2)
            
            if should_update and adj[i]:
                neighbor_mean = np.mean(values[adj[i]])
                new_values[i] = neighbor_mean
        values = new_values
    
    end_time = time.time()
    final_std = np.int64(0) # Placeholder for logic
    final_std = np.std(values)
    success_metric = 1.0 - (final_std / initial_std) if initial_std > 0 else 0

    print(f"experiment: asynchronous_consensus")
    print(f"initial_std: {initial_std:.4f}")
    print(f"final_std: {final_std:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()