import numpy as np
import time

def local_consensus(nodes, edges, iterations=20):
    """
    Each node updates its value to the average of its neighbors.
    """
    values = nodes.copy()
    # Pre-calculate adjacency list for speed
    adj = [[] for _ in range(len(nodes))]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for _ in range(iterations):
        new_values = np.copy(values)
        for i in range(len(nodes)):
            if adj[i]:
                neighbor_values = values[adj[i]]
                new_values[i] = np.mean(neighbor_values)
            else:
                new_values[i] = values[i]
        values = new_values
    return values

def run_experiment():
    num_nodes = 50
    # Create a random graph (Erdos-Renyi style)
    p = 0.1  # probability of edge
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < p:
                edges.append((i, j))
    
    # Initial values are random noise [0, 1]
    initial_values = np.random.rand(num_nodes)
    initial_std = np.std(initial_values)
    
    start_time = time.time()
    final_values = local_consensus(initial_values, edges, iterations=30)
    end_time = time.time()
    
    final_std = np.std(final_values)
    
    # Success is the reduction in standard deviation (convergence to consensus)
    success_metric = 1.0 - (final_std / initial_std) if initial_std > 0 else 0

    print(f"experiment: local_consensus")
    print(f"initial_std: {initial_std:.4f}")
    print(f"final_std: {final_std:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()