import numpy as np
import time

def run_consensus(num_nodes, num_edges, iterations=30):
    values = np.random.rand(num_nodes)
    initial_std = np.std(values)
    
    # Create random edges
    edges = []
    all_possible_edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            all_possible_edges.append((i, j))
    
    if len(all_possible_edges) > 0:
        indices = np.random.choice(len(all_possible_edges), min(num_edges, len(all_possible_edges)), replace=False)
        edges = [all_possible_edges[idx] for idx in indices]

    adj = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    start_time = time.time()
    for _ in range(iterations):
        new_values = np.copy(values)
        for i in range(num_nodes):
            if adj[i]:
                neighbor_values = values[adj[i]]
                new_values[i] = np.mean(neighbor_values)
        values = new_values
    end_time = time.time()

    final_std = np.std(values)
    success_metric = 1.0 - (final_std / initial_std) if initial_std > 0 else 0
    return success_metric, end_time - start_time

def run_experiment():
    num_nodes = 50
    # Test edge counts from very sparse to very dense
    edge_counts = [10, 30, 60, 120, 240]
    
    print(f"experiment: sparsity_sensitivity")
    for ec in edge_counts:
        success, duration = run_consensus(num_nodes, ec)
        print(f"edges_{ec}_success_{success:.4f}_time_{duration:.4f}")

if __name__ == "__main__":
    run_experiment()