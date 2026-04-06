import numpy as np
import time

def local_consensus(nodes, edges, iterations=20):
    values = nodes.copy()
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

def run_meta_experiment():
    num_nodes = 40
    probabilities = np.linspace(0.01, 0.5, 10)
    results = []

    for p in probabilities:
        # Create graph
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.rand() < p:
                    edges.append((i, j))
        
        initial_values = np.random.rand(num_nodes)
        initial_std = np.std(initial_values)
        
        # Run algorithm
        final_values = local_consensus(initial_values, edges, iterations=25)
        final_std = np.std(final_values)
        
        success_metric = 1.0 - (final_std / initial_std) if initial_std > 0 else 0
        results.append((p, success_metric))

    print("experiment: meta_connectivity_analysis")
    for p, score in results:
        print(f"prob_{p:.2f}_success_{score:.4f}")
    print(f"total_seconds: {time.time() - start_time if 'start_time' in locals() else 0}")

if __name__ == "__main__":
    import time
    start_time = time.time()
    run_meta_experiment()