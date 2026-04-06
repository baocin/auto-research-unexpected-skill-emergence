import numpy as np
import time

def run_experiment():
    """
    Investigates 'Consensus with Inertia':
    Nodes update via: V_{t+1} = (alpha * V_{t}) + ((1 - alpha) * Mean(Neighbors))
    We measure how the inertia coefficient (alpha) affects convergence speed and success.
    """
    num_nodes = 50
    iterations = 30
    alphas = [0.0, 0.5, 0.9] # No inertia, moderate, high inertia
    
    # Create random graph edges
    p = 0.1
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < p:
                edges.append((i, j))

    adj = [[] for _ in range(num_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    results = []
    initial_values = np.random.rand(num_nodes)
    initial_std = np.inf # Placeholder

    for alpha in alphas:
        values = initial_values.copy()
        start_time = time.time()
        
        for _ in range(iterations):
            new_values = np.copy(values)
            for i in range(num_nodes):
                if adj[i]:
                    neighbor_mean = np.mean(values[adj[i]])
                    # Inertia formula
                    new_values[i] = (alpha * values[i]) + ((1 - alpha) * neighbor_mean)
                else:
                    new_values[i] = values[i]
            values = new_values
        
        end_time = time.time()
        final_std = np.std(values)
        
        # We use the standard deviation of the initial random distribution as baseline
        if alpha == alphas[0]:
            initial_std = np.std(initial_values)
            
        success_metric = 1.0 - (final_std / initial_std) if initial_std != np.inf else 0
        results.append((alpha, success_metric, end_time - start_time))

    print(f"experiment: consensus_with_inertia")
    for alpha, success, duration in results:
        print(f"alpha_{alpha:.1f}_success_{success:.4f}_time_{duration:.4f}")

if __name__ == "__main__":
    run_experiment()