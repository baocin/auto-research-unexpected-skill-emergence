import numpy as np
import time

def run_experiment():
    """
    Investigates 'Consensus with Edge Noise':
    Each edge $(u, v)$ has a probability $p$ of transmitting the wrong value.
    We measure if consensus still emerges under high-frequency edge corruption.
    """
    num_nodes = 50
    iterations = 40
    # Probabilities of an edge being 'corrupted' (sending random noise)
    corruption_probs = [0.0, 0.1, 0.3, 0.5]
    
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

    results = []

    for cp in corruption_probs:
        values = initial_values.copy()
        start_time = time.time()
        
        for _ in range(iterations):
            new_values = np.copy(values)
            for i in range(num_nodes):
                if adj[i]:
                    # Collect neighbor values, but some are corrupted by noise
                    neighbor_vals = []
                    for neighbor in adj[i]:
                        if np.random.rand() < cp:
                            # Corrupted edge sends random noise [0, 1]
                            neighbor_vals.append(np.random.rand())
                        else:
                            neighbor_vals.append(values[neighbor])
                    
                    new_values[i] = np.mean(neighbor_vals)
            values = new_values
        
        end_time = time.time()
        final_std = np.std(values)
        success_metric = 1.0 - (final_std / initial_std) if initial_std > 0 else 0
        results.append((cp, success_metric, end_time - start_time))

    print(f"experiment: consensus_edge_noise")
    for cp, success, duration in results:
        print(f"p_corrupt_{cp:.1f}_success_{success:.4f}_time_{duration:.4f}")

if __name__ == "__main__":
    run_experiment()