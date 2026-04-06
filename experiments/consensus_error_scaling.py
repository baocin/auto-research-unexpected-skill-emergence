import numpy as np
import time

def run_experiment():
    """
    Investigates 'Consensus Error Scaling':
    As the probability of error on each edge increases, how does 
    the final standard deviation (entropy) scale?
    """
    num_nodes = 50
    iterations = 30
    error_probs = [0.0, 0.1, 0.2, 0.4]
    
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

    results = []

    for ep in error_probs:
        values = initial_values.copy()
        start_time = time.time()
        
        for _ in range(iterations):
            new_values = np.copy(values)
            for i in range(num_nodes):
                if adj[i]:
                    neighbor_vals = []
                    for neighbor in adj[i]:
                        if np.random.rand() < ep:
                            # Error: inject random noise into the edge communication
                            neighbor_vals.append(np.random.rand())
                        else:
                            neighbor_vals.append(values[neighbor])
                    new_int_val = np.mean(neighbor_vals)
                    new_values[i] = new_int_val
            values = new_values
        
        end_time = time.time()
        final_std = np.std(values)
        # Success is the reduction in standard deviation
        success_metric = 1.0 - (final_std / initial_std) if initial_std > 0 else 0
        results.append((ep, success_metric, end_time - start_time))

    print(f"experiment: consensus_error_scaling")
    for ep, success, duration in results:
        print(f"err_{ep:.1f}_success_{success:.4f}_time_{duration:.4f}")

if __name__ == "__main__":
    run_experiment()