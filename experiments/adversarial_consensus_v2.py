import numpy as np
import time

def run_experiment():
    """
    Investigates 'Adversarial Consensus' with a dynamic topology.
    Instead of a static small-world graph, we introduce 'edge failure' 
    to see if the consensus can survive random disconnection.
    """
    n_nodes = 100
    num_trials = 5
    iterations = 50
    failure_rates = [0.0, 0.1, 0.3, 0.5] # Probability of an edge being broken
    results = []

    for failure in failure_rates:
        p_avg_list = []
        t_list = []
        
        for _ in range(num_trials):
            # Connectivity: Ring topology + random edges
            adj = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                adj[i, (i + 1) % n_nodes] = 1
                adj[i, (i - 1) % n_nodes] = 1
                target = np.random.randint(0, n_nodes)
                if target != i:
                    adj[i, target] = 1
                    adj[target, i] = 1

            # Apply edge failure (damage to the network structure)
            mask = np.random.rand(n_nodes, n_nodes) < failure
            adj[mask] = 0
            # Ensure symmetry after masking
            adj = (adj + adj.T) / 2
            # Remove self-loops
            np.fill_diagonal(adj, 0)

            # Initialize states
            states = np.random.rand(n_nodes)
            
            # Define Adversarial Anchors: Half nodes fixed at 0 and 1
            num_anchors = n_nodes // 2
            anchor_indices = np.arange(num_anchors)
            is_frozen = np.zeros(n_nodes, dtype=bool)
            is_frozen[anchor_indices] = True
            
            # Set anchor values: first half of anchors to 0, second half to 1
            states[anchor_indices[:num_anchors//2]] = 0.0
            states[anchor_indices[num_anchors//2:]] = 1.0

            s_time = time.time()
            for _ in range(iterations):
                new_states = states.copy()
                # Only update non-frozen nodes
                agent_indices = np.where(~is_frozen)[0]
                for i in agent_indices:
                    neighbors = np.where(adj[i] == 1)[0]
                    if len(neighbors) > 0:
                        new_states[i] = np.mean(states[neighbors])
                states = new_states

            t_elapsed = time.time() - s_time
            
            # Metric: Variance of the final state (lower means better consensus)
            final_variance = np.var(states)
            avg_val = np.mean(states)
            
            p_avg_list.append(avg_val)
            t_list.append(t_elapsed)
            results.append({
                'failure': failure,
                'variance': final_variance,
                'avg': avg_val,
                'time': t_elapsed
            })

        # Calculate mean for this failure rate
        mean_p = np.mean(p_avg_list)
        mean_t = np.mean(t_list)
        
        # We will print the summary later using a different format
        # For now, we append to a global list for processing
        pass

    print("experiment: adversarial_consensus_v2")
    # To make it easy to grep, I'll re-structure the loop output below.
    # (Self-correction: The current structure is fine for the agent)

if __name__ == "__main__":
    # Re-running logic within the function to ensure clean printing
    import sys
    
    # Redefining run_experiment inside main to handle printing properly
    n_nodes = 100
    num_trials = 5
    iterations = 50
    failure_rates = [0.0, 0.1, 0.3, 0.5]
    all_data = []

    for failure in failure_rates:
        p_avgs = []
        t_avgs = []
        for _ in range(num_trials):
            adj = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                adj[i, (i + 1) % n_nodes] = 1
                adj[i, (i - 1) % n_nodes] = 1
                target = np.random.randint(0, n_nodes)
                if target != i:
                    adj[i, target] = 1
                    adj[target, i] = 1

            mask = np.random.rand(n_nodes, n_nodes) < failure
            adj[mask] = 0
            adj = (adj + adj.T) / 2
            np.fill_diagonal(adj, 0)

            states = np.random.rand(n_nodes)
            num_anchors = n_nodes // 2
            anchor_indices = np.arange(num_anchors)
            is_frozen = np.zeros(n_nodes, dtype=bool)
            is_frozen[anchor_indices] = True
            states[anchor_indices[:num_anchors//2]] = 0.0
            states[anchor_indices[num_anchors//2:]] = 1.0
            agent_indices = np.where(~is_frozen)[0]

            s_time = time.time()
            for _ in range(iterations):
                new_states = states.copy()
                for i in agent_indices:
                    neighbors = np.where(adj[i] == 1)[0]
                    if len(neighbors) > 0:
                        new_states[i] = np.mean(states[neighbors])
                states = new_states
            t_elapsed = time.time() - s_time
            p_avgs.append(np.mean(states))
            t_avgs.append(t_elapsed)
        
        all_data.append((failure, np.mean(p_avgs), np.mean(t_avgs)))

    print("experiment: adversarial_consensus_v2")
    for f, p, t in all_data:
        print(f"failure:{f:.1f} avg:{p:.4f} t:{t:.6f}")