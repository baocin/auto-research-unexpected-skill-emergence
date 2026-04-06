import numpy as np
import time

def run_experiment():
    """
    Investigates 'Adversarial Consensus': A network of nodes where half are 
    'frozen' at value 0 and the other half are 'frozen' at value 1.
    We observe how a local averaging algorithm (consensus) attempts to reach 
    a global state despite these fixed adversarial anchors.
    """
    n_nodes = 100
    num_trials = 10
    iterations = 50
    results = []

    # Connectivity: Ring topology with some random long-range edges (Small World)
    adj = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        adj[i, (i + 1) % n_nodes] = 1
        adj[i, (i - 1) % n_nodes] = 1
        # Add random long-range connections to simulate small-world effect
        target = np.random.randint(0, n_nodes)
        if target != i:
            adj[i, target] = 1
            adj[target, i] = 1

    for _ in range(num_trials):
        # Initialize states randomly
        states = np.random.rand(n_nodes)
        
        # Define Adversarial Anchors (Frozen Nodes)
        # Let's make the first half frozen at 0 and the second half frozen at 1
        is_frozen = np.zeros(n_nodes, dtype=bool)
        
        # First half: indices 0 to n/2 - 1 are frozen at 0
        # Second half: indices n/2 to n-1 are frozen at 1
        # Wait, if we want "half frozen at 0 and half at 1", that means ALL nodes are frozen.
        # That's not a consensus problem; that's just a static array.
        # Let's redefine: The first half of the nodes are 'Anchors' (fixed).
        # Half of those anchors are 0, half are 1.
        # The rest of the nodes are 'Agents' (can move).
        
        num_anchors = n_nodes // 2
        num_agents = n_nodes - num_anchors
        
        anchor_indices = np.arange(num_anchors)
        agent_indices = np.arange(num_anchors, n_nodes)
        
        # Assign values to anchors: half 0, half 1
        states[anchor_indices[:num_anchors//2]] = 0.0
        states[anchor_indices[num_anchors//2:]] = 1.0
        is_frozen = np.zeros(n_nodes, dtype=bool)
        is_frozen[anchor_indices] = True
        
        # Agents start random
        states[agent_indices] = np.random.rand(num_agents)

        s_time = time.time()
        for _ in range(iterations):
            new_states = states.copy()
            for i in agent_indices:
                # Local averaging: average of neighbors
                neighbors = np.where(adj[i] == 1)[0]
                if len(neighbors) > 0:
                    new_states[i] = np.mean(states[neighbors])
            states = new_states

        t_elapsed = time.time() - s_time
        
        # Metric: Variance of the final state (lower means better consensus)
        # Also metric: Mean deviation from the global mean of anchors
        final_variance = np.var(states)
        avg_val = np.mean(states)
        
        results.append({
            'variance': final_variance,
            'avg_val': avg_val,
            'time': t_elapsed
        })

    print("experiment: adversarial_consensus_v1")
    for r in results:
        print(f"var:{r['variance']:.6f} avg:{r['avg_val']:.4f} t:{r['time']:.6f}")

if __name__ == "__main__":
    run_experiment()