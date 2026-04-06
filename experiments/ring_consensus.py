import numpy as np

def ring_consensus_step(nodes, noise_prob=0.0):
    """
    One step of consensus on a ring topology. 
    Each node looks at its two neighbors and updates its value.
    """
    n = len(nodes)
    new_values = nodes.copy()
    for i in range(n):
        # Neighbors: left and right
        left = nodes[i-1]
        right = nodes[(i+1)%n]
        
        # The node observes its neighbors with potential noise
        obs_left = left if np.random.rand() > noise_prob else 1 - left
        obs_right = right if np.random.rand() > noise_prob else 1 - right
        
        # Update rule: Majority of (self, left, right)
        observations = [nodes[i], obs_left, obs_right]
        new_values[i] = 1 if np.mean(observations) > 0.5 else 0
    return new_values

def run_experiment(n_nodes_list, noise_levels, adversarial_ratio, max_steps=5):
    for n in n_nodes_list:
        for p in noise_levels:
            trials = 20
            agreements = 0
            divergence_count = 0 # Number of nodes that differ from the first node
            
            for _ in range(trials):
                # Setup nodes: fixed adversaries at index 0, others random
                num_adversaries = int(n * adversarial_ratio)
                num_random = n - num_adversaries
                initial_nodes = np.concatenate([
                    np.zeros(num_adversaries),
                    np.random.choice([0, 1], size=num_random)
                ])
                np.random.shuffle(initial_nodes)
                
                current_state = initial_nodes.copy()
                for _ in range(max_steps):
                    current_state = ring_consensus_step(current_state, noise_prob=p)
                
                # Check if all nodes in the ring agree with the first node
                if np.all(current_state == current_state[0]):
                    agreements += 1
                else:
                    divergence_count += 1
            
            print(f"n:{n} p:{p} agreement_rate:{agreements/trials} divergence_rate:{divergence_count/trials}")

if __name__ == "__main__":
    n_nodes_list = [10, 50, 100]
    noise_levels = [0.0, 0.1, 0.3]
    adversarial_ratio = 0.2 # 20% fixed at 0
    print(f"experiment: ring_topology_consensus")
    run_experiment(n_nodes_list, noise_levels, adversarial_ratio)