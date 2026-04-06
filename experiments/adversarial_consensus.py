import numpy as np

def consensus_protocol(nodes, noise_prob=0.0):
    n = len(nodes)
    received_messages = []
    for i in range(n):
        msg = nodes[i]
        if np.random.rand() < noise_prob:
            msg = 1 - msg
        received_messages.append(msg)
    
    majority_val = 1 if np.mean(received_messages) > 0.5 else 0
    return np.full(n, majority_val)

def run_experiment(n_nodes_list, noise_levels, adversarial_ratio):
    for n in n_nodes_list:
        for p in noise_levels:
            trials = 30
            agreements = 0
            target_1_ratios = []
            
            for _ in range(trials):
                num_adversaries = int(n * adversarial_ratio)
                num_random = n - num_adversaries
                initial_nodes = np.concatenate([
                    np.zeros(num_adversaries),
                    np.random.choice([0, 1], size=num_random)
                ])
                np.random.shuffle(initial_nodes)
                
                final_values = consensus_protocol(initial_nodes, noise_prob=p)
                if np.all(final_values == final_values[0]):
                    agreements += 1
                target_1_ratios.append(np.mean(final_values == 1))
            
            print(f"n:{n} p:{p} agreement_rate:{agreements/trials} avg_target_1:{np.mean(target_1_ratios)}")

if __name__ == "__main__":
    n_nodes_list = [10, 50, 100]
    noise_levels = [0.0, 0.1, 0.3]
    adversarial_ratio = 0.3
    print(f"experiment: adversarial_consensus")
    run_experiment(n_nodes_list, noise_levels, adversarial_ratio)