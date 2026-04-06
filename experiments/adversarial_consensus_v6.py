import numpy as np
import time

def consensus_protocol(nodes, noise_prob=0.0):
    """
    Simulates a consensus protocol on a complete graph (broadcast).
    Nodes attempt to reach agreement based on the observed majority.
    """
    n = len(nodes)
    received_messages = []
    for i in range(n):
        msg = nodes[i]
        if np.random.rand() < noise_prob:
            msg = 1 - msg
        received_messages.append(msg)
    
    # Decision rule: majority of the received messages
    decision = 1 if np.mean(received_messages) > 0.5 else 0
    return np.full(n, decision)

def run_experiment(n_nodes_list, noise_levels, adversarial_ratio):
    """

    Tests how an adversarial block (fixed at 0) affects the probability 
    of reaching a '1' consensus in a noisy broadcast environment.
    """
    results = []
    for n in n_nodes_list:
        for p in noise_levels:
            trials = 40
            successes_target_1 = 0
            agreements = 0
            start_time = time.time()
            
            for _ in range(trials):
                # Setup: fixed adversaries at 0, others random
                num_adversaries = int(n * adversarial_ratio)
                num_random = n - num_adversaries
                
                initial_nodes = np.concatenate([
                    np.zeros(num_adversaries),
                    np.random.choice([0, 1], size=num_random)
                ])
                np.random.shuffle(initial_nodes)
                
                final_states = consensus_protocol(initial_nodes, noise_prob=p)
                
                if np.all(final_states == final_states[0]):
                    agreements += 1
                
                if np.mean(final_states) == 1.0:
                    successes_target_1 += 1
            
            duration = time.time() - start_time
            results.append({
                'n': n,
                'p': p,
                'agreement_rate': agreements / trials,
                'target_1_rate': successes_target_1 / trials,
                'avg_time': duration / trials
            })
    return results

if __name__ == "__main__":
    n_nodes_list = [10, 50, 100]
    noise_levels = [0.0, 0.2, 0.4, 0.6]
    adversarial_ratio = 0.5
    
    print(f"experiment: adversarial_consensus_v6")
    res = run_experiment(n_nodes_list, noise_levels, adversarial_ratio)
    for r in res:
        print(f"n:{r['n']} p:{r['p']} agreement_rate:{r['agreement_rate']:.2f} target_1_rate:{r['target_1_rate']:.2f} time:{r['avg_time']:.4f}")