import numpy as np
import time
import json
from collections import defaultdict

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

ARRAY_SIZE = 50            # Number of nodes in a ring topology
NUM_TRIALS = 30            # Number of trials per damage rate
MAX_STEPS = 200            # Steps for consensus to reach stability
DAMAGE_RATES = [0.0, 0.1, 0.2, 0.4]  # Fraction of frozen nodes
NOISE_RATES = [0.0, 0.05] # Probability of a node flipping its value randomly
EXPERIMENT_NAME = "consensus_robustness"
SEED = 42

# ============================================================================
# ALGORITHMS
# ============================================================================

def local_majority_consensus(initial_states, frozen_indices, noise_rate, max_steps, rng):
    """
    Nodes in a ring topology. Each step, an active node looks at neighbors 
    and adopts the majority value.
    """
    states = list(initial_states)
    n = len(states)
    
    for step in range(max_steps):
        # Pick a random node to act
        idx = rng.integers(0, n)
        
        if idx in frozen_indices:
            continue
            
        # Neighbors in ring topology
        left = (idx - 1) % n
        right = (idx + 1) % n
        
        # Local rule: Majority of {self, left, right}
        votes = [states[idx], states[left], states[right]]
        new_val = 1 if sum(votes) >= 2 else 0
        
        # Apply noise
        if rng.random() < noise_rate:
            new_val = 1 - new_val
            
        states[idx] = new_val
        
        # Check for consensus
        if all(s == states[0] for s in states):
            return states, step + 1, True
            
    return states, max_steps, False

def check_consensus(states):
    """Returns 1 if all nodes have the same value, else 0."""
    return 1 if len(set(states)) == 1 else 0

# ============================================================================
# RUNNER
# ============================================================================

def run_experiment():
    rng = np.random.default_rng(SEED)
    results = {
        "experiment": EXPERIMENT_NAME,
        "array_size": ARRAY_SIZE,
        "trials": NUM_TRIALS,
        "damage_rates": {},
        "noise_rates": {}
    }

    for nr in NOISE_RATES:
        results["noise_rates"][str(nr)] = {}
        for dr in DAMAGE_RATES:
            successes = 0
            steps_list = []
            final_agreement_levels = [] # how close to consensus (fraction of same value)

            num_frozen = int(ARRAY_SIZE * dr)

            for trial in range(NUM_TRIALS):
                # Initial state: random bits
                initial_states = rng.integers(0, 2, size=ARRAY_SIZE).tolist()
                
                # Frozen nodes are fixed to a specific value (adversarial)
                frozen_indices = set(rng.choice(ARRAY_SIZE, size=num_frozen, replace=False))
                # For simplicity, frozen nodes are stuck with their initial values 
                # but they don't change even if neighbors suggest otherwise.
                # Actually, to be adversarial, let's say they are all '1'.
                for idx in frozen_indices:
                    initial_states[idx] = 1

                final_states, steps, success = local_majority_consensus(
                    initial_states, frozen_indices, nr, MAX_STEPS, rng
                )
                
                if success:
                    successes += 1
                steps_list.append(steps)
                
                # Measure how close we got (fraction of nodes with the majority value)
                v0 = final_states.count(0)
                v1 = final_states.count(1)
                agreement = max(v0, v1) / ARRAY_SIZE
                final_agreement_levels.append(agreement)

            results["noise_rates"][str(nr)][str(dr)] = {
                "success_rate": successes / NUM_TRIALS,
                "mean_steps": np.mean(steps_list),
                "mean_agreement": np.mean(final_agreement_levels)
            }

    return results

if __name__ == "__main__":
    start_time = time.time()
    res = run_experiment()
    elapsed = time.time() - start_time
    
    # Print in format compatible with previous logs
    print(f"experiment: {EXPERIMENT_NAME}")
    for nr, drs in res["noise_rates"].items():
        for dr, metrics in drs.items():
            print(f"noise={nr}, damage={dr}: success_rate={metrics['success_rate']:.2f}, "
                  f"mean_steps={metrics['mean_steps']:.0f}, agreement={metrics['mean_agreement']:.4f}")

    with open("consensus_results.json", "w") as f:
        json.dump(res, f)
    print(f"total_seconds: {elapsed:.1f}")