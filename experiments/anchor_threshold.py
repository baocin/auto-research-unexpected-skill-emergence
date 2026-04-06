import numpy as np
import time
import json

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

ARRAY_SIZE = 50
NUM_TRIALS = 20
MAX_STEPS = 300
DAMAGE_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # Testing high density of anchors
NOISE_RATE = 0.05
EXPERIMENT_NAME = "anchor_threshold_analysis"
SEED = 42

# ============================================================================
# ALGORITHMS
# ============================================================================

def local_majority_consensus(initial_states, frozen_indices, noise_rate, max_steps, rng):
    states = list(initial_states)
    n = len(states)
    
    for step in range(max_steps):
        idx = rng.integers(0, n)
        if idx in frozen_indices:
            continue
            
        left = (idx - 1) % n
        right = (idx + 1) % n
        
        votes = [states[idx], states[left], states[right]]
        new_val = 1 if sum(votes) >= 2 else 0
        
        if rng.random() < noise_rate:
            new_val = 1 - new_val
            
        states[idx] = new_val
        
        # Check for consensus (we use a slightly relaxed check to see if it's "stable")
        # If no changes happen in a window, we consider it stable.
        # But for simplicity here, we just return the state.

    return states

def run_experiment():
    rng = np.random.default_rng(SEED)
    results = {
        "experiment": EXPERIMENT_NAME,
        "damage_rates": [],
        "agreement_levels": []
    }

    for dr in DAMAGE_RATES:
        trial_agreements = []
        num_frozen = int(ARRAY_SIZE * dr)

        for trial in range(NUM_TRIALS):
            initial_states = rng.integers(0, 2, size=ARRAY_SIZE).tolist()
            frozen_indices = set(rng.choice(ARRAY_SIZE, size=num_frozen, replace=False))
            
            # Force frozen nodes to '1' to act as anchors
            for idx in frozen_indices:
                initial_states[idx] = 1

            final_states = local_majority_consensus(
                initial_states, frozen_indices, NOISE_RATE, MAX_STEPS, rng
            )
            
            v1 = final_states.count(1)
            agreement = v1 / ARRAY_SIZE
            trial_agreements.append(agreement)

        avg_agreement = np.mean(trial_agreements)
        results["damage_rates"].append(dr)
        results["agreement_levels"].append(avg_agreement)
        print(f"damage={dr:.1f}, avg_agreement={avg_agreement:.4f}")

    return results

if __name__ == "__main__":
    start_time = time.time()
    res = run_experiment()
    elapsed = time.time() - start_time
    
    with open("anchor_threshold_results.json", "w") as f:
        json.dump(res, f)
    print(f"total_seconds: {elapsed:.1f}")