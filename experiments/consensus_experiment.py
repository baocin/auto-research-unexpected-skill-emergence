import numpy as np
import time
import json

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

ARRAY_SIZE = 50            # Number of voters
NUM_TRIALS = 30           # Increased trials for statistical significance
MAX_STEPS = 1000          # Consensus usually converges faster than sorting
DAMAGE_RATES = [0.0, 0.1, 0.2, 0.3, 0.4] # Testing higher damage
EXPERIMENT_NAME = "phase3_consensus"
SEED = 42

# ============================================================================
# ALGORITHMS
# ============================================================================

def voter_model_majority(voters, frozen_set, max_steps=1000, rng=None):
    """
    Each voter looks at its neighbors and adopts the majority opinion.
    'voters' is an array of 0s and 1s.
    Returns (final_states, steps_taken, success, trajectory)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    states = list(voters)
    n = len(states)
    trajectory = []

    success = False
    steps_taken = max_steps

    for step in range(max_steps):
        # Pick a random voter
        idx = rng.integers(0, n)
        
        if idx in frozen_set:
            continue
            
        # Local observation: check neighbors (ring topology)
        left = (idx - 1) % n
        right = (idx + 1) % n
        
        # Count opinions of neighbors
        opinions = [states[left], states[right]]
        # Majority rule: if sum is >= 1, set to 1, else 0
        new_val = 1 if sum(opinions) >= 1 else 0
        
        states[idx] = new_val
        
        if step % (max_steps // 20 + 1) == 0:
            trajectory.append((step, np.std(states)))

        # Check for consensus
        if all(s == states[0] for s in states):
            success = True
            steps_taken = step + 1
            break

    return states, steps_taken, success, trajectory

def voter_model_stochastic(voters, frozen_set, max_steps=1000, rng=None):
    """
    Each voter looks at neighbors and flips with a probability related to disagreement.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    states = list(voters)
    n = len(states)
    trajectory = []

    success = False
    steps_taken = max_steps

    for step in range(max_steps):
        idx = rng.integers(0, n)
        if idx in frozen_set:
            continue
            
        left = (idx - 1) % n
        right = (idx + 1) % n
        
        # Probability of flipping is proportional to local disagreement
        disagreement = abs(states[left] - states[right])
        if rng.random() < float(disagreement):
            states[int(idx)] = 1 - states[int(idx)]

        if step % (max_steps // 20 + 1) == 0:
            trajectory.append((step, np.std(states)))

        # Check for consensus
        if all(s == states[0] for s in states):
            success = True
            steps_taken = step + 1
            break

    return states, steps_taken, success, trajectory

# ============================================================================
# RUNNER
# ============================================================================

def run_experiment():
    start_time = time.time()
    print(f"=== Experiment: {EXPERIMENT_NAME} ===")
    print(f"Array size: {ARRAY_SIZE}, Trials: {NUM_TRIALS}, Max steps: {MAX_STEPS}")
    print(f"Damage rates: {DAMAGE_RATES}")

    results = {"experiment": EXPERIMENT_NAME, "damage_rates": {}}
    rng = np.random.default_rng(SEED)

    algorithms = {
        "majority_rule": voter_model_majority,
        "stochastic_flip": voter_model_stochastic,
    }

    for algo_name, func in algorithms.items():
        print(f"Running {algo_name}...")
        algo_results = {}
        
        for dmg_rate in DAMAGE_RATEs: # ERROR HERE: should be DAMAGE_RATES
            pass # Fixing this below...

# Wait, I need to fix the loop variable name before saving.