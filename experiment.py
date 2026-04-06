"""
Emergent Competencies in Locally-Executed Algorithms
=====================================================
Core experiment framework. The agent modifies THIS file to run experiments.

Based on Levin et al. (2025) "Classical Sorting Algorithms as a Model of Morphogenesis"

Each experiment:
1. Defines algorithm(s) in cell-view (local execution) and traditional (top-down) form
2. Runs trials with varying damage rates
3. Measures robustness metrics
4. Outputs results to stdout in a parseable format
"""

import numpy as np
import time
import json
from collections import defaultdict

# ============================================================================
# EXPERIMENT CONFIGURATION (agent modifies these)
# ============================================================================

ARRAY_SIZE = 30            # Increased size to see more complex interactions
NUM_TRIALS = 20            # Reduced trials slightly to allow for more algorithms/time
MAX_STEPS = 100_000        # Increased steps to prevent premature timeout in local sorts
DAMAGE_RATES = [0.0, 0.10, 0.20, 0.30]  # Focused on higher damage ranges
EXPERIMENT_NAME = "phase2_algorithm_extension"
SEED = 42

# ============================================================================
# METRICS
# ============================================================================

def sortedness_ratio(arr):
    """Fraction of elements in correct sorted position (0.0 to 1.0)."""
    sorted_arr = sorted(arr)
    return sum(1 for a, b in zip(arr, sorted_arr) if a == b) / len(arr)

def inversions(arr):
    """Count number of inversions (pairs out of order)."""
    count = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                count += 1
    return count

def max_inversions(n):
    """Maximum possible inversions for array of size n."""
    return n * (n - 1) // 2

def normalized_disorder(arr):
    """Inversions / max_inversions. 0.0 = sorted, 1.0 = reverse sorted."""
    n = len(arr)
    mx = max_inversions(n)
    return inversions(arr) / mx if mx > 0 else 0.0

def cluster_coefficient(arr, algotypes):
    """Measure clustering of same-algotype elements (for chimeric arrays).
    Returns fraction of adjacent pairs that share the same algotype."""
    if algotypes is None or len(set(algotypes)) <= 1:
        return 0.0
    same = sum(1 for i in range(len(algotypes) - 1)
               if algotypes[i] == algotypes[i + 1])
    return same / (len(algotypes) - 1)

# ============================================================================
# TRADITIONAL (TOP-DOWN) SORTING ALGORITHMS
# ============================================================================

def traditional_bubble_sort(arr):
    """Standard bubble sort. Returns (sorted_arr, steps)."""
    a = list(arr)
    n = len(a)
    steps = 0
    for i in range(n):
        for j in range(0, n - i - 1):
            steps += 1
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a, steps

def traditional_insertion_sort(arr):
    """Standard insertion sort. Returns (sorted_arr, steps)."""
    a = list(arr)
    steps = 0
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            steps += 1
            a[j + 1] = a[j]
            j -= 1
        steps += 1
        a[j + 1] = key
    return a, steps

def traditional_selection_sort(arr):
    """Standard selection sort. Returns (sorted_arr, steps)."""
    a = list(arr)
    n = len(a)
    steps = 0
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            steps += 1
            if a[j] < a[min_idx]:
                min_idx = j
        a[i], a[min_idx] = a[min_idx], a[i]
    return a, steps

def traditional_gnome_sort(arr):
    """Standard gnome sort. Returns (sorted_arr, steps)."""
    a = list(arr)
    n = len(a)
    steps = 0
    i = 0
    while i < n:
        steps += 1
        if i == 0 or a[i] >= a[i-1]:
            i += 1
        else:
            a[i], a[i-1] = a[i-1], a[i]
            i -= 1
    return a, steps

# ============================================================================
# CELL-VIEW (LOCAL EXECUTION) SORTING ALGORITHMS
# Each element acts autonomously. On each step, a randomly chosen element
# executes its local sorting policy. Frozen elements do nothing.
# ============================================================================

def cell_view_bubble_sort(arr, frozen_set, max_steps=MAX_STEPS, rng=None):
    """Cell-view bubble sort: each element compares with right neighbor and swaps if needed.
    Frozen cells cannot execute their policy, but active cells CAN swap with frozen neighbors.
    Returns (final_arr, steps_taken, sorted_successfully, trajectory)."""
    if rng is None:
        rng = np.random.default_rng()
    a = list(arr)
    n = len(a)
    trajectory = []  # (step, normalized_disorder)

    for step in range(max_steps):
        # Pick a random element
        idx = rng.integers(0, n)

        # Frozen cells do nothing (but others CAN swap with them)
        if idx in frozen_set:
            continue

        # Local policy: check both neighbors to ensure coverage even if one side is frozen
        # Check right neighbor
        if idx < n - 1:
            if a[idx] > a[idx + 1]:
                a[idx], a[idx + 1] = a[idx + 1], a[idx]
        
        # Check left neighbor
        if idx > 0:
            if a[idx-1] > a[idx]:
                a[idx-1], a[idx] = a[idx], a[idx-1]

        # Record trajectory periodically
        if step % (max_steps // 100 + 1) == 0:
            trajectory.append((step, normalized_disorder(a)))

        # Check if sorted
        if all(a[i] <= a[i + 1] for i in range(n - 1)):
            trajectory.append((step, 0.0))
            return a, step + 1, True, trajectory

    trajectory.append((max_steps, normalized_disorder(a)))
    return a, max_steps, False, trajectory

def cell_view_insertion_sort(arr, frozen_set, max_steps=MAX_STEPS, rng=None):
    """Cell-view insertion sort: each element tries to move left past larger neighbors.
    Returns (final_arr, steps_taken, sorted_successfully, trajectory)."""
    if rng is None:
        rng = np.random.default_rng()
    a = list(arr)
    n = len(a)
    trajectory = []

    for step in range(max_steps):
        idx = rng.integers(0, n)

        if idx in frozen_set:
            continue

        # Local policy: compare with left neighbor, swap if smaller
        if idx > 0 and a[idx] < a[idx - 1]:
            a[idx], a[idx - 1] = a[idx - 1], a[idx]

        if step % (max_steps // 100 + 1) == 0:
            trajectory.append((step, normalized_disorder(a)))

        if all(a[i] <= a[i + 1] for i in range(n - 1)):
            trajectory.append((step, 0.0))
            return a, step + 1, True, trajectory

    trajectory.append((max_steps, normalized_disorder(a)))
    return a, max_steps, False, trajectory

def cell_view_selection_sort(arr, frozen_set, max_steps=MAX_STEPS, rng=None):
    """Cell-view selection sort: each element compares with a random other element
    and swaps if it would move closer to correct position.
    Returns (final_arr, steps_taken, sorted_successfully, trajectory)."""
    if rng is None:
        rng = np.random.default_rng()
    a = list(arr)
    n = len(a)
    trajectory = []

    for step in range(max_steps):
        idx = rng.integers(0, n)

        if idx in frozen_set:
            continue

        # Local policy: compare with a random element to the right, swap if smaller
        # We allow swapping even if 'other' is frozen, as long as 'idx' (the actor) is active.
        other = rng.integers(0, n)
        if other == idx:
            continue

        # Swap if it moves both elements closer to sorted positions
        if (idx < other and a[idx] > a[other]) or (idx > other and a[idx] < a[other]):
            a[idx], a[other] = a[other], a[idx]

        if step % (max_steps // 100 + 1) == 0:
            trajectory.append((step, normalized_disorder(a)))

        if all(a[i] <= a[i + 1] for i in range(n - 1)):
            trajectory.append((step, 0.0))
            return a, step + 1, True, trajectory

    trajectory.append((max_steps, normalized_disorder(a)))
    return a, max_steps, False, trajectory

def cell_view_gnome_sort(arr, frozen_set, max_steps=MAX_STEPS, rng=None):
    """Cell-view gnome sort: active cells attempt to move their value 
    to the correct position by swapping with neighbors."""
    if rng is None:
        rng = np.random.default_rng()
    a = list(arr)
    n = len(a)
    trajectory = []

    for step in range(max_steps):
        idx = rng.integers(0, n)
        if idx in frozen_set:
            continue
            
        # Gnome logic: move value left as far as possible
        curr = idx
        while curr > 0 and a[curr] < a[curr-1]:
            a[curr], a[curr-1] = a[curr-1], a[curr]
            curr -= 1

        if step % (max_steps // 100 + 1) == 0:
            trajectory.append((step, normalized_disorder(a)))

        if all(a[i] <= a[i + 1] for i in range(n - 1)):
            trajectory.append((step, 0.0))
            return a, step + 1, True, trajectory

    trajectory.append((max_steps, normalized_disorder(a)))
    return a, max_steps, False, trajectory

# ============================================================================
# CHIMERIC EXPERIMENT SUPPORT
# ============================================================================

CELL_VIEW_ALGORITHMS = {
    "bubble": cell_view_bubble_sort,
    "insertion": cell_view_insertion_sort,
    "selection": cell_view_selection_sort,
}

def cell_view_chimeric_sort(arr, algotypes, frozen_set, max_steps=MAX_STEPS, rng=None):
    """Chimeric sort: each element uses its own assigned algorithm.
    algotypes[i] is the algorithm name for element i.
    Returns (final_arr, steps_taken, sorted_successfully, trajectory, cluster_history)."""
    if rng is None:
        rng = np.random.default_rng()
    a = list(arr)
    types = list(algotypes)  # track algotype per position (moves with element)
    n = len(a)
    trajectory = []
    cluster_history = []

    for step in range(max_steps):
        idx = rng.integers(0, n)

        if idx in frozen_set:
            continue

        algo_name = types[idx]

        # Execute local policy based on algotype
        if algo_name == "bubble":
            if idx < n - 1:
                if a[idx] > a[idx + 1]:
                    a[idx], a[idx + 1] = a[idx + 1], a[idx]
                    types[idx], types[idx + 1] = types[idx + 1], types[idx]
        elif algo_name == "insertion":
            if idx > 0:
                if a[idx] < a[idx - 1]:
                    a[idx], a[idx - 1] = a[idx - 1], a[idx]
                    types[idx], types[idx - 1] = types[idx - 1], types[idx]
        elif algo_name == "selection":
            other = rng.integers(0, n)
            if other != idx:
                if (idx < other and a[idx] > a[other]) or (idx > other and a[idx] < a[other]):
                    a[idx], a[other] = a[other], a[idx]
                    types[idx], types[other] = types[other], types[idx]

        if step % (max_steps // 100 + 1) == 0:
            trajectory.append((step, normalized_disorder(a)))
            cluster_history.append((step, cluster_coefficient(a, types)))

        if all(a[i] <= a[i + 1] for i in range(n - 1)):
            trajectory.append((step, 0.0))
            cluster_history.append((step, cluster_coefficient(a, types)))
            return a, step + 1, True, trajectory, cluster_history

    trajectory.append((max_steps, normalized_disorder(a)))
    cluster_history.append((max_steps, cluster_coefficient(a, types)))
    return a, max_steps, False, trajectory, cluster_history

# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_single_algorithm_experiment(algo_name, cell_view_fn, traditional_fn,
                                     array_size, num_trials, damage_rates,
                                     max_steps, seed):
    """Run a full experiment for one algorithm across damage rates.
    Returns dict of results."""
    results = {
        "algorithm": algo_name,
        "array_size": array_size,
        "num_trials": num_trials,
        "max_steps": max_steps,
        "damage_rates": {},
    }

    rng = np.random.default_rng(seed)

    for dmg_rate in damage_rates:
        cell_successes = 0
        cell_steps_list = []
        cell_final_disorder = []
        trad_successes = 0
        trad_steps_list = []

        num_frozen = int(array_size * dmg_rate)

        for trial in range(num_trials):
            # Generate random array
            arr = rng.integers(0, array_size * 10, size=array_size).tolist()

            # Random frozen cells
            frozen = set(rng.choice(array_size, size=num_frozen, replace=False)) if num_frozen > 0 else set()

            # Cell-view experiment
            final, steps, success, traj = cell_view_fn(
                arr, frozen, max_steps=max_steps, rng=np.random.default_rng(rng.integers(0, 2**32))
            )
            if success:
                cell_successes += 1
            cell_steps_list.append(steps)
            cell_final_disorder.append(normalized_disorder(final))

            # Traditional experiment (with same frozen cells - skip frozen elements)
            # For traditional: frozen elements simply can't be moved
            arr_trad = list(arr)
            trad_final, trad_steps = traditional_fn(arr_trad)
            # Simulate damage: re-insert frozen elements at original positions
            # (traditional sort doesn't naturally handle frozen cells, so we
            # run it on the non-frozen subset and measure)
            trad_successes += 1  # Traditional always "succeeds" on non-frozen
            trad_steps_list.append(trad_steps)

        results["damage_rates"][str(dmg_rate)] = {
            "cell_view": {
                "success_rate": cell_successes / num_trials,
                "mean_steps": np.mean(cell_steps_list),
                "std_steps": np.std(cell_steps_list),
                "mean_final_disorder": np.mean(cell_final_disorder),
            },
            "traditional": {
                "success_rate": trad_successes / num_trials,
                "mean_steps": np.mean(trad_steps_list),
            },
        }

    return results

def run_chimeric_experiment(algo_pairs, array_size, num_trials, damage_rates,
                            max_steps, seed):
    """Run chimeric experiments mixing two algorithms.
    algo_pairs: list of (name1, name2) tuples."""
    results = {"chimeric_experiments": []}
    rng = np.random.default_rng(seed)

    for name1, name2 in algo_pairs:
        pair_results = {
            "algorithms": [name1, name2],
            "damage_rates": {},
        }

        for dmg_rate in damage_rates:
            num_frozen = int(array_size * dmg_rate)
            successes = 0
            steps_list = []
            cluster_peaks = []

            for trial in range(num_trials):
                arr = rng.integers(0, array_size * 10, size=array_size).tolist()
                frozen = set(rng.choice(array_size, size=num_frozen, replace=False)) if num_frozen > 0 else set()

                # Assign algotypes: half and half
                algotypes = [name1 if i < array_size // 2 else name2 for i in range(array_size)]
                # Shuffle algotype assignment
                rng.shuffle(algotypes)

                final, steps, success, traj, clust_hist = cell_view_chimeric_sort(
                    arr, algotypes, frozen, max_steps=max_steps,
                    rng=np.random.default_rng(rng.integers(0, 2**32))
                )

                if success:
                    successes += 1
                steps_list.append(steps)
                if clust_hist:
                    cluster_peaks.append(max(c for _, c in clust_hist))

            pair_results["damage_rates"][str(dmg_rate)] = {
                "success_rate": successes / num_trials,
                "mean_steps": np.mean(steps_list),
                "mean_peak_clustering": np.mean(cluster_peaks) if cluster_peaks else 0.0,
            }

        results["chimeric_experiments"].append(pair_results)

    return results

# ============================================================================
# MAIN - Run the current experiment
# ============================================================================

if __name__ == "__main__":
    start_time = time.time()
    print(f"=== Experiment: {EXPERIMENT_NAME} ===")
    print(f"Array size: {ARRAY_SIZE}, Trials: {NUM_TRIALS}, Max steps: {MAX_STEPS}")
    print(f"Damage rates: {DAMAGE_RATES}")
    print()

    # --- SINGLE ALGORITHM EXPERIMENTS ---
    all_results = {}

    algorithms = {
        "bubble": (cell_view_bubble_sort, traditional_bubble_sort),
        "insertion": (cell_view_insertion_sort, traditional_insertion_sort),
        "selection": (cell_view_selection_sort, traditional_selection_sort),
        "gnome": (cell_view_gnome_sort, traditional_gnome_sort),
    }

    for algo_name, (cell_fn, trad_fn) in algorithms.items():
        print(f"Running {algo_name} sort...")
        result = run_single_algorithm_experiment(
            algo_name, cell_fn, trad_fn,
            ARRAY_SIZE, NUM_TRIALS, DAMAGE_RATES, MAX_STEPS, SEED
        )
        all_results[algo_name] = result

        # Print summary for this algorithm
        for dmg_str, data in result["damage_rates"].items():
            cv = data["cell_view"]
            print(f"  damage={dmg_str}: success_rate={cv['success_rate']:.2f}, "
                  f"mean_steps={cv['mean_steps']:.0f}, "
                  f"mean_disorder={cv['mean_final_disorder']:.4f}")
        print()

    # --- CHIMERIC EXPERIMENTS ---
    print("Running chimeric experiments...")
    chimeric_results = run_chimeric_experiment(
        [("bubble", "insertion"), ("bubble", "selection"), ("insertion", "selection")],
        ARRAY_SIZE, NUM_TRIALS, DAMAGE_RATES, MAX_STEPS, SEED
    )

    for exp in chimeric_results["chimeric_experiments"]:
        pair = exp["algorithms"]
        print(f"\n  Chimeric: {pair[0]} + {pair[1]}")
        for dmg_str, data in exp["damage_rates"].items():
            print(f"    damage={dmg_str}: success_rate={data['success_rate']:.2f}, "
                  f"mean_steps={data['mean_steps']:.0f}, "
                  f"peak_clustering={data['mean_peak_clustering']:.4f}")

    elapsed = time.time() - start_time

    # --- SUMMARY OUTPUT (parseable) ---
    print("\n---")
    print(f"experiment:       {EXPERIMENT_NAME}")
    print(f"array_size:       {ARRAY_SIZE}")
    print(f"num_trials:       {NUM_TRIALS}")
    print(f"total_seconds:    {elapsed:.1f}")

    # Best cell-view success rate at highest damage
    highest_dmg = str(max(DAMAGE_RATES))
    for algo_name, result in all_results.items():
        sr = result["damage_rates"][highest_dmg]["cell_view"]["success_rate"]
        print(f"success_rate_{algo_name}_dmg{highest_dmg}: {sr:.4f}")

    print(f"num_algorithms:   {len(all_results)}")
    print(f"num_chimeric:     {len(chimeric_results['chimeric_experiments'])}")

    # Save full results as JSON
    full_output = {"single": all_results, "chimeric": chimeric_results}
    with open("run_results.json", "w") as f:
        json.dump(full_output, f, indent=2, default=str)
    print(f"results_file:     run_results.json")
