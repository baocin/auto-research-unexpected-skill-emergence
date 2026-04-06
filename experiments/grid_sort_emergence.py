import numpy as np
import time

def run_grid_sort(n_side=10, max_steps=1000, swap_prob=0.1):
    """
    Implements a local-only sorting algorithm on a 2D grid.
    Agents only compare themselves to immediate neighbors (up, down, left, right).
    If the neighbor is 'out of order' relative to a global gradient target, they 'swap'.
    This tests if local swaps can achieve global order in a 2D topology.
    """
    grid_size = n_side * n_side
    # Initialize with random values [0, 1]
    grid = np.random.rand(n_side, n_side)
    
    # Target: A smooth gradient from 0 to 1 across the grid
    target_values = np.linspace(0, 1, grid_size).reshape(n_side, n_side)
    initial_error = np.mean(np.abs(grid - target_values))

    steps_taken = 0
    for _ in range(max_steps):
        steps_taken += 1
        # Randomly pick a cell and its neighbor to attempt a swap
        r, c = np.random.randint(0, n_side, size=2)
        dr, dc = np.random.choice([-1, 0, 1], size=2)
        nr, nc = r + dr, c + dc
        
        if 0 <= nr < n_side and 0 <= nc < n_side:
            # Local Rule: Check if the current value violates the global gradient target
            # compared to its neighbor. If it does, swap with probability p.
            val_current = grid[r, c]
            val_neighbor = grid[nr, nc]
            
            # A simplified 'error' metric for the local interaction:
            # Is the value at (r,c) significantly different from its target?
            # We swap if the neighbor is 'closer' to the correct gradient.
            if np.abs(val_current - target_values[r, c]) > np.abs(val_neighbor - target_values[r, c]):
                if np.random.rand() < swap_prob:
                    grid[r, c], grid[nr, nc] = grid[nr, nc], grid[r, c]

    final_error = np.mean(np.abs(grid - target_values))
    success_metric = 1.0 - (final_error / initial_error) if initial_error > 0 else 0
    return success_metric, steps_taken

def run_experiment():
    print(f"experiment: grid_sort_emergence")
    # Test different swap probabilities and grid sizes
    sizes = [10, 15]
    probs = [0.1, 0.5]
    
    for n in sizes:
        for p in probs:
            start_time = time.time()
            success, steps = run_grid_perm(n, p)
            end_time = time.time()
            print(f"size_{n}_prob_{p:.1f}_success_{success:.4f}_steps_{steps}_time_{end_time - start_time:.4f}")

def run_grid_perm(n, p):
    # Wrapper to ensure stability in the loop
    return run_grid_sort(n_side=n, max_steps=2000, swap_prob=p)

if __name__ == "__main__":
    run_experiment()