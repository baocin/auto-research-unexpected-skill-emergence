import numpy as np
import time

def run_grid_sort_entropy(grid_size=10, entropy_rate=0.0, max_steps=500):
    """
    Investigates how 'Entropy Injection' (randomly flipping values) 
    affects the stability of a 2D sorting algorithm.
    """
    np.random.seed(42)
    # Initialize grid with random values
    grid = np.random.rand(grid_size, grid_size)
    
    # Create obstacles (fixed cells that cannot be swapped)
    obstacles = np.random.rand(grid_size, grid_size) < 0.1
    
    steps = 0
    converged = False
    
    for step in range(max_steps):
        # Inject entropy: randomly perturb some cells
        if entropy_rate > 0:
            entropy_mask = np.random.rand(grid_size, grid_size) < entropy_rate
            grid[entropy_mask] = np.random.rand(np.sum(entropy_mask))

        swaps = 0
        # Flattened indices for iteration
        indices = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        np.random.shuffle(indices)
        
        for r, c in indices:
            if obstacles[r, c]:
                continue
                
            # Check 4-connectivity neighbors (Right and Down to prevent infinite loops/double counting)
            neighbors = []
            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    if not obstacles[nr, nc]:
                        neighbors.append((nr, nc))
            
            for nr, nc in neighbors:
                # If current cell value is greater than neighbor, swap (descending sort)
                if grid[r, c] < grid[nr, nc]:
                    grid[r, c], grid[nr, nc] = grid[nr, nc], grid[r, c]
                    swaps += 1
        
        steps += 1
        if swaps == 0:
            converged = True
            break
            
    # Success is defined as reaching a stable state (no more swaps possible)
    success_metric = 1.0 if converged else 0.0
    return success_metric, steps

def run_experiment():
    print(f"experiment: grid_sort_entropy_injection")
    # Test different entropy rates (probability of a cell being randomized per step)
    entropy_rates = [0.0, 0.01, 0.05, 0.1]
    for er in entropy_rates:
        start_time = time.time()
        success, steps = run_grid_sort_entropy(entropy_rate=er)
        end_time = time.time()
        print(f"entropy_{er:.2f}_success_{success:.1f}_steps_{steps}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()