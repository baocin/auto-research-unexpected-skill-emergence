import numpy as np
import time

def run_grid_sort(grid_size=10, obstacle_density=0.2, max_steps=500):
    """
    Simulates a 2D Bubble Sort on a grid. Agents attempt to swap with 
    neighbors if they are out of order (value-wise). Obstacles cannot be swapped.
    """
    np.random.seed(42)
    # Initialize grid with random values
    grid = np.random.rand(grid_size, grid_size)
    
    # Create obstacles
    obstacles = np.random.rand(grid_size, grid_size) < obstacle_density
    
    steps = 0
    converged = False
    
    for step in range(max_steps):
        swaps = 0
        # Flattened indices for easier iteration
        indices = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        np.random.shuffle(indices)
        
        for r, c in indices:
            if obstacles[r, c]:
                continue
                
            # Check neighbors (Right and Down to avoid double counting/infinite loops)
            neighbors = []
            if r + 1 < grid_size and not obstacles[r+1, c]:
                neighbors.append((r+1, c))
            if c + 1 < grid_size and not obstacles[r, c+1]:
                neighbors.append((s := (r, c+1))) # wait, syntax error check
                # Let's just do standard 4-connectivity neighbors
            
            # Re-doing neighbor logic for clarity
            adj = []
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    if not obstacles[nr, nc]:
                        adj.append((nr, nc))
            
            for nr, nc in adj:
                # If current value is greater than neighbor, swap (descending sort attempt)
                if grid[r, c] < grid[nr, nc]:
                    grid[r, c], grid[nr, nc] = grid[nr, nc], grid[r, c]
                    swaps += 1
        
        steps += 1
        if swaps == 0:
            converged = True
            break
            
    # Success is defined as whether the grid reached a stable (sorted) state
    # In a grid with obstacles, "sorted" means no more valid swaps can occur.
    success_metric = 1.0 if converged else 0.0
    return success_metric, steps

def run_experiment():
    print(f"experiment: grid_sort_obstacle_interference")
    densities = [0.0, 0.2, 0.4]
    for d in densities:
        start_time = time.time()
        success, steps = run_grid_sort(grid_size=10, obstacle_density=d)
        end_time = time.time()
        print(f"density_{d:.1f}_success_{success:.1f}_steps_{steps}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()