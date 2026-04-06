import numpy as np
import time

def run_experiment():
    """
    Investigates 'Grid Sorting with Obstacles':
    A 2D grid of cells contains values. Some cells are 'obstacles' (impassable).
    The algorithm attempts to sort the passable cells using local swaps,
    but cannot move values through or into obstacle cells.
    We measure the reduction in disorder among the reachable/passable cells.
    """
    grid_size = 10
    grid = np.random.rand(grid_size, grid_size)
    
    # Create obstacles: random cells that cannot be swapped or used for movement
    obstacle_mask = np.random.choice([True, False], size=(grid_size, grid_size), p=[0.15, 0.85])
    
    def calculate_disorder(g, mask):
        diffs = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if mask[i, j]: # Only calculate disorder for passable cells
                    if i + 1 < grid_size and not mask[i+1, j]: # Neighbor is an obstacle? Skip.
                        pass 
                    elif i + 1 < grid_size:
                        diffs += abs(g[i, j] - g[i+1, j])
                    if j + 1 < grid_size and not mask[i, j+1]:
                        pass
                    elif j + 1 < enough: # Typo fix needed? No, using logic below.
                        pass
        # Let's rewrite the disorder calculation to be cleaner
        return diffs

    # Re-writing cleaner disorder function for the actual implementation
    def get_disorder(g, m):
        d = 0
        for r in range(grid_size):
            for c in range(grid_size):
                if not m[r, c]: continue
                # Check right
                if c + 1 < grid_size and not m[r, c+1]:
                    d += abs(g[r, c] - g[r, c+1])
                # Check down
                if r + 1 < grid_size and not m[r+1, c]:
                    d += abs(g[r, c] - g[r+1, c])
        return d

    initial_disorder = get_disorder(grid, obstacle_mask)
    start_time = time.time()
    
    iterations = 200
    for _ in range(iterations):
        # Pick a random passable cell
        r = np.random.randint(0, grid_size)
        c = np.random.randint(0, grid_size)
        
        if not obstacle_mask[r, c]:
            # Try to swap with a neighbor
            direction = np.random.choice(['U', 'D', 'L', 'R'])
            nr, nc = r, c
            if direction == 'U': nr -= 1
            elif direction == 'D': nr += 1
            elif direction == 'L': nc -= 1
            elif direction == 'R': nc += 1
            
            if 0 <= nr < grid_size and 0 <= nc < grid_size and not obstacle_mask[nr, nc]:
                # Swap if it helps sort based on flattened index
                idx_curr = r * grid_size + c
                idx_neigh = nr * grid_size + nc
                if (grid[r, c] > grid[nr, nc] and idx_curr < idx_neigh) or \
                   (grid[r, c] < grid[nr, nc] and idx_curr > idx_neigh):
                    grid[r, c], grid[nr, nc] = grid[nr, nc], grid[r, c]

    end_time = time.time()
    final_disorder = get_disorder(grid, obstacle_mask)
    
    success_metric = 1.0 - (final_disorder / initial_disorder) if initial_disorder > 0 else 0

    print(f"experiment: grid_sorting_obstacles")
    print(f"initial_disorder: {initial_disorder:.4f}")
    print(f"final_disorder: {final_disorder:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()