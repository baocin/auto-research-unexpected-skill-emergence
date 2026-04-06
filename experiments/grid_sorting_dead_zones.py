import numpy as np
import time

def run_experiment():
    """
    Investigates 'Grid Sorting with Dead Zones':
    Large, contiguous blocks of the grid are made impassable (Dead Zones).
    We measure if local swaps can still achieve any reduction in disorder 
    in a heavily fragmented landscape.
    """
    grid_size = 12
    num_iterations = 400
    
    grid = np.random.rand(grid_size, grid_size)
    
    # Create one large 'Dead Zone' (e.g., a 4x4 block in the center)
    dead_zone_mask = np.zeros((grid_size, grid_size), dtype=bool)
    dz_start, dz_end = grid_size // 2 - 2, grid_size // 2 + 2
    dead_zone_mask[dz_start:dz_end, dz_start:dz_end] = True
    
    def get_disorder(g, m):
        d = 0
        for r in range(grid_size):
            for c in range(grid_size):
                if not m[r, c]: # If cell is NOT a dead zone
                    # Check right
                    if c + 1 < grid_size and not m[r, c+1]:
                        d += abs(g[r, c] - g[r, c+1])
                    # Check down
                    if r + 1 < grid_size and not m[r+1, c]:
                        d += abs(g[r, c] - g[r+1, c])
        return d

    initial_disorder = get_disorder(grid, dead_zone_mask)
    start_time = time.time()
    
    for _ in range(num_iterations):
        # Pick a random passable cell
        r = np.random.randint(0, grid_size)
        c = np.random.randint(0, grid_size)
        
        if not dead_zone_mask[r, c]:
            direction = np.random.choice(['U', 'D', 'L', 'R'])
            nr, nc = r, c
            if direction == 'U': nr -= 1
            elif direction == 'D': nr += 1
            elif direction == 'L': nc -= 1
            elif direction == 'R': nc += 1
            
            if 0 <= nr < grid_size and 0 <= nc < grid_size and not dead_zone_mask[nr, nc]:
                idx_curr = r * grid_size + c
                idx_neigh = nr * grid_size + nc
                # Sort based on flattened index priority
                if (grid[r, c] > grid[nr, nc] and idx_curr < idx_neigh) or \
                   (grid[r, c] < grid[nr, nc] and idx_curr > idx_neigh):
                    grid[r, c], grid[nr, nc] = grid[nr, nc], grid[r, c]

    end_time = time.time()
    final_disorder = get_disorder(grid, dead_zone_mask)
    success_metric = 1.0 - (final_disorder / initial_disorder) if initial_disorder > 0 else 0

    print(f"experiment: grid_sorting_dead_zones")
    print(f"initial_disorder: {initial_disorder:.4f}")
    print(f"final_disorder: {final_disorder:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()