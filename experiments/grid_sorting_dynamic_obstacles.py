import numpy as np
import time

def run_experiment():
    grid_size = 10
    num_iterations = 200
    obstacle_density = 0.2
    
    grid = np.random.rand(grid_size, grid_size)
    
    def get_disorder(g, m):
        d = 0
        for r in range(grid_size):
            for c in range(grid_size):
                if not m[r, c]:
                    if r + 1 < grid_size and not m[r+1, c]:
                        d += abs(g[int(r), int(c)] - g[int(r+1), int(c)])
                    if c + 1 < grid_size and not m[r, c+1]:
                        d += abs(g[int(r), int(c)] - g[int(r), int(c+1)])
        return d

    initial_disorder = get_disorder(grid, np.zeros((grid_size, grid_size), dtype=bool))
    start_time = time.time()
    
    for _ in range(num_iterations):
        obstacle_mask = np.random.rand(grid_size, grid_size) < obstacle_density
        r = np.random.randint(0, grid_size)
        c = np.random.randint(0, grid_size)
        
        if not obstacle_mask[r, c]:
            direction = np.random.choice(['U', 'D', 'L', 'R'])
            nr, nc = r, c
            if direction == 'U': nr -= 1
            elif direction == 'D': nr += 1
            elif direction == 'L': nc -= 1
            elif direction == 'R': nc += 1
            
            if 0 <= nr < grid_size and 0 <= nc < grid_size and not obstacle_mask[nr, nc]:
                idx_curr = r * grid_size + c
                idx_neigh = nr * grid_size + nc
                if (grid[r, c] > grid[nr, nc] and idx_curr < idx_neigh) or \
                   (grid[r, c] < grid[nr, nc] and idx_curr > idx_neigh):
                    grid[r, c], grid[nr, nc] = grid[nr, nc], grid[r, c]

    end_time = time.time()
    final_disorder = get_disorder(grid, np.zeros((grid_size, grid_size), dtype=bool))
    success_metric = 1.0 - (final_disorder / initial_disorder) if initial_disorder > 0 else 0

    print(f"experiment: grid_sorting_dynamic_obstacles")
    print(f"initial_disorder: {initial_disorder:.4f}")
    print(f"final_disorder: {final_disorder:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()