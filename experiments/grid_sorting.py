import numpy as np
import time

def run_experiment():
    grid_size = 10
    grid = np.random.rand(grid_size, grid_size)
    
    def calculate_disorder(g):
        diffs = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if i + 1 < grid_size:
                    diffs += abs(g[i, j] - g[i+1, j])
                if j + 1 < grid_size:
                    diffs += abs(g[i, j] - g[i, j+1])
        return diffs

    initial_disorder = calculate_disorder(grid)
    start_time = time.time()
    
    iterations = 100
    for _ in range(iterations):
        for _ in range(grid_size * grid_size // 2):
            i = np.random.randint(0, grid_size)
            j = np.random.randint(0, grid_size)
            
            ni, nj = i, j
            direction = np.random.choice(['U', 'D', 'L', 'R'])
            if direction == 'U': ni -= 1
            elif direction == 'D': ni += 1
            elif direction == 'L': nj -= 1
            elif direction == 'R': nj += 1
            
            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                idx_curr = i * grid_size + j
                idx_neigh = ni * grid_size + nj
                # Swap if it moves larger values towards higher flattened indices
                if (grid[i, j] > grid[ni, nj] and idx_curr < idx_neigh) or \
                   (grid[i, j] < grid[ni, nj] and idx_curr > idx_neigh):
                    grid[i, j], grid[ni, nj] = grid[ni, nj], grid[i, j]

    end_time = time.time()
    final_disorder = calculate_disorder(grid)
    success_metric = 1.0 - (final_disorder / initial_disorder) if initial_disorder > 0 else 0

    print(f"experiment: grid_sorting")
    print(f"initial_disorder: {initial_disorder:.4f}")
    print(f"final_disorder: {final_disorder:.4f}")
    print(f"success_rate: {success_metric:.4f}")
    print(f"total_seconds: {end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment()