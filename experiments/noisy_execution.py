import numpy as np
import time

def bubble_sort_noisy(arr, noise_prob=0.0):
    """
    Bubble sort implementation where each comparison has a chance 
    to return the wrong result (simulating bit-flip/noise).
    """
    n = len(arr)
    data = arr.copy()
    for i in range(n):
        for j in range(0, n - i - 1):
            # Simulate noisy comparison
            val_a = data[j]
            val_b = data[j+1]
            
            # The 'true' comparison result
            true_comparison = val_a > val_b
            
            # Apply noise
            if np.random.rand() < noise_prob:
                actual_comparison = not true_comparison
            else:
                actual_comparison = true_comparison
            
            if actual_comparison:
                data[j], data[j+1] = data[j+1], data[j]
    return data

def run_experiment(n_range, noise_probs):
    results = []
    for n in n_range:
        for p in noise_probs:
            # Run multiple trials for each configuration to get average success rate
            trials = 20
            successes = 0
            start_time = time.time()
            
            for _ in range(trials):
                original = np.random.permutation(n)
                sorted_arr = bubble_sort_noisy(original, noise_prob=p)
                if np.array_equal(sorted_arr, np.sort(original)):
                    successes += 1
            
            duration = time.time() - start_time
            success_rate = successes / trials
            
            results.append({
                'n': n,
                'noise_prob': p,
                'success_rate': success_rate,
                'avg_time_per_trial': duration / trials
            })
    return results

if __name__ == "__main__":
    # Test parameters: small arrays to keep execution within limits
    n_values = [10, 20, 50]
    noise_levels = [0.0, 0.01, 0.05, 0.1]
    
    print(f"experiment: noisy_bubble_sort")
    results = run_experiment(n_values, noise_levels)
    
    for res in results:
        print(f"n:{res['n']} noise:{res['noise_prob']} success_rate:{res['success_rate']} avg_time:{res['avg_time_per_trial']:.4f}")