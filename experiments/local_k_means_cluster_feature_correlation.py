import numpy as np
import time

def run_correlation_experiment(n_agents=100, correlation_strength=0.5, iterations=30):
    """
    Investigates 'Feature Correlation': How the dependency between 
    the two dimensions (x and y) affects the convergence of clustering.
    High correlation makes clusters appear as lines rather than blobs.
    """
    np.random.seed(42)
    # Generate correlated data using a covariance matrix
    mean = [0, 0]
    # Covariance matrix: [[1, rho], [rho, 1]]
    cov = [[1.0, correlation_strength], [correlation_strength, 1.0]]
    
    # Cluster 1 centered at [0,0]
    c1 = np.random.multivariate_normal(mean, cov, n_agents // 2)
    # Cluster 2 centered at [5,5]
    c2 = np.random.multivariate_normal([5, 5], cov, n_agents // 2)
    
    points = np.vstack([c1, c2])
    initial_variance = np.var(points)
    current_points = points.copy()
    
    for _ in range(iterations):
        snapshot = current_points.copy()
        new_points = []
        for i in range(n_agents):
            p_i = snapshot[i]
            dists = np.linalg.norm(snapshot - p_i, axis=1)
            neighbors_mask = (dists <= 3.0) & (np.arange(n_agents) != i)
            
            if np.any(neighbors_mask):
                centroid = np.mean(snapshot[neighbors_mask], axis=0)
                new_p = p_i + 0.5 * (centroid - p_i)
                new_points.append(new_p)
            else:
                new_points.append(p_i)
        current_points = np.array(new_points)
    
    final_variance = np.var(current_points)
    success_metric = 1.0 - (final_variance / initial_variance) if initial_variance > 0 else 0
    return success_metric

def run_experiment():
    print(f"experiment: k_means_feature_correlation")
    # Test correlation strengths from -0.9 to 0.9
    correlations = [-0.9, -0.5, 0.0, 0.5, 0.9]
    for rho in correlations:
        start_time = time.time()
        success = run_correl_experiment(correlation_strength=rho)
        end_time = time.time()
        print(f"rho_{rho:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

# Need to fix the function name mismatch in run_experiment
def run_experiment_fixed():
    print(f"experiment: k_means_feature_correlation")
    correlations = [-0.9, -0.5, 0.0, 0.5, 0.9]
    for rho in correlations:
        start_time = time.time()
        success = run_correlation_experiment(correlation_strength=rho)
        end_time = time.time()
        print(f"rho_{rho:.1f}_success_{success:.4f}_time_{end_time - start_time:.4f}")

if __name__ == "__main__":
    run_experiment_fixed()