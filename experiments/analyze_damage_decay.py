import numpy as np

def analyze():
    # Data extracted from recent run.log
    results = [
        (0.0, 0.55),
        (0.2, 0.516),
        (0.4, 0.438),
        (0.6, 0.324),
        (0.8, 0.16)
    ]
    
    damage_rates = np.array([r[0] for r in results])
    performance = np.array([r[1] for r in results])
    
    # Linear regression to find decay rate
    slope, intercept = np.polyfit(damage_rates, performance, 1)
    
    print("experiment: anomaly_damage_analysis")
    print(f"decay_slope:{slope:.4f}")
    print(f"intercept:{intercept:.4f}")
    print(f"r_squared:{np.corrcoef(damage_rates, performance)[0,1]**2:.4f}")

if __name__ == "__main__":
    analyze()