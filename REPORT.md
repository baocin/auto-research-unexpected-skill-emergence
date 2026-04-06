# Emergent Competencies in Locally-Executed Algorithms: Research Report

## Abstract

This project investigates whether emergent competencies observed in locally-executed sorting algorithms (Levin et al., 2025) generalize to other algorithms and computational domains. We replicate the original findings and extend them to additional sorting algorithms and non-sorting domains (target search). Our key finding is that **algorithm interaction range** is the primary predictor of emergent robustness: non-local algorithms (selection sort) maintain functionality at far higher damage rates than local ones (bubble, insertion, gnome sort).

## 1. Introduction

Zhang, Goldstein, and Levin (2025) demonstrated that classical sorting algorithms, when re-framed from top-down execution to autonomous "cell-view" execution (each element executes its sorting policy locally), exhibit surprising emergent properties: robust error tolerance, detour navigation, and spontaneous clustering in chimeric (mixed-algorithm) arrays. This project asks: **how general are these phenomena?**

## 2. Methodology

### Cell-View Model
Each array element is an autonomous agent. On each time step, a random element is selected and executes its local sorting policy. "Frozen" (damaged) elements skip their turn but can be interacted with by active neighbors.

### Parameters
- Array sizes: 20-30 elements
- Damage rates: 0%, 5%, 10%, 20%, 30% frozen cells
- Trials: 20-50 per configuration
- Max steps: 50,000-100,000
- Metrics: success rate, mean steps to sort, normalized disorder, cluster coefficient

## 3. Phase 1 Results: Replication (Bubble, Insertion, Selection)

### 3.1 Success Rate vs. Damage (N=20, 30 trials, 50K steps)

| Algorithm | 0% dmg | 5% dmg | 10% dmg | 20% dmg | 30% dmg |
|-----------|--------|--------|---------|---------|---------|
| **Bubble** | 100% | 100% | 97% | 57% | 3% |
| **Insertion** | 100% | 0% | 0% | 0% | 0% |
| **Selection** | 100% | 100% | 100% | 80% | 27% |

### 3.2 Key Findings

**Selection sort is the most robust.** At 10% damage, selection sort maintains 100% success while bubble drops to 97% and insertion collapses to 0%. This matches Levin et al.'s findings.

**Insertion sort collapses immediately.** Even 5% damage causes complete failure. Insertion sort's unidirectional policy (elements only move left) means a single frozen cell creates an impassable barrier - elements to the right can never cross past a frozen neighbor on their left.

**Bubble sort shows graceful degradation.** The improved bidirectional bubble sort (checking both left and right neighbors) degrades smoothly: 100% -> 100% -> 97% -> 57% -> 3%. This bidirectional local interaction allows elements to route around frozen cells from either direction.

**Why selection sort wins:** Selection sort compares with random elements anywhere in the array (non-local interaction), allowing it to bypass frozen cells entirely. Local algorithms (bubble, insertion) propagate information only through neighbor-to-neighbor swaps, making them vulnerable to frozen "blockades."

### 3.3 Chimeric Results (N=20, 30 trials)

| Pair | 0% dmg | 5% dmg | 10% dmg | 20% dmg |
|------|--------|--------|---------|---------|
| Bubble + Insertion | 0% | 0% | 0% | 0% |
| **Bubble + Selection** | 100% | 67% | 43% | 13% |
| **Insertion + Selection** | 100% | 60% | 43% | 10% |

**Chimeric arrays with selection sort are more robust than either pure local algorithm.** The selection sort elements act as "bridges" that can move values past frozen cells, helping the entire array sort. Bubble+insertion chimeras fail entirely because both algorithms are local.

**Clustering coefficient** hovered around 0.50-0.59 across conditions, suggesting modest but consistent same-algotype clustering during sorting.

## 4. Phase 2 Results: Extended Algorithms

### 4.1 Gnome Sort (N=30, 20 trials, 100K steps)

| Algorithm | 0% dmg | 10% dmg | 20% dmg | 30% dmg |
|-----------|--------|---------|---------|---------|
| Bubble | 100% | 95% | 20% | 0% |
| Insertion | 100% | 0% | 0% | 0% |
| Selection | 100% | 95% | 45% | 35% |
| **Gnome** | 100% | 15% | 5% | 0% |

**Gnome sort is fragile.** Despite its "back-and-forth" movement (similar to bubble sort), gnome sort collapses rapidly under damage. At 10% damage, only 15% success vs. 95% for bubble sort. This is because gnome sort's sequential traversal pattern makes it more dependent on continuous element accessibility.

### 4.2 Robustness Hierarchy

From most to least robust under damage:
1. **Selection Sort** (non-local interaction) - highest tolerance
2. **Bubble Sort** (bidirectional local) - graceful degradation
3. **Gnome Sort** (sequential local) - rapid collapse
4. **Insertion Sort** (unidirectional local) - immediate collapse

## 5. Phase 3 Results: Beyond Sorting (Target Search)

### 5.1 Search Experiment Design
An agent on a 1D array attempts to find a target value. Three strategies:
- **Random walk**: move left or right randomly
- **Scent gradient**: move toward target (deterministic)
- **Wall follower**: move toward target but navigate around frozen cells

### 5.2 Results (N=30, 20% damage)

| Strategy | Success Rate | Avg Steps |
|----------|-------------|-----------|
| Random Walk | 20% | 4,004 |
| **Scent Gradient** | 5% | 4,750 |
| Wall Follower | 10% | 4,500 |

**The simplest strategy wins.** Random walk outperforms both "intelligent" strategies because its stochastic nature allows it to explore around frozen barriers. The deterministic scent gradient gets trapped against frozen cells it cannot pass. This parallels the sorting findings: stochasticity provides robustness.

## 6. Discussion

### 6.1 The Interaction Range Hypothesis

Our central finding is that **interaction range** is the strongest predictor of emergent robustness:

- **Non-local algorithms** (selection sort, random walk) can bypass damage because they interact with distant elements
- **Local algorithms** (bubble, insertion, gnome) propagate information only through neighbors, making them vulnerable to "blockades" where frozen cells create impassable barriers
- **Bidirectional local** (improved bubble sort) is intermediate - it can route around damage from either direction but is still limited by locality

### 6.2 Stochasticity as Robustness

A second key finding: **stochastic elements provide robustness.** Random walk outperforms deterministic gradient following in damaged environments. This parallels biological systems where noise and randomness enable exploration past local obstacles.

### 6.3 Chimeric Synergy

Mixing a non-local algorithm with a local one creates a chimeric system that inherits some of the non-local algorithm's robustness. The non-local elements act as "bridges" that transport values past frozen barriers, assisting the local elements.

### 6.4 Comparison to Levin et al.

Our results broadly replicate the original findings:
- Cell-view sorting works and shows emergent robustness
- Different algorithms show different damage tolerance profiles
- Chimeric arrays exhibit interesting cooperative behavior

We extend the findings by:
- Identifying interaction range as the key variable
- Showing gnome sort is surprisingly fragile
- Demonstrating the pattern extends beyond sorting to search tasks
- Finding that stochasticity is a robustness mechanism

## 7. Phase 4 Results: Distributed Consensus

### 7.1 Experiment Design
A ring of N=50 nodes, each holding a binary value (0 or 1). On each step, a random active node adopts the majority value among itself and its two neighbors. Frozen nodes are fixed to value 1 (acting as "anchors"). Optional noise flips the decision with probability 0.05.

### 7.2 Results: Damage HELPS Consensus

| Damage Rate | Agreement (fraction at value 1) |
|-------------|--------------------------------|
| 0% | 0.498 (random) |
| 10% | 0.617 |
| 20% | 0.678 |
| 30% | 0.760 |
| 40% | 0.824 |
| 50% | 0.899 |
| 60% | 0.926 |

**This is the opposite of sorting.** In sorting, frozen cells are obstacles that hinder the algorithm. In consensus, frozen cells are anchors that *drive* the system toward agreement. Agreement monotonically increases with damage rate.

### 7.3 Interpretation: Context-Dependent Role of Damage

The frozen nodes exert a persistent directional pressure. In a system that would otherwise wander randomly (majority voting on random initial conditions converges to ~50/50), the frozen anchors provide a "gravitational pull" toward their value. More anchors = stronger pull = faster/higher agreement.

This reveals that **the effect of "damage" depends on alignment between the frozen state and the task goal:**
- In sorting: frozen cells are randomly placed and hold random values → they're obstacles
- In consensus: frozen cells hold a consistent value → they're helpful anchors
- This suggests a deeper principle: damage that is *structured* (consistent) can be constructive, while *random* damage is destructive

### 7.4 Noise Interaction
Adding 5% noise slightly reduces agreement at low damage but actually marginally *increases* it at high damage (0.836 vs 0.833 at 40%). The noise helps explore past local minima, similar to the random walk vs. gradient finding in Phase 3.

## 8. Emerging Theory: A Taxonomy of Damage Effects

| Domain | Damage Effect | Why |
|--------|--------------|-----|
| **Sorting** | Destructive | Frozen cells block information flow between neighbors |
| **Search** | Partially destructive | Frozen cells block paths but stochastic agents can route around |
| **Consensus** | **Constructive** | Frozen cells act as persistent anchors toward a consistent state |

**Key insight:** The effect of structural damage depends on three factors:
1. **Interaction range** of the algorithm (local vs. non-local)
2. **Alignment** between frozen state and task goal (random vs. structured)
3. **Stochasticity** of the active agents (deterministic agents get trapped; random ones explore)

## 9. Phase 5 Results: Adversarial Consensus and Extended Domains

### 9.1 Adversarial Consensus (Half Frozen at 0, Half at 1)

When frozen nodes are split between opposing values, consensus collapses to 3% success. The system enters permanent tension — active nodes are pulled in both directions and cannot settle. This confirms that constructive damage requires **alignment** (all anchors pulling the same way).

At 10% adversarial frozen nodes, the system still achieves 75% success. The critical threshold where adversarial anchors overwhelm the free population falls between 30-50%.

### 9.2 Noisy Execution (New Damage Model)

Instead of frozen cells, elements randomly fail to sort correctly with probability p. Results with redundancy (k retries per comparison):

| N | Noise (p) | Retries (k) | Bubble Success | Insertion Success |
|---|-----------|-------------|---------------|------------------|
| 10 | 0.05 | 1 | 40% | 30% |
| 10 | 0.05 | 3 | 80% | 85% |
| 10 | 0.05 | 5 | 95% | 90% |
| 20 | 0.05 | 1 | 5% | 0% |
| 20 | 0.05 | 5 | 95% | 85% |

**Redundancy rescues noisy sorting.** Multiple retries act as a majority-vote filter, suppressing random errors. This is analogous to biological error-correction mechanisms (e.g., DNA proofreading).

### 9.3 Anomaly Detection with Damaged Sensors

Local averaging filters out sensor noise with remarkable resilience:
- 100% detection success even at 40% sensor failure and noise level 0.5
- Breakdown occurs at noise=1.0: success drops to 40-53%

**Failure-Induced Detection Paradox**: At high noise (p=1.0), increasing sensor failure from 0% to 40% actually *improves* detection from 43% to 63%. Dead sensors (outputting zero) reduce background variance, making real anomalies more visible against a quieter baseline. This is a variance-reduction effect, distinct from the regularization seen in k-means.

### 9.4 Grid Sorting (2D)

2D grid sorting with noise shows rapid degradation with scale:
- 3x3 grid: 100% at 0% noise, 45% at 5% noise
- 4x4 grid: 100% at 0% noise, 25% at 5% noise

Adaptive rerouting (gradient-aware detour logic) maintains positive progress even at 40% obstacle density, but at the cost of 10x more swap operations.

## 10. Phase 6 Results: K-means Clustering

### 10.1 Interaction Radius Phase Transition

| Radius | Success |
|--------|---------|
| 0.5 | 0.4% |
| 2.0 | 11% |
| 4.0 | 11% |
| 8.0 | 99.9% |

A sharp transition occurs when interaction radius exceeds the gap between clusters. Below threshold: local refinement only. Above: global convergence.

### 10.2 Dropout as Regularization

| Dropout Rate | Success |
|-------------|---------|
| 0% | 11.75% |
| 10% | 96.7% |
| 30% | 100% |
| 50% | 100% |
| 80% | 100% |

Feature dropout *dramatically* improves convergence — from 12% to 100%. Analogous to dropout regularization in neural networks: noise prevents overfitting to local configurations and enables global convergence.

### 10.3 Noise as Exploration

| Noise Level | Success |
|-------------|---------|
| 0.1 | 0.2% |
| 0.5 | 5% |
| 1.0 | 19% |
| 2.0 | 67% |
| 4.0 | 50% |

An **optimal noise level** exists. Too little noise = stuck in local minima. Too much = destructive randomness. Peak at noise=2.0. This mirrors simulated annealing temperature schedules.

### 10.4 Additional K-means Findings

- **Scale invariance**: Success is perfectly constant (11.75%) across 0.1x-100x spatial scaling when radius scales proportionally
- **Heterogeneous radius**: A few long-range agents (radius 6.0) acting as "bridges" boost success from 12% to 86%
- **Structural entropy**: Dispersed clusters (high entropy) enable 99.9% success vs 29% for compact clusters — dispersion creates scaffolding for convergence
- **Sensor damage**: K-means maintains 100% precision at K=2 even with 50% feature corruption
- **Population balance**: Balanced populations (50/50) converge better (25%) than imbalanced (90/10 = 15%)

## 11. Phase 7 Results: Extended Consensus Experiments

### 11.1 Oscillating Adversary

A node that flips between 0 and 1 every iteration: 75% consensus success. The network's collective inertia dampens the high-frequency oscillation.

### 11.2 Malicious Injection

An attacker injects a new random value every iteration: 92% success. The "consensus momentum" of healthy nodes overwhelms the single adversary.

### 11.3 Edge Error Scaling

Corrupting communication channels (edges) rather than nodes:

| Edge Error Rate | Success |
|----------------|---------|
| 0% | 100% |
| 10% | 91% |
| 20% | 88% |
| 40% | 78% |

Graceful degradation with no phase transition — information redundancy across multiple edges provides natural resilience.

### 11.4 Critical Tracking Threshold

Consensus with drifting anchors (anchors that move over time):

| Drift Rate | Success |
|-----------|---------|
| 0.01 | 92% |
| 0.05 | collapse |
| 0.10 | collapse |

A sharp threshold exists where the rate of environmental change exceeds the convergence rate of local updates. Below 0.01: successful tracking. Above 0.05: catastrophic failure.

## 12. Unified Theory

### 12.1 Three Axes of Emergent Robustness

Our experiments across sorting, search, consensus, anomaly detection, and clustering reveal three independent axes that determine whether a locally-executed algorithm exhibits emergent robustness under damage:

**Axis 1: Interaction Range**
- Non-local > bidirectional local > unidirectional local
- A few long-range "bridge" agents can transform a fragile system into a robust one

**Axis 2: Damage Alignment**
- Structured/consistent damage can be constructive (consensus anchors)
- Random damage is destructive (sorting blockades)
- Adversarial damage (opposing anchors) is maximally destructive

**Axis 3: Stochasticity**
- Noise enables exploration past obstacles and local minima
- Optimal noise levels exist (too much is also harmful)
- Dropout/feature erasure acts as regularization
- Redundancy (multiple retries) provides error correction

### 12.2 Domain-Specific Effects

| Domain | Damage Effect | Key Mechanism |
|--------|--------------|---------------|
| Sorting | Destructive | Frozen cells block information propagation |
| Search | Partially destructive | Stochastic agents route around barriers |
| Consensus | Constructive (aligned) / Destructive (adversarial) | Anchors exert directional pressure |
| Anomaly Detection | Partially constructive | Dead sensors reduce background variance |
| K-means | Constructive (as noise/dropout) | Feature erasure prevents local trapping |

### 12.3 Broader Implications

These findings suggest that Levin et al.'s observations about sorting algorithms are instances of a more general phenomenon: **locally-executed algorithms can exhibit emergent competencies that are not explicitly programmed**, and the robustness of these competencies depends on the algorithm's communication topology, the nature of the damage, and the degree of stochasticity in the system.

This connects to broader themes in:
- **Biological robustness**: organisms tolerate cell damage through redundancy and non-local signaling
- **Distributed systems**: consensus protocols, gossip algorithms, and epidemic broadcasting
- **Machine learning**: dropout regularization, noise injection, ensemble methods

## 13. Experimental Infrastructure

### Runtime

| Metric | Value |
|--------|-------|
| Hardware | Apple M4 Pro |
| Research model | Gemma 4 26B A4B IT (Q4_K_M) via LM Studio |
| Orchestration | Claude Code (Opus 4.6) |
| Wall clock | 7h 13m |
| LLM requests | 747 |
| Total tokens | 26,338,967 |
| Generation throughput | 22.7 tok/s |
| Prompt eval throughput | 226 tok/s (97.6% KV cache reuse) |
| Agent iterations | ~600 |
| Experiment files | 189 |
| Git commits | 19 |

### Methodology Notes

- All experiments used NumPy with fixed random seeds for reproducibility
- Array sizes kept to N=20-50 to stay within memory and time constraints
- Each configuration tested across 20-50 trials
- The agent autonomously designed, implemented, ran, and analyzed experiments
- Human intervention limited to: fixing initial frozen-cell model, redirecting agent when stuck in repetition loops (3 times)

## 14. References

- Zhang, T., Goldstein, A., & Levin, M. (2025). "Classical Sorting Algorithms as a Model of Morphogenesis." *Adaptive Behavior*, 33(1). [arXiv:2401.05375](https://arxiv.org/abs/2401.05375)
- Levin, M. (2025). ["Algorithms Redux: finding unexpected properties in truly minimal systems."](https://thoughtforms.life/algorithms-redux-finding-unexpected-properties-in-truly-minimal-systems/)
- Original code: [github.com/Zhangtaining/sorting_with_noise](https://github.com/Zhangtaining/sorting_with_noise)
- Karpathy, A. (2026). [autoresearch](https://github.com/karpathy/autoresearch)

- Zhang, T., Goldstein, A., & Levin, M. (2025). "Classical Sorting Algorithms as a Model of Morphogenesis." *Adaptive Behavior*, 33(1). [arXiv:2401.05375](https://arxiv.org/abs/2401.05375)
- Code: [github.com/Zhangtaining/sorting_with_noise](https://github.com/Zhangtaining/sorting_with_noise)
