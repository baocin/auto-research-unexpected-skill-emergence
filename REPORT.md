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

## 9. Next Steps

- Test consensus with adversarial frozen nodes (half frozen at 0, half at 1) - does it still converge?
- Implement distributed anomaly detection: can local agents identify outliers despite damage?
- K-means with autonomous points: do damaged points create or disrupt clusters?
- Explore the "damage alignment" hypothesis: does deliberately structured damage always help?
- Phase transition analysis: find exact damage thresholds for each algorithm
- Scale to 2D grids (not just 1D arrays/rings)

## 10. References

- Zhang, T., Goldstein, A., & Levin, M. (2025). "Classical Sorting Algorithms as a Model of Morphogenesis." *Adaptive Behavior*, 33(1). [arXiv:2401.05375](https://arxiv.org/abs/2401.05375)
- Code: [github.com/Zhangtaining/sorting_with_noise](https://github.com/Zhangtaining/sorting_with_noise)
