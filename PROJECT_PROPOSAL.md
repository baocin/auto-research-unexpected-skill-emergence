# Emergent Competencies in Locally-Executed Algorithms

## Research Proposal

### Motivation

Michael Levin, Taining Zhang, and Adam Goldstein published ["Classical Sorting Algorithms as a Model of Morphogenesis"](https://arxiv.org/abs/2401.05375) (2025), demonstrating that when sorting algorithms are re-framed from top-down (a controller iterating over the array) to bottom-up (each element autonomously executing the sorting policy locally), surprising emergent behaviors appear:

1. **Robust error tolerance** - "Cell-view" arrays with autonomous elements sort themselves more reliably than traditional implementations when elements are "damaged" (frozen/non-functional).
2. **Detour navigation** - Elements temporarily reduce sorting progress to route around defective neighbors.
3. **Chimeric clustering** - In arrays where elements follow different sorting algorithms ("algotypes"), elements spontaneously cluster by algorithm type during sorting, then disperse once sorted. This behavior is nowhere in the code.

The original paper tested three algorithms: **Bubble Sort, Insertion Sort, and Selection Sort**, with "frozen cells" (elements that fail to execute) as the damage model. Bubble Sort showed superior robustness to disruptions; Selection Sort performed worse than its traditional counterpart.

**Reference code**: [github.com/Zhangtaining/sorting_with_noise](https://github.com/Zhangtaining/sorting_with_noise)

### Research Questions

**Primary**: Do emergent competencies (error tolerance, detour navigation, chimeric clustering) generalize beyond the three sorting algorithms tested in the original paper?

**Secondary**:
1. Do other locally-executable algorithms (beyond sorting) exhibit similar emergent properties when run in a decentralized, fault-tolerant manner?
2. What properties of an algorithm predict whether it will exhibit emergent competencies under local execution with noise?
3. Is there a relationship between algorithm complexity and the degree of emergent robustness?

### Proposed Experiments

#### Phase 1: Replication (Weeks 1-2)

Replicate the Levin et al. results using their published code and methodology:

- Implement cell-view versions of Bubble Sort, Insertion Sort, and Selection Sort
- Introduce frozen cells at varying rates (5%, 10%, 20%, 30%)
- Measure sorting success rate, steps to completion, and detour behavior
- Reproduce chimeric array experiments (mixed algotypes)
- **Success criteria**: Qualitatively match the published results

**Metrics**:
- `sortedness_ratio` - fraction of array in correct sorted order (0.0 to 1.0)
- `steps_to_sort` - number of local execution steps until sorted (or timeout)
- `success_rate` - fraction of trials that reach fully sorted state
- `cluster_coefficient` - degree to which same-algotype elements are adjacent (for chimeric experiments)

#### Phase 2: Extended Sorting Algorithms (Weeks 2-3)

Test additional sorting algorithms in cell-view:

| Algorithm | Why interesting |
|-----------|----------------|
| **Merge Sort** | Divide-and-conquer; requires cooperation between sub-arrays |
| **Quick Sort** | Pivot-based partitioning; local decisions have global impact |
| **Shell Sort** | Gap-based comparison; non-local element interaction |
| **Cocktail Shaker Sort** | Bidirectional bubble sort; richer local dynamics |
| **Gnome Sort** | Simple, single-element-focused; minimal local policy |
| **Comb Sort** | Gap-shrinking; bridges local and global |

For each:
- Design a cell-view (locally-executable) variant
- Test with frozen cells at 0%, 5%, 10%, 20% damage rates
- Compare robustness against traditional (top-down) implementation
- Test in chimeric arrays (mixed with Bubble Sort as baseline)

#### Phase 3: Beyond Sorting (Weeks 3-5)

Test whether emergent competencies appear in other locally-executable algorithms:

| Domain | Algorithm | Local execution model |
|--------|-----------|----------------------|
| **Search** | Binary search on distributed elements | Each element decides if target is left/right of it |
| **Graph** | Distributed shortest path (Bellman-Ford) | Each node relaxes its own edges |
| **Consensus** | Voting / majority finding | Each element polls neighbors |
| **Anomaly detection** | Local outlier detection | Each element compares itself to neighbors |
| **Clustering** | K-means with local updates | Each point reassigns itself to nearest centroid |
| **Load balancing** | Work stealing | Each processor decides to share/request work |
| **Cellular automata** | Rule-based (Game of Life variants) | Each cell updates based on neighbors |

For each:
- Define a "correct outcome" metric (analogous to sorted order)
- Introduce element failures (frozen, noisy, adversarial)
- Measure robustness vs. centralized implementation
- Test chimeric versions (mixed algorithms for same goal)

#### Phase 4: Analysis and Theory (Week 5+)

- Characterize which algorithm properties predict emergent robustness
- Map the "problem space traversal" for each algorithm (as Levin et al. did for sorting)
- Identify common patterns across domains
- Propose a taxonomy of emergent competencies in locally-executed algorithms

### Experimental Setup

**Hardware constraints**: Local machine (macOS, no GPU required for this research). Experiments must be memory-conscious:
- Array sizes: start at N=50, scale to N=500 max
- Trial counts: 100 trials per configuration (increase if variance is high)
- Timeout: 10,000 local steps per trial
- Parallelism: use Python multiprocessing where helpful, but cap at available cores

**Software stack**:
- Python 3.10+
- NumPy for array operations
- Matplotlib for visualization
- No heavy ML frameworks needed (this is algorithmic, not neural)

**Output artifacts**:
- `REPORT.md` - continuously updated research report with findings
- `experiments/` - all experiment scripts, organized by phase
- `results/` - raw data (CSV/TSV) from each experiment
- `figures/` - generated plots and visualizations
- `EXPERIMENT_LOG.md` - chronological log of every experiment run

### Relationship to Auto-Research Framework

The Karpathy auto-research framework is designed for autonomous neural network optimization (modify `train.py`, measure `val_bpb`, iterate). This project is fundamentally different:

- **No neural network training** - we're running algorithmic experiments
- **No GPU required** - CPU-only Python experiments
- **Different metrics** - sorting success rate, robustness, clustering coefficients
- **Different loop** - each "experiment" tests a new algorithm/configuration, not a hyperparameter tweak

We will adapt the auto-research `program.md` to drive an autonomous experiment loop that:
1. Implements a new algorithm variant or configuration
2. Runs trials with multiple damage rates
3. Records metrics
4. Decides whether results are interesting enough to keep
5. Moves to the next experiment

### Expected Outcomes

1. A replication of Levin et al.'s sorting results
2. Data on whether additional sorting algorithms exhibit similar emergent properties
3. Evidence for or against emergent competencies in non-sorting algorithms
4. A characterization of what makes an algorithm amenable to robust local execution
5. A comprehensive research report with all experimental results

### References

- Zhang, T., Goldstein, A., & Levin, M. (2025). "Classical Sorting Algorithms as a Model of Morphogenesis: Self-sorting arrays reveal unexpected competencies in a minimal model of basal intelligence." *Adaptive Behavior*, 33(1). [arXiv:2401.05375](https://arxiv.org/abs/2401.05375)
- Levin, M. (2025). "Algorithms Redux: finding unexpected properties in truly minimal systems." [Blog post](https://thoughtforms.life/algorithms-redux-finding-unexpected-properties-in-truly-minimal-systems/)
- Code: [github.com/Zhangtaining/sorting_with_noise](https://github.com/Zhangtaining/sorting_with_noise)
