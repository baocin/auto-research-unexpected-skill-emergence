# Emergent Competencies in Locally-Executed Algorithms

This is an autonomous research experiment exploring emergent behaviors in locally-executed algorithms, inspired by Levin et al. (2025).

## Background

Michael Levin's paper "Classical Sorting Algorithms as a Model of Morphogenesis" showed that when sorting algorithms are re-framed from top-down (a controller iterates over the array) to **cell-view** (each element autonomously executes its local sorting policy), emergent behaviors appear:
- Robust error tolerance with "frozen" (damaged) elements
- Detour navigation around defects
- Spontaneous clustering by algorithm type in chimeric (mixed-algorithm) arrays

They tested Bubble Sort, Insertion Sort, and Selection Sort. **Our goal: replicate, extend, and explore whether these properties generalize.**

## Setup

To set up a new experiment:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr6`). The branch `research/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b research/<tag>` from current main.
3. **Read the in-scope files**:
   - `PROJECT_PROPOSAL.md` — research plan and motivation
   - `experiment.py` — the file you modify. Contains algorithms, metrics, and experiment runner.
   - `REPORT.md` — the research report you update with findings (create if missing)
   - `EXPERIMENT_LOG.md` — chronological log of each experiment (create if missing)
4. **Verify Python works**: `python3 -c "import numpy; print('OK')"`. If numpy is missing, install it: `pip3 install numpy matplotlib`.
5. **Establish baseline**: Run `python3 experiment.py > run.log 2>&1` to verify everything works.
6. **Confirm and go**.

## Experimentation

Each experiment modifies `experiment.py` to test a hypothesis, runs it, and records results.

**What you CAN do:**
- Modify `experiment.py` — this is the primary file you edit. Add new algorithms, new metrics, new experiment configurations.
- Create new Python files in `experiments/` for complex experiments that don't fit in the main file.
- Update `REPORT.md` with findings after significant results.
- Create visualization scripts that save to `figures/`.

**What you CANNOT do:**
- Install heavy packages (no PyTorch, TensorFlow, etc.). Stick to numpy, matplotlib, and standard library.
- Run experiments that take more than 5 minutes wall clock. If an experiment is slow, reduce `NUM_TRIALS` or `ARRAY_SIZE`.
- Use more than ~2GB of memory. Keep `ARRAY_SIZE` ≤ 500 and `NUM_TRIALS` ≤ 200.

**The goal: discover and characterize emergent competencies in locally-executed algorithms.**

## Research Phases

### Phase 1: Replication
Replicate Levin et al.'s core findings:
- Cell-view bubble/insertion/selection sort with frozen cells
- Verify cell-view outperforms traditional under damage
- Reproduce chimeric clustering behavior
- **Success criteria**: cell-view success rate > traditional at ≥10% damage for bubble and insertion sort

### Phase 2: Extended Sorting
Test additional sorting algorithms in cell-view:
- Merge sort, quick sort, shell sort, cocktail shaker sort, gnome sort, comb sort
- For each: design cell-view variant, test with damage, compare to traditional
- Test chimeric combinations

### Phase 3: Beyond Sorting
Test non-sorting algorithms:
- Distributed search, graph algorithms, consensus, anomaly detection, clustering, load balancing
- For each: define success metric, implement cell-view, test with damage

### Phase 4: Analysis
- What algorithm properties predict emergent robustness?
- Taxonomy of emergent competencies
- Problem space traversal analysis

## Output format

After each run, the script prints a parseable summary:

```
---
experiment:       phase1_replication_bubble_sort
array_size:       100
num_trials:       50
total_seconds:    45.2
success_rate_bubble_dmg0.3: 0.8400
num_algorithms:   3
num_chimeric:     3
results_file:     run_results.json
```

Extract key metrics: `grep "^success_rate\|^experiment:" run.log`

## Logging results

Log every experiment to `results.tsv` (tab-separated):

```
commit	experiment	key_metric	value	time_sec	status	description
```

1. Git commit hash (short, 7 chars)
2. Experiment name
3. Key metric name (e.g. `success_rate_bubble_dmg0.3`)
4. Metric value
5. Wall-clock seconds
6. Status: `keep`, `discard`, `crash`, or `interesting`
7. Short description

Use `interesting` status for experiments that reveal unexpected or noteworthy emergent behavior (even if the metric isn't strictly "better").

## Updating the report

After every 3-5 experiments (or whenever you have a significant finding), update `REPORT.md`:
- Add a section for the new finding
- Include key numbers and what they mean
- Note any surprising or unexpected results
- Add figures if generated

The report should be readable by someone who hasn't seen the raw data. Write it as a research report, not a log.

## The experiment loop

LOOP FOREVER:

1. **Plan**: Based on current phase and prior results, choose the next experiment. Start with Phase 1 and advance when success criteria are met.
2. **Implement**: Modify `experiment.py` (or create a new file in `experiments/`) with the experimental setup.
3. **Commit**: `git commit -am "experiment: <short description>"`
4. **Run**: `python3 experiment.py > run.log 2>&1`
5. **Analyze**: `grep "^success_rate\|^experiment:\|^total_seconds:" run.log`
   - If grep is empty, check `tail -50 run.log` for errors.
6. **Record**: Add results to `results.tsv`
7. **Report**: Update `REPORT.md` if this is a significant finding
8. **Log**: Append to `EXPERIMENT_LOG.md` with timestamp, what you tried, what happened
9. **Decide**:
   - If results are interesting → keep the commit, advance
   - If results are boring/expected → keep anyway (negative results are data)
   - If crash → fix or move on
10. **Repeat**

**Memory/time guard rails:**
- If an experiment takes > 3 minutes, reduce parameters and re-run
- If you get a MemoryError, halve `ARRAY_SIZE` or `NUM_TRIALS`
- Prefer running many small experiments over few large ones

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. Run experiments continuously. If you run out of ideas in one phase, advance to the next. If all phases feel explored, go deeper — try different damage models (random noise instead of frozen, adversarial damage), different array initializations (nearly-sorted, reverse-sorted), or entirely new algorithm families.

## What makes a finding "interesting"

- Cell-view outperforms traditional under damage (replicates Levin)
- An algorithm shows WORSE cell-view performance (counter-example)
- Chimeric arrays show spontaneous clustering
- An algorithm shows detour behavior (temporarily increasing disorder to route around damage)
- A non-sorting algorithm exhibits any of the above
- Unexpected scaling behavior (works at N=50 but fails at N=200, or vice versa)
