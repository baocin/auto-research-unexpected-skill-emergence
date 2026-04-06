# Experiment Log

| # | Date | Experiment | Key Finding | Status |
|---|------|------------|-------------|--------|
| 1 | 2026-04-06 | Phase 1: Baseline (N=50, 100K steps) | All algorithms 0% success at any damage - frozen cell model was blocking active neighbors | crash/fixed |
| 2 | 2026-04-06 | Phase 1: Fixed frozen cells (N=50, 100K steps) | Still 0% - MAX_STEPS too low for random element selection | discarded |
| 3 | 2026-04-06 | Phase 1: Fixed bubble sort bidirectional + reduced N=20 | Bubble 97% at 10% dmg, Selection 100% at 10%, Insertion 0% at 5% | keep |
| 4 | 2026-04-06 | Phase 2: Extended algorithms (N=30, gnome sort added) | Gnome sort surprisingly fragile (15% at 10% dmg). Selection sort best (95%) | keep |
| 5 | 2026-04-06 | Phase 2: Chimeric experiments (N=30) | Bubble+selection chimera works, bubble+insertion fails completely | keep |
| 6 | 2026-04-06 | Phase 3: Target search (N=30, 20% damage) | Random walk (20%) outperforms scent gradient (5%) and wall follower (10%) | interesting |
