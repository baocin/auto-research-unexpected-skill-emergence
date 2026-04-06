# LMStudio Log Analysis — 2026-04-06

## Session Overview
- **Model:** gemma-4-26b-a4b-it (Q4_K_M, via LMStudio)
- **Time span:** 00:55:18 — 08:08:52 (~7h 13m wall clock)
- **Total inference time:** 6.32 hours (22,754s)
- **Log files:** 10 rotated logs, ~95.6 MB total

## Token Usage

| File | Requests | Prompt Tokens | Completion Tokens | Total Tokens |
|------|----------|--------------|-------------------|--------------|
| 2026-04-06.1.log | 112 | 2,542,805 | 58,878 | 2,601,683 |
| 2026-04-06.2.log | 112 | 2,579,269 | 56,938 | 2,636,207 |
| 2026-04-06.3.log | 71 | 2,800,195 | 52,424 | 2,852,619 |
| 2026-04-06.4.log | 54 | 2,850,093 | 36,719 | 2,886,812 |
| 2026-04-06.5.log | 59 | 2,706,294 | 47,343 | 2,753,637 |
| 2026-04-06.6.log | 91 | 2,545,678 | 68,860 | 2,614,538 |
| 2026-04-06.7.log | 71 | 2,780,279 | 31,228 | 2,811,507 |
| 2026-04-06.8.log | 59 | 2,825,495 | 27,549 | 2,853,044 |
| 2026-04-06.9.log | 66 | 2,643,216 | 43,184 | 2,686,400 |
| 2026-04-06.10.log | 52 | 1,611,534 | 30,986 | 1,642,520 |

### Aggregate Totals
| Metric | Value |
|--------|-------|
| **Total requests** | 747 |
| **Prompt tokens** | 25,884,858 |
| **Completion tokens** | 454,109 |
| **Total tokens** | **26,338,967** |
| Avg prompt/request | 34,652 |
| Avg completion/request | 608 |

## Throughput
| Phase | Tokens | Time | Throughput |
|-------|--------|------|------------|
| Prompt eval | 612,994 | 2,711.7s | 226 tok/s |
| Generation | 454,109 | 20,042.4s | 22.7 tok/s |

**Note:** Prompt eval tokens (612,994) are lower than prompt_tokens (25.9M) because llama.cpp caches previously evaluated prompt tokens via KV cache — only new/changed tokens require re-evaluation.

## Files
- `2026-04-06.{1-10}.log` — original log files
- `merged_2026-04-06.log` — concatenated merged log
