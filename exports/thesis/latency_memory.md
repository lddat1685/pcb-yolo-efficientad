# latency_memory

| Model | Split | Total mean (ms) | Total p50 (ms) | Total p95 (ms) | Peak RSS (MB) |
|---|---|---:|---:|---:|---:|
| B01 | calibration 460 | 150.9 | 148.2 | 180.7 | 392.9 |
| B03 | calibration 460 | 8409.2 | 5406.4 | 7198.4 | 361.0 |
| H01 | calibration 460 | 4986.5 | 456.2 | 13970.8 | 498.8 |
| H00 | calibration 20 (ref) | 12011.9 | 12052.1 | 12525.7 | 488.0 |

*% CPU float32, batch 1, PyTorch backend (Step 9 profiler, 14 stages). H01 and H00 use different sample counts; no paired speedup claim. H01 bimodal: p50 early-exit vs p95 AD-routed. Source: artifacts/runtime/*/profile.json.*
