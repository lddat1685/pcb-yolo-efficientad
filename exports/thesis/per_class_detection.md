# per_class_detection

| Model | Class | GT boxes | Detected | AP@0.5 | Recall |
|---|---|---:|---:|---:|---:|
| B01 | `open_circuit` | 298 | 406 | 0.8175 | 87.58% |
| B01 | `short` | 215 | 237 | 0.8029 | 79.07% |
| B01 | `mouse_bite` | 250 | 287 | 0.7529 | 76.80% |
| B01 | `spur` | 228 | 217 | 0.7358 | 71.49% |
| B01 | `spurious_copper` | 194 | 225 | 0.8806 | 90.72% |
| B01 | `pin_hole` | 222 | 415 | 0.7680 | 90.54% |
| B02 | `open_circuit` | 298 | 345 | 0.8764 | 92.28% |
| B02 | `short` | 215 | 195 | 0.8740 | 81.86% |
| B02 | `mouse_bite` | 250 | 242 | 0.8465 | 82.80% |
| B02 | `spur` | 228 | 190 | 0.7578 | 71.49% |
| B02 | `spurious_copper` | 194 | 169 | 0.8974 | 84.54% |
| B02 | `pin_hole` | 222 | 647 | 0.7825 | 97.30% |
| H01 | `open_circuit` | 298 | 406 | 0.8175 | 87.58% |
| H01 | `short` | 215 | 237 | 0.8029 | 79.07% |
| H01 | `mouse_bite` | 250 | 287 | 0.7529 | 76.80% |
| H01 | `spur` | 228 | 217 | 0.7358 | 71.49% |
| H01 | `spurious_copper` | 194 | 225 | 0.8806 | 90.72% |
| H01 | `pin_hole` | 222 | 415 | 0.7680 | 90.54% |

*% Per-class detection on test (box-level; H01 inherits B01 boxes). Source: artifacts/test/benchmark_per_class.csv.*
