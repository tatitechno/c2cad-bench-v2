Case-level Spearman correlation (n = 897 outputs). Chamfer and orientation error are sign-flipped so that higher = better.

| | iou | f2 | chamfer | orient_err |
|---|---|---|---|---|
| coverage | 0.470 | 0.456 | 0.340 | 0.121 |
| geometry | 0.812 | 0.821 | 0.755 | 0.481 |
| geometry_equiv | 0.809 | 0.819 | 0.753 | 0.455 |
| semantic | 0.810 | 0.767 | 0.695 | 0.461 |
| global_v2 | 0.803 | 0.790 | 0.713 | 0.453 |

Model-level rank agreement with v2 Global (Spearman over 13 model means):

- iou: rho = 0.945
- f2: rho = 0.978
- chamfer: rho = 0.923
- orient_err: rho = 0.687

| model | Global v2 | Geom | Sem | IoU | F@2% | Chamfer | orient err (deg) |
|---|---|---|---|---|---|---|---|
| gemini-3.1-pro-preview | 83.6 | 80.9 | 77.7 | 0.806 | 0.916 | 0.008 | 4.7 |
| gpt-5.4 | 81.8 | 80.1 | 74.8 | 0.726 | 0.872 | 0.012 | 2.6 |
| gemini-2.5-pro | 79.8 | 73.0 | 74.7 | 0.742 | 0.894 | 0.010 | 10.3 |
| gemini-3-flash-preview | 71.2 | 66.3 | 71.6 | 0.777 | 0.856 | 0.013 | 6.6 |
| claude-opus-4-6 | 70.8 | 60.0 | 64.8 | 0.657 | 0.815 | 0.019 | 14.9 |
| gemini-2.5-flash | 70.6 | 60.7 | 64.9 | 0.676 | 0.833 | 0.014 | 14.4 |
| gpt-4.1 | 70.1 | 56.6 | 67.2 | 0.631 | 0.808 | 0.018 | 10.5 |
| gpt-5.4-mini | 66.2 | 54.1 | 62.2 | 0.561 | 0.771 | 0.023 | 10.9 |
| claude-sonnet-4-6 | 64.0 | 52.0 | 59.0 | 0.536 | 0.703 | 0.021 | 13.5 |
| deepseek-reasoner | 60.1 | 53.4 | 55.1 | 0.555 | 0.712 | 0.019 | 12.2 |
| kimi-k2.5 | 57.3 | 43.7 | 53.0 | 0.491 | 0.668 | 0.035 | 12.9 |
| gemini-3.1-flash-lite-preview | 57.2 | 42.3 | 54.7 | 0.537 | 0.683 | 0.026 | 13.0 |
| deepseek-chat | 55.8 | 33.0 | 52.5 | 0.409 | 0.618 | 0.036 | 18.5 |

High IoU (>= 0.8) but Sem < 60: 54 / 897 (6.0%)
Low IoU (< 0.3) but Geom >= 80: 21 / 897 (2.3%)
F@2% >= 0.9 but not exact (some stated constraint violated): 278 / 897 (31.0%)

Examples (high IoU, low Sem):
- kimi-k2.5 / clock_tower_mechanism_level_3: IoU 0.92, F@2% 1.00, Geom 6.2, Sem 18.8
- kimi-k2.5 / clock_tower_mechanism_level_2: IoU 0.92, F@2% 1.00, Geom 10.0, Sem 19.1
- kimi-k2.5 / clock_tower_mechanism_level_1: IoU 0.92, F@2% 1.00, Geom 16.7, Sem 19.7
- gemini-3.1-flash-lite-preview / compound_eye_level_3: IoU 0.98, F@2% 0.85, Geom 1.6, Sem 27.4
- gemini-3.1-flash-lite-preview / compound_eye_level_2: IoU 0.99, F@2% 0.90, Geom 2.6, Sem 27.5
- gpt-5.4-mini / compound_eye_level_3: IoU 0.98, F@2% 0.84, Geom 2.1, Sem 30.7
- kimi-k2.5 / compound_eye_level_1: IoU 0.99, F@2% 0.93, Geom 23.9, Sem 31.3
- kimi-k2.5 / compound_eye_level_2: IoU 0.99, F@2% 0.88, Geom 8.6, Sem 32.4

Examples (low IoU, high Geom):
- claude-sonnet-4-6 / ball_bearing_assembly_level_2: IoU 0.09, F@2% 0.34, Geom 85.7, Sem 57.0
- gpt-5.4-mini / ball_bearing_assembly_level_2: IoU 0.09, F@2% 0.34, Geom 85.7, Sem 57.0
- deepseek-chat / ball_bearing_assembly_level_2: IoU 0.09, F@2% 0.34, Geom 85.7, Sem 57.0
- kimi-k2.5 / ball_bearing_assembly_level_2: IoU 0.09, F@2% 0.34, Geom 85.7, Sem 57.0
- gpt-5.4 / ball_bearing_assembly_level_2: IoU 0.09, F@2% 0.34, Geom 85.7, Sem 57.0
- gpt-4.1 / spiral_staircase_level_3: IoU 0.14, F@2% 0.90, Geom 81.2, Sem 29.6
- deepseek-reasoner / ball_bearing_assembly_level_3: IoU 0.15, F@2% 0.48, Geom 90.9, Sem 58.9
- claude-sonnet-4-6 / ball_bearing_assembly_level_3: IoU 0.15, F@2% 0.48, Geom 90.9, Sem 58.9
