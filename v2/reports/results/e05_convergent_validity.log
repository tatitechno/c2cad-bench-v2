Case-level Spearman correlation (n = 897 outputs). Chamfer and orientation error are sign-flipped so that higher = better.

| | iou | f2 | chamfer | orient_err |
|---|---|---|---|---|
| coverage | 0.484 | 0.465 | 0.361 | 0.152 |
| geometry | 0.807 | 0.816 | 0.753 | 0.463 |
| geometry_equiv | 0.804 | 0.815 | 0.752 | 0.436 |
| semantic | 0.823 | 0.777 | 0.703 | 0.464 |
| global_v2 | 0.805 | 0.789 | 0.715 | 0.444 |

Model-level rank agreement with v2 Global (Spearman over 13 model means):

- iou: rho = 0.896
- f2: rho = 0.951
- chamfer: rho = 0.918
- orient_err: rho = 0.736

| model | Global v2 | Geom | Sem | IoU | F@2% | Chamfer | orient err (deg) |
|---|---|---|---|---|---|---|---|
| gemini-3.1-pro-preview | 84.1 | 81.4 | 78.0 | 0.807 | 0.916 | 0.008 | 4.4 |
| gpt-5.4 | 82.2 | 80.2 | 75.9 | 0.726 | 0.872 | 0.012 | 2.6 |
| gemini-2.5-pro | 81.5 | 75.8 | 77.0 | 0.767 | 0.907 | 0.009 | 7.4 |
| claude-opus-4-6 | 72.1 | 61.6 | 67.0 | 0.668 | 0.821 | 0.019 | 12.1 |
| gemini-2.5-flash | 71.6 | 62.0 | 66.7 | 0.689 | 0.841 | 0.013 | 12.9 |
| gemini-3-flash-preview | 71.6 | 66.9 | 72.1 | 0.780 | 0.860 | 0.013 | 5.6 |
| gpt-4.1 | 70.4 | 56.7 | 67.8 | 0.631 | 0.808 | 0.018 | 10.4 |
| gpt-5.4-mini | 66.6 | 54.1 | 63.2 | 0.561 | 0.771 | 0.023 | 10.9 |
| claude-sonnet-4-6 | 64.4 | 52.1 | 60.0 | 0.536 | 0.703 | 0.021 | 13.5 |
| deepseek-reasoner | 62.6 | 55.5 | 57.6 | 0.555 | 0.712 | 0.019 | 11.7 |
| deepseek-chat | 58.5 | 33.8 | 55.5 | 0.409 | 0.619 | 0.036 | 17.9 |
| gemini-3.1-flash-lite-preview | 58.0 | 42.9 | 56.5 | 0.548 | 0.688 | 0.025 | 12.2 |
| kimi-k2.5 | 57.7 | 43.7 | 54.1 | 0.491 | 0.668 | 0.035 | 13.2 |

High IoU (>= 0.8) but Sem < 60: 54 / 897 (6.0%)
Low IoU (< 0.3) but Geom >= 80: 21 / 897 (2.3%)
F@2% >= 0.9 but not exact (some stated constraint violated): 284 / 897 (31.7%)

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
