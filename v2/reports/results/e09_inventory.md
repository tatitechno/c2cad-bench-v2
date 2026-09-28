# E09. Benchmark inventory

Script: `v2/experiments/e09_inventory.py`.

| split | cases | families | parts min | median | max | total | constraints | references exact |
|---|---|---|---|---|---|---|---|---|
| main | 75 | 25 | 5 | 30 | 805 | 5042 | 25371 | 75/75 |
| sweep | 121 | 19 | 3 | 32 | 701 | 8024 | 48528 | 121/121 |
| heldout | 48 | 8 | 7 | 21 | 64 | 1252 | 7948 | 48/48 |

Constraints by kind (main split): anchor 551, dimension 12402, mate 5537, orientation 1561, pattern 4678, topology 642
Primitive types (main split): beam 2265, box 237, cone 446, cylinder 689, pipe 160, sphere 1160, torus 85

Prompt sentences (main split): 786; traced to constraint keys 642, to an id-ordering rule only 99, to a symbolic declaration only 33, to a note only 12; untraced 0 (enforced by tests).

Families with a CML gate program (16): Axle Bearing, BCC Lattice, Ball Bearing Assembly, Cannonball Pyramid, Clock Tower Mechanism, DNA Helix, Domino Ring, Flanged Pipe Joint, Fractal Y-Tree, Furniture Assembly, Gantry Crane Assembly, Pipe Manifold, Planetary Array, Spiral Staircase, Suspension Bridge, Voxel Grid
