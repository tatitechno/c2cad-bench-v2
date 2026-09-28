# E08. CAD-kernel round trip on every reference

Script: `v2/experiments/e08_kernel_roundtrip.py` (CadQuery 2.8 / OpenCascade in `v2/.venv-cad`).

- primitives converted: 5042; valid OpenCascade solids: 5042; recovered as a primitive: 5042 (unrecognised 0); of which label-free rectangular prisms: 2502
- by type: beam 2265, box 237, cone 446, cylinder 689, pipe 160, sphere 1160, torus 85
- full cadquery-arm path (program -> sandbox -> kernel -> recovery -> evaluator): 75 / 75 cases exact; lowest Geometry 99.999998 (armillary_sphere_level_3)

A model's CadQuery program is therefore scored on exactly the same footing as its JSON: a correct program scores 100, and nothing is lost in conversion.
