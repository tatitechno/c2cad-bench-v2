def _reference_solution(wall):
    """Return a canonical valid enclosure for the visualizer."""
    OW, OD, OH = 80.0, 50.0, 40.0          # outer casing dims
    IW = OW - wall * 2                       # inner cavity
    ID_ = OD - wall * 2
    IH = OH - wall                           # open top
    clearance = 0.2
    lid_H = 8.0
    lip_W = IW - clearance * 2
    lip_D = ID_ - clearance * 2
    lip_H = 5.0
    return [
        {"id": 0, "type": "box", "center": [0.0, 0.0, OH / 2.0], "size": [OW, OD, OH]},
        {"id": 1, "type": "box", "center": [0.0, 0.0, (IH / 2.0) + wall], "size": [IW, ID_, IH]},
        {"id": 2, "type": "box", "center": [0.0, 0.0, OH + lid_H / 2.0], "size": [OW, OD, lid_H]},
        {"id": 3, "type": "box", "center": [0.0, 0.0, OH - lip_H / 2.0 + 0.5], "size": [lip_W, lip_D, lip_H]},
    ]

def generate_enclosure(scale):
    wall = scale # scale is effectively the requested wall thickness here.
    clearance = 0.2
    prompt = f"""Design a hollow electronics casing with perfectly fitting lid. You can freely choose the absolute dimensional scales.
You must use perfectly exactly these IDs:
- Outer Casing: Box, ID=0
- Subtraction Tool to hollow the casing: Box, ID=1 (Use a subtract op where target=0, tool=1)
- Lid: Box, ID=2
- Lid Inner Lip: Box, ID=3

Semantic Constraints:
1. The minimum volumetric wall thickness of the hollow casing must be exactly {wall}.0 mm globally (this implies calculating the specific bounding size difference between the Outer Casing and Subtraction Tool).
2. The Lid Inner Lip (ID 3) must slide and fit perfectly inside the casing's cut hollow opening (ID 1) with exactly {clearance} mm of radial clearance gap on all sides.
Output only the raw JSON array of geometric elements.
"""
    specs = {
        "wall_thickness": [{"outer_id": 0, "inner_id": 1, "min_wall_mm": float(wall)}],
        "clearance_fit": [{"shaft_id": 3, "hole_id": 1, "expected_clearance": float(clearance), "tol": 0.05}],
        "reference": _reference_solution(wall)
    }
    return prompt, specs
