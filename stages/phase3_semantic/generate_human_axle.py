def _reference_solution(clearance_block):
    """Return a canonical valid axle assembly for the visualizer."""
    BW, BD, BH = 80.0, 50.0, 50.0      # support block
    shaft_r = 8.0
    bore_r = shaft_r + clearance_block  # bore slightly larger
    bearing_inner_r = shaft_r           # press-fit on shaft
    bearing_outer_r = shaft_r + 6.0
    bearing_H = 12.0
    shaft_len = BW + 20.0
    return [
        {"id": 0, "type": "box", "center": [0.0, 0.0, BH / 2.0], "size": [BW, BD, BH]},
        {"id": 1, "type": "cylinder", "center": [0.0, 0.0, BH / 2.0], "radius": bore_r, "height": BW + 2, "axis": [1, 0, 0]},
        {"id": 2, "type": "cylinder", "center": [0.0, 0.0, BH / 2.0], "radius": shaft_r, "height": shaft_len, "axis": [1, 0, 0]},
        {"id": 3, "type": "pipe", "center": [-(BW / 2.0 - bearing_H / 2.0), 0.0, BH / 2.0], "inner_radius": bearing_inner_r, "outer_radius": bearing_outer_r, "height": bearing_H, "axis": [1, 0, 0]},
        {"id": 4, "type": "pipe", "center": [ (BW / 2.0 - bearing_H / 2.0), 0.0, BH / 2.0], "inner_radius": bearing_inner_r, "outer_radius": bearing_outer_r, "height": bearing_H, "axis": [1, 0, 0]},
    ]

def generate_axle(scale):
    clearance_block = float(scale) * 0.5 # 0.5, 1.0, 1.5
    prompt = f"""Model an axle assembly with radial block-bore clearance
{clearance_block}. A box support block has X width 80, Y depth 50 and Z height 50.
It stands on the XY ground plane with its footprint midpoint at the origin.
The horizontal bore passes through the block centroid parallel to X. Represent
its cut envelope as cylinder ID 1, extending 1 beyond each block end face.
The centered cylinder shaft, ID 2, has radius 8 and protrudes 10 beyond each
block end face. Derive the bore radius from the specified clearance.
Pipe bearings each have axial width 12 and radial wall thickness 6; their
bores fit the shaft with zero radial clearance. Bearings lie inside the block,
with their outer end faces flush with its opposite X end faces, and are
concentric with the shaft. ID 0 is the block; IDs 3 and 4 are negative-X and
positive-X bearings. The bore cylinder is a symbolic cut envelope; return
primitive geometry only, without a subtraction operation record.

Use millimetres and the canonical box, cylinder, sphere, cone, torus, pipe,
beam schema. For centered primitives, center denotes the geometric centroid;
beams use start and end centerline endpoints. Supply unit axial directions,
and give box size in world X, Y, Z order. Preserve the specified component IDs;
otherwise assign unique integer IDs. Return only valid JSON: an array of
primitive objects, or an object with the single key shapes containing that
array, without markdown or explanation."""
    specs = {
        "interference_check": True,
        "clearance_fit": [
            {"shaft_id": 2, "hole_id": 1, "expected_clearance": clearance_block, "tol": 0.05},
            {"shaft_id": 2, "hole_id": 3, "expected_clearance": 0.0, "tol": 0.05},
            {"shaft_id": 2, "hole_id": 4, "expected_clearance": 0.0, "tol": 0.05}
        ],
        "mates": [
            {"type": "concentric", "ids": [2, 3]},
            {"type": "concentric", "ids": [2, 4]}
        ],
        "reference": _reference_solution(clearance_block)
    }
    return prompt, specs
