def _reference_solution(scale):
    """Return a canonical valid table with (scale*2) legs for the visualizer."""
    legs = scale * 2 if scale else 4
    W, D = 100.0, 60.0           # tabletop width / depth
    T = 5.0                       # tabletop thickness
    LH = 70.0                     # leg height
    LR = 3.0                      # leg radius
    table_z = LH + T / 2.0       # top-face centre Z
    shapes = [
        {"id": 0, "type": "box", "center": [0.0, 0.0, table_z], "size": [W, D, T]}
    ]
    # Place legs at the four corners; if more than 4, add along the long edges
    import math
    corners = [
        ( W / 2 - LR * 2,  D / 2 - LR * 2),
        (-W / 2 + LR * 2,  D / 2 - LR * 2),
        (-W / 2 + LR * 2, -D / 2 + LR * 2),
        ( W / 2 - LR * 2, -D / 2 + LR * 2),
    ]
    if legs > 4:
        # Add mid-edge legs along the two long sides
        for k in range(1, (legs - 4) // 2 + 1):
            t = k / ((legs - 4) // 2 + 1)
            xm = W / 2 - LR * 2 - t * (W - 4 * LR * 2)
            corners += [(xm, D / 2 - LR * 2), (xm, -D / 2 + LR * 2)]
    for i in range(legs):
        rx, ry = corners[i] if i < len(corners) else corners[i % 4]
        shapes.append({
            "id": i + 1,
            "type": "cylinder",
            "center": [round(rx, 2), round(ry, 2), LH / 2.0],
            "radius": LR,
            "height": LH,
            "axis": [0, 0, 1]
        })
    return shapes

def generate_furniture(scale):
    legs = scale * 2 if scale else 4
    prompt = f"""Model a table with {legs} cylinder legs of radius 3
and height 70, supporting a box tabletop of X length 100, Y depth 60 and
thickness 5. The origin is on the ground below the tabletop midpoint; Z is up.
Legs stand vertically on the XY ground plane and their top faces touch the
tabletop underside. The corner-leg axes are inset from each adjacent tabletop
edge by twice a leg radius. Extra legs, if needed, occur in pairs mirrored across
the XZ plane, along the two long edges. Their stations divide an interior
support zone into equal intervals, excluding both boundary stations. The
positive-X boundary is the positive-X corner-leg axis; the negative-X boundary
is inset six leg radii from the tabletop's negative-X edge. ID 0 is the tabletop. Index corner legs starting in the
positive-X, positive-Y corner, then negative-X positive-Y, negative-X negative-Y,
positive-X negative-Y. Index extra pairs from positive X toward negative X,
positive-Y leg before its negative-Y partner.

Use millimetres and the canonical box, cylinder, sphere, cone, torus, pipe,
beam schema. For centered primitives, center denotes the geometric centroid;
beams use start and end centerline endpoints. Supply unit axial directions,
and give box size in world X, Y, Z order. Preserve the specified component IDs;
otherwise assign unique integer IDs. Return only valid JSON: an array of
primitive objects, or an object with the single key shapes containing that
array, without markdown or explanation."""
    specs = {
        "gravity_check": True,
        "mates": [ {"type": "coincident", "ids": [i, 0], "face_a": "top", "face_b": "bottom"} for i in range(1, legs+1) ],
        "reference": _reference_solution(scale)
    }
    return prompt, specs
