import math

def generate_fractal(scale):
    """
    Fractal Y-Branch Recursion (Difficulty Level 4+ of Phase 2)
    Scale defines the recursion depth.
    Level 0 = Trunk only. Level 1 = Trunk + 2 branches. Level 2 = Trunk + 2 + 4.
    """
    depth = scale
    
    prompt = f"""Model a symmetric binary branching beam tree in the XZ
plane with {scale} branching generations beyond its trunk. The trunk has length
100, starts at the origin, and points along positive Z. All beams have square
section 1. At each branch tip the children meet their parent's endpoint,
each child has half its parent's length, and the two children deviate by
45 degrees on opposite sides of the parent direction in the branching plane.
For IDs use depth-first order, trunk first, the child turned toward positive X
relative to the upward trunk before its opposite sibling.

Use millimetres and the canonical box, cylinder, sphere, cone, torus, pipe,
beam schema. For centered primitives, center denotes the geometric centroid;
beams use start and end centerline endpoints. Supply unit axial directions,
and give box size in world X, Y, Z order. Preserve the specified component IDs;
otherwise assign unique integer IDs. Return only valid JSON: an array of
primitive objects, or an object with the single key shapes containing that
array, without markdown or explanation."""

    shapes = []
    sid = 1
    
    # Internal recursion
    def recurse(start_pt, angle_deg, length, current_level):
        nonlocal sid
        
        angle_rad = math.radians(angle_deg)
        dx = length * math.sin(angle_rad)
        dz = length * math.cos(angle_rad)
        
        end_pt = [
            round(start_pt[0] + dx, 4),
            0.0,
            round(start_pt[2] + dz, 4)
        ]
        
        shapes.append({
            "id": sid, "type": "beam",
            "start": start_pt, "end": end_pt,
            "width": 1, "height": 1
        })
        sid += 1
        
        if current_level < depth:
            child_len = length / 2.0
            recurse(end_pt, angle_deg + 45.0, child_len, current_level + 1)
            recurse(end_pt, angle_deg - 45.0, child_len, current_level + 1)

    recurse([0.0, 0.0, 0.0], 0.0, 100.0, 0)
    
    return prompt, shapes
