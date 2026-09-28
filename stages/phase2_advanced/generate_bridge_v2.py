import math

def generate_bridge(scale):
    """
    Suspension Bridge (Difficulty Level 1+ of Phase 2)
    Scale defines the number of cables per tower.
    """
    n_cables = scale
    
    prompt = f"""Model a symmetric cable bridge in the XZ plane.
Its deck is a beam of span 100 and square section 1, with its centerline 10
above the XY ground plane. The deck midpoint projects to the origin and its
length follows X. At each deck endpoint is a vertical cylinder tower of
radius 2 and height 100 standing on the ground.
For each tower, {scale} cable beams of square section 1 join its top-face
center to uniformly spaced attachment points on its half of the deck centerline.
The outermost attachment is 2 inward from the tower axis; the innermost is
2 from the deck midpoint. Include both limiting attachments. Index deck,
negative-X tower, positive-X tower, then cables from negative X to positive X
on the negative half followed by the positive half.

Use millimetres and the canonical box, cylinder, sphere, cone, torus, pipe,
beam schema. For centered primitives, center denotes the geometric centroid;
beams use start and end centerline endpoints. Supply unit axial directions,
and give box size in world X, Y, Z order. Preserve the specified component IDs;
otherwise assign unique integer IDs. Return only valid JSON: an array of
primitive objects, or an object with the single key shapes containing that
array, without markdown or explanation."""

    shapes = []
    
    # Towers
    shapes.append({"id": 1, "type": "cylinder", "center": [-50, 0, 50], "radius": 2, "height": 100, "axis": [0,0,1]})
    shapes.append({"id": 2, "type": "cylinder", "center": [50, 0, 50], "radius": 2, "height": 100, "axis": [0,0,1]})
    
    # Deck
    shapes.append({"id": 3, "type": "beam", "start": [-50, 0, 10], "end": [50, 0, 10], "width": 2, "height": 1})
    
    sid = 4
    # Left Cables
    if n_cables > 1:
        step = (46.0) / (n_cables - 1)
    else:
        step = 0
        
    for i in range(n_cables):
        x = -48.0 + (i * step)
        shapes.append({
            "id": sid, "type": "beam", 
            "start": [-50, 0, 100], "end": [x, 0, 10],
            "width": 1, "height": 1
        })
        sid += 1
        
    # Right Cables
    for i in range(n_cables):
        x = 2.0 + (i * step)
        shapes.append({
            "id": sid, "type": "beam", 
            "start": [50, 0, 100], "end": [x, 0, 10],
            "width": 1, "height": 1
        })
        sid += 1
        
    return prompt, shapes
