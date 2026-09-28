def generate_truss(scale):
    """
    Cross-Braced Truss Tower (Difficulty Level 3+ of Phase 2)
    Scale defines the number of stories in the tower.
    """
    stories = scale
    
    prompt = f"""Model a tower with {scale} stories, each of height 10,
and a square centerline footprint of side 10. The footprint midpoint is the
origin in the XY ground plane, with edges parallel to X and Y. Each story has
vertical beam columns at its corners and X-shaped diagonal bracing across
each exterior face. Beam ends coincide at story-corner joints. Every beam has
square section 1. Index stories upward, columns around the square from its
negative-X, negative-Y corner toward positive X, followed by braces around
the faces in that same order; the rising diagonal precedes the falling one.

Use millimetres and the canonical box, cylinder, sphere, cone, torus, pipe,
beam schema. For centered primitives, center denotes the geometric centroid;
beams use start and end centerline endpoints. Supply unit axial directions,
and give box size in world X, Y, Z order. Preserve the specified component IDs;
otherwise assign unique integer IDs. Return only valid JSON: an array of
primitive objects, or an object with the single key shapes containing that
array, without markdown or explanation."""

    shapes = []
    sid = 1
    
    corners = [
        [-5, -5],
        [5, -5],
        [5, 5],
        [-5, 5]
    ]
    
    faces = [
        ([0, 1]), # Front Y=-5
        ([1, 2]), # Right X=5
        ([2, 3]), # Back Y=5
        ([3, 0]), # Left X=-5
    ]
    
    for z_idx in range(stories):
        z_base = z_idx * 10
        z_top = z_base + 10
        
        # Pillars
        for c in corners:
            shapes.append({
                "id": sid, "type": "beam",
                "start": [c[0], c[1], z_base],
                "end": [c[0], c[1], z_top],
                "width": 1, "height": 1
            })
            sid += 1
            
        # Faces diagonals
        for f in faces:
            c1 = corners[f[0]]
            c2 = corners[f[1]]
            
            # Diagonal 1
            shapes.append({
                "id": sid, "type": "beam",
                "start": [c1[0], c1[1], z_base],
                "end": [c2[0], c2[1], z_top],
                "width": 1, "height": 1
            })
            sid += 1
            
            # Diagonal 2
            shapes.append({
                "id": sid, "type": "beam",
                "start": [c2[0], c2[1], z_base],
                "end": [c1[0], c1[1], z_top],
                "width": 1, "height": 1
            })
            sid += 1
            
    return prompt, shapes
