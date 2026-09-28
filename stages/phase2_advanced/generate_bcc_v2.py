def generate_bcc(scale):
    """
    Crystalline BCC Lattice Array (Difficulty Level 5+ of Phase 2)
    Scale defines N in an NxNxN array.
    """
    n = scale
    
    prompt = f"""Model a body-centered cubic lattice with {scale} unit cells
along each world axis. Unit-cell side length is 10. The minimum lattice corner
is the origin and the lattice extends along positive X, Y, Z. Corner and body
nodes are spheres of radius 1, with shared corner nodes represented once.
Beam struts of square section 1 join each body-node center to the centers of
its unit cell's corner nodes. Include only these BCC links, without exterior
edge struts. IDs may follow any unique sequential order.

Use millimetres and the canonical box, cylinder, sphere, cone, torus, pipe,
beam schema. For centered primitives, center denotes the geometric centroid;
beams use start and end centerline endpoints. Supply unit axial directions,
and give box size in world X, Y, Z order. Preserve the specified component IDs;
otherwise assign unique integer IDs. Return only valid JSON: an array of
primitive objects, or an object with the single key shapes containing that
array, without markdown or explanation."""

    shapes = []
    sid = 1
    
    # Track corner positions to deduplicate
    corners_set = set()
    centers = []
    
    # 1. Centers and Beams logic
    for x in range(n):
        for y in range(n):
            for z in range(n):
                cx = x * 10 + 5
                cy = y * 10 + 5
                cz = z * 10 + 5
                centers.append((cx, cy, cz))
                
                # Register the 8 corners for this unit cell
                cell_corners = [
                    (x*10, y*10, z*10), (x*10+10, y*10, z*10),
                    (x*10, y*10+10, z*10), (x*10+10, y*10+10, z*10),
                    (x*10, y*10, z*10+10), (x*10+10, y*10, z*10+10),
                    (x*10, y*10+10, z*10+10), (x*10+10, y*10+10, z*10+10)
                ]
                
                for corner in cell_corners:
                    corners_set.add(corner)
                    # Beams mapping center to corner
                    shapes.append({
                        "id": sid, "type": "beam",
                        "start": [cx, cy, cz],
                        "end": [corner[0], corner[1], corner[2]],
                        "width": 1, "height": 1
                    })
                    sid += 1

    # 2. Add Deduplicated Corner Spheres
    for corner in sorted(list(corners_set)):
        shapes.append({
            "id": sid, "type": "sphere",
            "center": [corner[0], corner[1], corner[2]],
            "radius": 1
        })
        sid += 1
        
    # 3. Add Center Spheres
    for center in centers:
        shapes.append({
            "id": sid, "type": "sphere",
            "center": [center[0], center[1], center[2]],
            "radius": 1
        })
        sid += 1

    return prompt, shapes
