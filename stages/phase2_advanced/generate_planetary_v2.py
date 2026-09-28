import math

def generate_planetary(scale):
    """
    Planetary Gear Array (Difficulty Level 2+ of Phase 2)
    Scale defines the number of planet gears.
    """
    n_planets = scale
    
    prompt = f"""Model a central sun cylinder of radius 10 and height 2
with {scale} planet cylinders of radius 3 and height 2. The sun center is the
origin and all axes are parallel to Z. All centers lie in the XY plane.
Each planet is externally tangent to the sun's cylindrical surface. Planets
are equally spaced angularly; the first lies on positive X, followed
counterclockwise when viewed from positive Z. Index the sun before the planets.
Only sun-to-planet tangency is prescribed; the design count is fixed.

Use millimetres and the canonical box, cylinder, sphere, cone, torus, pipe,
beam schema. For centered primitives, center denotes the geometric centroid;
beams use start and end centerline endpoints. Supply unit axial directions,
and give box size in world X, Y, Z order. Preserve the specified component IDs;
otherwise assign unique integer IDs. Return only valid JSON: an array of
primitive objects, or an object with the single key shapes containing that
array, without markdown or explanation."""

    shapes = []
    shapes.append({
        "id": 0, "type": "cylinder", 
        "center": [0,0,0], "radius": 10, "height": 2, "axis": [0,0,1]
    })
    
    sid = 1
    dist = 13.0
    for i in range(n_planets):
        angle = (2 * math.pi * i) / n_planets
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        shapes.append({
            "id": sid, "type": "cylinder", 
            "center": [round(x, 4), round(y, 4), 0], 
            "radius": 3, "height": 2, "axis": [0,0,1]
        })
        sid += 1

    return prompt, shapes
