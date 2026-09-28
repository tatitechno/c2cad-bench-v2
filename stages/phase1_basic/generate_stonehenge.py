import argparse
import json
import math

def generate_stonehenge(num_arches):
    shapes = []
    pid = 0
    R = 80.0
    pillar_radius = 4.0
    pillar_height = 30.0
    
    dTheta_deg = 3.0 
    arch_spacing_deg = 360.0 / num_arches
    
    for i in range(num_arches):
        theta = i * arch_spacing_deg
        
        # Pillar 1
        t1 = math.radians(theta - dTheta_deg)
        x1 = R * math.cos(t1)
        y1 = R * math.sin(t1)
        z_cyl = pillar_height / 2.0
        shapes.append({
            "id": pid, "type": "cylinder", "radius": pillar_radius, "height": pillar_height,
            "center": [float(x1), float(y1), float(z_cyl)], "axis": [0.0, 0.0, 1.0]
        })
        pid += 1
        
        # Pillar 2
        t2 = math.radians(theta + dTheta_deg)
        x2 = R * math.cos(t2)
        y2 = R * math.sin(t2)
        shapes.append({
            "id": pid, "type": "cylinder", "radius": pillar_radius, "height": pillar_height,
            "center": [float(x2), float(y2), float(z_cyl)], "axis": [0.0, 0.0, 1.0]
        })
        pid += 1
        
        # Header beam connecting top centers
        shapes.append({
            "id": pid, "type": "beam", "width": pillar_radius * 2, "height": pillar_radius * 1.5,
            "start": [float(x1), float(y1), float(pillar_height)],
            "end": [float(x2), float(y2), float(pillar_height)]
        })
        pid += 1

    prompt = f"""Model {num_arches} equally spaced archways around a circular
monument centered at the origin in the XY ground plane. Pillar axes lie on a
circle of radius 80. Each arch has a symmetric pair of vertical cylinder pillars,
radius 4 and height 30; the angle subtended by the pair at the monument center
is 6 degrees. Both pillars stand on the ground. A beam lintel's centerline joins the
centers of the two pillar top faces. Its width equals a pillar
diameter and its thickness is 1.5 times the pillar radius. The first arch's radial
bisector points along positive X; arches follow counterclockwise when viewed
from positive Z. Within each arch index the clockwise pillar, counterclockwise
pillar, then lintel.

Use millimetres and the canonical box, cylinder, sphere, cone, torus, pipe,
beam schema. For centered primitives, center denotes the geometric centroid;
beams use start and end centerline endpoints. Supply unit axial directions,
and give box size in world X, Y, Z order. Preserve the specified component IDs;
otherwise assign unique integer IDs. Return only valid JSON: an array of
primitive objects, or an object with the single key shapes containing that
array, without markdown or explanation."""
    return prompt, shapes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=int, default=12, help="Number of arches in the ring")
    args = parser.parse_args()
    
    prompt, shapes = generate_stonehenge(args.scale)
    
    out_file = "stonehenge_golden.json"
    with open(out_file, "w") as f:
        json.dump(shapes, f, indent=2)
        
    print("="*60)
    print(f"PROMPT FOR STONEHENGE RING (SCALE/ARCHES: {args.scale}):")
    print("="*60)
    print(prompt)
    print("\nNote: End your prompt instructing the AI to strictly output only the raw JSON format for C2CAD-Bench.")
    print("="*60)
    print(f"Golden JSON exactly representing this constraint saved to {out_file} ({len(shapes)} parts).")
