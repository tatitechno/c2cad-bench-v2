import argparse
import json
import math

def generate_dna(base_pairs):
    shapes = []
    pid = 0
    R = 20.0 
    z_step = 4.0
    angle_step = 36.0 
    sphere_radius = 3.0
    
    for i in range(base_pairs):
        theta1 = math.radians(i * angle_step)
        theta2 = math.radians(i * angle_step + 180.0)
        z = i * z_step
        
        x1 = R * math.cos(theta1)
        y1 = R * math.sin(theta1)
        
        x2 = R * math.cos(theta2)
        y2 = R * math.sin(theta2)
        
        # Sphere 1
        shapes.append({
            "id": pid, "type": "sphere", "radius": sphere_radius,
            "center": [float(x1), float(y1), float(z)]
        })
        pid += 1
        
        # Sphere 2
        shapes.append({
            "id": pid, "type": "sphere", "radius": sphere_radius,
            "center": [float(x2), float(y2), float(z)]
        })
        pid += 1
        
        # Rung
        shapes.append({
            "id": pid, "type": "beam", "width": 2.0, "height": 2.0,
            "start": [float(x1), float(y1), float(z)],
            "end": [float(x2), float(y2), float(z)]
        })
        pid += 1

    turns_per_revolution = round(360.0 / angle_step)
    prompt = f"""Model a right-handed double helix of {base_pairs} base pairs about
the Z-axis through the origin. The backbone locus has radius 20. Each level
contains diametrically opposite sphere nodes of radius 3 joined by a beam of
square section 2 at their centers. The first pair lies in the XY plane with
the first backbone node on positive X. Successive levels rise by 4, winding
counterclockwise when viewed from positive Z, with 10 levels per revolution.
Index upward, first backbone node, opposite node, then bond at each level.

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
    parser.add_argument("--scale", type=int, default=10, help="Number of base pairs in the helix")
    args = parser.parse_args()
    
    prompt, shapes = generate_dna(args.scale)
    
    out_file = "dna_golden.json"
    with open(out_file, "w") as f:
        json.dump(shapes, f, indent=2)
        
    print("="*60)
    print(f"PROMPT FOR DNA HELIX (SCALE/BASE_PAIRS: {args.scale}):")
    print("="*60)
    print(prompt)
    print("\nNote: End your prompt instructing the AI to strictly output only the raw JSON format for C2CAD-Bench.")
    print("="*60)
    print(f"Golden JSON exactly representing this constraint saved to {out_file} ({len(shapes)} parts).")
