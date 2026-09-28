import argparse
import json
import math

def generate_pyramid(layers):
    R = 10.0
    shapes = []
    pid = 0
    
    dz = 2.0 * R * math.sqrt(2.0/3.0)
    
    for layer in range(layers):
        side = layers - layer
        z = R + layer * dz
        x0 = layer * R
        y0 = layer * R / math.sqrt(3.0)
        
        for row in range(side):
            y = y0 + row * R * math.sqrt(3.0)
            for col in range(side - row):
                x = x0 + row * R + col * 2.0 * R
                
                shapes.append({
                    "id": pid,
                    "type": "sphere",
                    "center": [float(x), float(y), float(z)],
                    "radius": R
                })
                pid += 1

    prompt = f"""Model a {layers}-layer tetrahedral cannonball pyramid
of tangent spheres of radius 10. Its lowest layer rests on the XY ground plane.
The vertical projection of one corner sphere center is the origin; the base
edge from that corner follows positive X and the triangle extends toward positive Y.
Each upper layer occupies the triangular pockets supported by three mutually
tangent spheres below, with one fewer sphere along each edge than the layer below.
Use the pocket orientation toward the interior of the base triangle.
Index from the bottom upward, within each layer in rows toward positive Y,
and within each row toward positive X.

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
    parser.add_argument("--scale", type=int, default=4, help="Number of layers in the pyramid")
    args = parser.parse_args()
    
    prompt, shapes = generate_pyramid(args.scale)
    
    out_file = "pyramid_golden.json"
    with open(out_file, "w") as f:
        json.dump(shapes, f, indent=2)
        
    print("="*60)
    print(f"PROMPT FOR CANNONBALL PYRAMID (SCALE/LAYERS: {args.scale}):")
    print("="*60)
    print(prompt)
    print("\nNote: End your prompt instructing the AI to strictly output only the raw JSON format for C2CAD-Bench.")
    print("="*60)
    print(f"Golden JSON exactly representing this constraint saved to {out_file} ({len(shapes)} parts).")
