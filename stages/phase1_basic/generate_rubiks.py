import argparse
import json

def generate_rubiks(scale):
    shapes = []
    pid = 0
    box_size = 10.0
    gap = 2.0
    pitch = box_size + gap
    
    for z_idx in range(scale):
        z = (box_size/2.0) + (z_idx * pitch)
        for y_idx in range(scale):
            y = (box_size/2.0) + (y_idx * pitch)
            for x_idx in range(scale):
                x = (box_size/2.0) + (x_idx * pitch)
                
                shapes.append({
                    "id": pid,
                    "type": "box",
                    "center": [float(x), float(y), float(z)],
                    "size": [box_size, box_size, box_size]
                })
                pid += 1

    prompt = f"""Model a cubic grid with {scale} boxes along each world axis.
Each box is a cube of side 10. Facing surfaces of neighboring boxes have a gap
of 2. The minimum outer corner of the entire grid is the origin and the grid
extends along positive X, Y, Z. Index X fastest, then Y, then Z.

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
    parser.add_argument("--scale", type=int, default=3, help="Grid dimension (e.g. 3 for a 3x3x3 grid)")
    args = parser.parse_args()
    
    prompt, shapes = generate_rubiks(args.scale)
    
    out_file = "rubiks_golden.json"
    with open(out_file, "w") as f:
        json.dump(shapes, f, indent=2)
        
    print("="*60)
    print(f"PROMPT FOR VOXEL GRID (SCALE: {args.scale}x{args.scale}x{args.scale}):")
    print("="*60)
    print(prompt)
    print("\nNote: End your prompt instructing the AI to strictly output only the raw JSON format for C2CAD-Bench.")
    print("="*60)
    print(f"Golden JSON exactly representing this constraint saved to {out_file} ({len(shapes)} parts).")
