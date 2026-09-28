import argparse
import json
import math
import os

def generate_staircase(steps):
    shapes = []
    
    pillar_radius = 5.0
    rise_per_step = 5.0
    total_height = steps * rise_per_step
    
    # 0 = Pillar
    shapes.append({
        "id": 0,
        "type": "cylinder",
        "center": [0.0, 0.0, total_height / 2.0],
        "radius": pillar_radius,
        "height": total_height,
        "axis": [0.0, 0.0, 1.0]
    })
    
    step_length = 20.0
    step_width = 5.0
    step_height = 2.0
    angle_increment = 15.0 # degrees

    for i in range(steps):
        angle_rad = math.radians(i * angle_increment)
        z = (i * rise_per_step) + (step_height / 2.0)
        
        start_x = pillar_radius * math.cos(angle_rad)
        start_y = pillar_radius * math.sin(angle_rad)
        
        end_r = pillar_radius + step_length
        end_x = end_r * math.cos(angle_rad)
        end_y = end_r * math.sin(angle_rad)
        
        shapes.append({
            "id": i + 1,
            "type": "beam",
            "start": [float(start_x), float(start_y), float(z)],
            "end": [float(end_x), float(end_y), float(z)],
            "width": step_width,
            "height": step_height
        })

    prompt = f"""Model a spiral staircase with {steps} beam treads.
The ground is the XY plane; the pillar axis passes through the origin along Z.
The cylindrical pillar has radius 5. Each tread has radial length 20, width 5,
and thickness 2. Adjacent tread bottom faces differ in elevation by 5.
There are 24 treads per complete revolution; the staircase winds counterclockwise
when viewed from positive Z. The first tread points along positive X and its
bottom face touches the ground. Each tread's inner centerline endpoint meets
the pillar's cylindrical surface and its outer endpoint lies radially outward.
The pillar starts on the ground and ends one rise above the last tread's bottom
face. ID 0 is the pillar; subsequent IDs follow the ascending tread order.

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
    parser.add_argument("--scale", type=int, default=24, help="Number of steps in the staircase")
    args = parser.parse_args()
    
    prompt, shapes = generate_staircase(args.scale)
    
    out_file = "staircase_golden.json"
    with open(out_file, "w") as f:
        json.dump(shapes, f, indent=2)
        
    print("="*60)
    print(f"PROMPT FOR SPIRAL STAIRCASE (SCALE/STEPS: {args.scale}):")
    print("="*60)
    print(prompt)
    print("\nNote: End your prompt instructing the AI to strictly output only the raw JSON format for C2CAD-Bench.")
    print("="*60)
    print(f"Golden JSON exactly representing this constraint saved to {out_file} ({len(shapes)} parts).")
