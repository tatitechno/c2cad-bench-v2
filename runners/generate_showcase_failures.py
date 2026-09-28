import json
import random
import os

def degrade_shapes(shapes, difficulty_level):
    """
    Simulates reasoning detachment. AI tends to track well early on,
    but math starts drifting or shapes get completely truncated at higher counts.
    """
    degraded = []
    
    # 1. Truncation: AI stops outputting JSON after a certain number of parts.
    # High difficulty structures (50+ parts) often get truncated at ~35-40 parts.
    max_parts = 1000
    if difficulty_level == 2:
        max_parts = len(shapes) - max(1, int(len(shapes) * 0.15)) # lose 15%
    elif difficulty_level == 3:
        max_parts = max(5, int(len(shapes) * 0.40)) # severe cutoff at 40%
        
    for i, shape in enumerate(shapes):
        if i >= max_parts:
            break
            
        new_shape = dict(shape)
        
        # 2. Math Drift (Positional Offset)
        # As loops get longer, the AI forgets the axis or radius math.
        if difficulty_level > 1 and "center" in new_shape:
            # 10% chance of random coordinate drift on Medium, 40% on Hard
            chance = 0.10 if difficulty_level == 2 else 0.40
            if random.random() < chance:
                drift_amt = 5.0 * difficulty_level
                new_shape["center"] = [
                    new_shape["center"][0] + random.uniform(-drift_amt, drift_amt),
                    new_shape["center"][1] + random.uniform(-drift_amt, drift_amt),
                    new_shape["center"][2] + random.uniform(-drift_amt, drift_amt)
                ]
                
        # Beam drifting
        if difficulty_level > 1 and new_shape.get("type") == "beam":
            chance = 0.20 if difficulty_level == 2 else 0.50
            if random.random() < chance:
                drift_amt = 8.0
                new_shape["end"] = [
                    new_shape["end"][0] + random.uniform(-drift_amt, drift_amt),
                    new_shape["end"][1] + random.uniform(-drift_amt, drift_amt),
                    new_shape["end"][2]
                ]

        # 3. Axis Twisting (Wrong rotation logic)
        if new_shape.get("type") == "cylinder" and "axis" in new_shape:
            chance = 0.05 if difficulty_level == 1 else 0.30
            if random.random() < chance:
                # Twist the axis randomly
                new_shape["axis"] = [
                    new_shape["axis"][0] + random.uniform(-0.5, 0.5),
                    new_shape["axis"][1] + random.uniform(-0.5, 0.5),
                    new_shape["axis"][2] + random.uniform(-0.5, 0.5)
                ]

        degraded.append(new_shape)
        
    return degraded

def generate_failures():
    golden_file = os.path.join(os.path.dirname(__file__), "showcase_golden.json")
    if not os.path.exists(golden_file):
        print("Please run generate_showcase.py first to create the golden baseline.")
        return
        
    with open(golden_file, "r") as f:
        golden_data = json.load(f)
        
    llm_output = []
    
    # Degrade each geometry array based on its difficulty metadata
    for suite in golden_data:
        diff = suite["difficultyID"]
        degraded = degrade_shapes(suite["shapes"], diff)
        
        llm_output.append({
            "family": suite["family"],
            "difficultyLabel": suite["difficultyLabel"],
            "difficultyID": diff,
            "shapes": degraded
        })
        
    out_file = os.path.join(os.path.dirname(__file__), "showcase_llm.json")
    with open(out_file, "w") as f:
        json.dump(llm_output, f, indent=2)
        
    print(f"Generated {out_file} containing mathematically degraded AI artifacts!")

if __name__ == "__main__":
    generate_failures()
