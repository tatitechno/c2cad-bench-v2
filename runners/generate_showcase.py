import sys
import os
import json

from generate_staircase import generate_staircase
from generate_pyramid import generate_pyramid
from generate_rubiks import generate_rubiks
from generate_stonehenge import generate_stonehenge
from generate_dna import generate_dna

TESTS = [
    {"family": "Spiral Staircase", "func": generate_staircase, "scales": [10, 24, 50]},
    {"family": "Cannonball Pyramid", "func": generate_pyramid, "scales": [3, 4, 5]},
    {"family": "Voxel Grid", "func": generate_rubiks, "scales": [2, 3, 4]},
    {"family": "Domino Ring", "func": generate_stonehenge, "scales": [5, 10, 20]},
    {"family": "DNA Helix", "func": generate_dna, "scales": [5, 10, 20]}
]

def generate_showcase():
    output = []
    
    for test in TESTS:
        for idx, scale in enumerate(test["scales"]):
            difficulty_level = idx + 1
            print(f"Generating {test['family']} (Level {difficulty_level}, Scale {scale})...")
            
            prompt, shapes = test["func"](scale)
            
            output.append({
                "family": test["family"],
                "difficultyLabel": f"Level {difficulty_level} (Scale {scale})",
                "difficultyID": difficulty_level,
                "shapes": shapes
            })
            
    out_file = "showcase_golden.json"
    temp_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(temp_dir, out_file)
    
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
        
    print("="*60)
    print(f"✅ Successfully bundled all 15 permutations into {out_path}!")
    print(f"Drop this file into the WebGL Visualizer to view the Museum Showcase.")
    print("="*60)

if __name__ == "__main__":
    generate_showcase()
