import math
import json

def generate():
    shapes = []
    
    # 0. Base Plate
    shapes.append({
        "id": 0,
        "type": "box",
        "center": [0.0, 0.0, 1.5],
        "size": [50.0, 50.0, 3.0]
    })
    
    # 1-30. Fiber Beams
    beam_id = 1
    offsets = [-16.0, -8.0, 0.0, 8.0, 16.0]
    half_length = 18.0
    radius = 1.0
    
    for layer in range(6):
        z = 3.0 + 1.0 + layer * 2.0
        angle_deg = layer * 30.0
        angle_rad = math.radians(angle_deg)
        
        # Fiber direction vector
        dir_x = math.cos(angle_rad)
        dir_y = math.sin(angle_rad)
        
        # Perpendicular direction vector (rotate dir by 90 deg in XY)
        perp_x = -math.sin(angle_rad)
        perp_y = math.cos(angle_rad)
        
        for d in offsets:
            # Center point of the fiber in XY
            cx = d * perp_x
            cy = d * perp_y
            
            # Start and End points
            start = [
                round(cx - half_length * dir_x, 3),
                round(cy - half_length * dir_y, 3),
                round(z, 3)
            ]
            end = [
                round(cx + half_length * dir_x, 3),
                round(cy + half_length * dir_y, 3),
                round(z, 3)
            ]
            
            shapes.append({
                "id": beam_id,
                "type": "beam",
                "start": start,
                "end": end,
                "start_radius": radius,
                "end_radius": radius
            })
            beam_id += 1
            
    # 31-34. Lock Pins
    pin_id = 31
    pin_radius = 1.5
    pin_height = 15.0
    pin_z = 7.5
    for px in [20.0, -20.0]:
        for py in [20.0, -20.0]:
            shapes.append({
                "id": pin_id,
                "type": "cylinder",
                "center": [px, py, pin_z],
                "radius": pin_radius,
                "height": pin_height,
                "axis": [0, 0, 1]
            })
            pin_id += 1
            
    return shapes

if __name__ == "__main__":
    result = generate()
    print(json.dumps(result, indent=2))
