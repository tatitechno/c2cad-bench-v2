import math

def rotate_x(vector, angle_deg):
    angle_rad = math.radians(angle_deg)
    x, y, z = vector
    y_new = y * math.cos(angle_rad) - z * math.sin(angle_rad)
    z_new = y * math.sin(angle_rad) + z * math.cos(angle_rad)
    return [float(x), float(y_new), float(z_new)]

# Parameters
m = 3.0
h = 10.0
sun_teeth = 15
planet_teeth = 12
ring_teeth = sun_teeth + 2 * planet_teeth
sun_r = m * sun_teeth / 2
planet_r = m * planet_teeth / 2
center_dist = sun_r + planet_r
ring_r = m * ring_teeth / 2

# Shapes (Center, Radius, Height, Axis)
# 1 Sun: ID 0
# 3 Planets: ID 1-3
# 3 Planet shafts: ID 4-6
# 1 Sun shaft: ID 7
# 1 Carrier disc: ID 8
# 1 Housing: ID 9
# 1 Ring (as cylinder): ID 10
shapes = []
# 0: Sun
shapes.append({"id": 0, "type": "cylinder", "center": [0, 0, 0], "radius": sun_r, "height": h, "axis": [0, 0, 1]})
# 1-3: Planets
for i in range(3):
    angle = 2 * math.pi * i / 3
    x, y = center_dist * math.cos(angle), center_dist * math.sin(angle)
    shapes.append({"id": 1+i, "type": "cylinder", "center": [x, y, 0], "radius": planet_r, "height": h, "axis": [0, 0, 1]})
# 4-6: Planet Shafts
for i in range(3):
    angle = 2 * math.pi * i / 3
    x, y = center_dist * math.cos(angle), center_dist * math.sin(angle)
    shapes.append({"id": 4+i, "type": "cylinder", "center": [x, y, 0], "radius": 2.5, "height": 25.0, "axis": [0, 0, 1]})
# 7: Sun Shaft
shapes.append({"id": 7, "type": "cylinder", "center": [0, 0, 0], "radius": 3.5, "height": 35.0, "axis": [0, 0, 1]})
# 8: Carrier Disc
shapes.append({"id": 8, "type": "cylinder", "center": [0, 0, h/2 + 2.5], "radius": center_dist + planet_r, "height": 5.0, "axis": [0, 0, 1]})
# 9: Housing
shapes.append({"id": 9, "type": "cylinder", "center": [0, 0, 0], "radius": ring_r + 8, "height": 20.0, "axis": [0, 0, 1]})
# 10: Ring
shapes.append({"id": 10, "type": "cylinder", "center": [0, 0, 0], "radius": ring_r, "height": h, "axis": [0, 0, 1]})

# Rotate all
for s in shapes:
    s["center"] = rotate_x(s["center"], 15.0)
    s["axis"] = rotate_x(s["axis"], 15.0)

import json
print(json.dumps(shapes, indent=2))
