import math
import json

def rotate_vector_y(vector, angle_degrees):
    """Rotates a 3D vector around the Y-axis by a given angle in degrees."""
    angle_radians = math.radians(angle_degrees)
    x, y, z = vector
    new_x = x * math.cos(angle_radians) + z * math.sin(angle_radians)
    new_z = -x * math.sin(angle_radians) + z * math.cos(angle_radians)
    return [new_x, y, new_z]

def normalize_vector(vector):
    """Normalizes a 3D vector."""
    magnitude = math.sqrt(sum(c**2 for c in vector))
    if magnitude == 0:
        return [0.0, 0.0, 0.0]
    return [c / magnitude for c in vector]

all_beams_data = []

def create_branch(level, start_point, direction_vector, length):
    """
    Recursively generates tree branches based on specified rules.

    Args:
        level (int): The current recursion level.
        start_point (list): [x, y, z] coordinates of the branch's start.
        direction_vector (list): Normalized [dx, dy, dz] vector indicating the branch's direction.
        length (float): The length of the current branch.
    """
    global all_beams_data

    if level > 5: # Base case: stop at Level 5
        return

    # Calculate end point of the current branch
    end_point = [
        start_point[0] + direction_vector[0] * length,
        start_point[1] + direction_vector[1] * length,
        start_point[2] + direction_vector[2] * length
    ]

    # Add current beam to the global list, rounding coordinates for cleaner output
    all_beams_data.append({
        "type": "beam",
        "start": [round(c, 6) for c in start_point],
        "end": [round(c, 6) for c in end_point],
        "width": 1,
        "height": 1
    })

    # If not at the maximum recursion level, create child branches
    if level < 5:
        child_start_point = end_point
        child_length = length / 2

        # Determine the reference vector for children's rotation
        if level == 0:
            # For children spawned from the vertical trunk, the 'parent's directional axis'
            # in the XZ plane is taken as [1, 0, 0] for rotation reference.
            base_xz_direction = [1, 0, 0]
        else:
            # For subsequent levels, the current branch's direction vector is
            # already in the XZ plane and serves as the reference for its children.
            base_xz_direction = direction_vector

        # Calculate left child's direction
        left_child_dir = normalize_vector(rotate_vector_y(base_xz_direction, 45))
        left_child_dir[1] = 0.0 # Ensure it lies entirely in the XZ plane

        # Calculate right child's direction
        right_child_dir = normalize_vector(rotate_vector_y(base_xz_direction, -45))
        right_child_dir[1] = 0.0 # Ensure it lies entirely in the XZ plane

        # Recursively call for left and right children
        create_branch(level + 1, child_start_point, left_child_dir, child_length)
        create_branch(level + 1, child_start_point, right_child_dir, child_length)

# Initial call for the trunk (Level 0)
# Starts at (0, 0, 0), vertical direction (along Y-axis), length 10
initial_start_point = [0, 0, 0]
initial_direction_vector = [0, 1, 0] # Points upwards along Y-axis
initial_length = 10.0

create_branch(0, initial_start_point, initial_direction_vector, initial_length)

# Output the collected beam data as raw JSON
print(json.dumps(all_beams_data, indent=None, separators=(',', ':')))