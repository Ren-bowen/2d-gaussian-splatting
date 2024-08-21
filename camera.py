import sqlite3
import json
import numpy as np
import glob
import os

# Function to convert rotation matrix to quaternion
def rotation_matrix_to_quaternion(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

# Connect to COLMAP database
conn = sqlite3.connect(r'C:\ren\code\2d-gaussian-splatting\data\textured_cloth\distorted\database.db')
cursor = conn.cursor()

# Get a list of all JSON files
json_files = glob.glob(r'C:\ren\code\2d-gaussian-splatting\conv\*.json')

for i, json_file in enumerate(json_files):
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract intrinsic parameters
    intrinsic_matrix = data["intrinsic"]["intrinsic_matrix"]
    fx = intrinsic_matrix[0]
    fy = intrinsic_matrix[4]
    cx = intrinsic_matrix[6]
    cy = intrinsic_matrix[7]
    width = data["intrinsic"]["width"]
    height = data["intrinsic"]["height"]

    # Extract extrinsic parameters
    extrinsic_matrix = np.array(data["extrinsic"]).reshape(4, 4)
    R = extrinsic_matrix[:3, :3]
    t = extrinsic_matrix[:3, 3]

    # Convert R to quaternion
    qvec = rotation_matrix_to_quaternion(R)

    # Generate unique IDs for camera and image
    camera_id = i + 1  # Or use a more sophisticated ID generator
    image_id = i + 1

    # Insert camera parameters
    model = 'PINHOLE'  # Or another model if appropriate
    prior_focal_length = 0
    cursor.execute("INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)", (camera_id, model, width, height, f"{fx},{fy},{cx},{cy}", prior_focal_length))
    print(f"Inserted camera with ID {camera_id}")

    # Insert image parameters
    image_name = os.path.basename(json_file).replace('.json', '.png')  # Assuming the image has the same name as the JSON file but with .png extension
    cursor.execute("INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (image_id, camera_id, image_name, qvec[0], qvec[1], qvec[2], qvec[3], t[0], t[1], t[2]))

# Commit changes and close the connection
conn.commit()
conn.close()
