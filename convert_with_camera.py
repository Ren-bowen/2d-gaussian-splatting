import os
import json
import logging
from argparse import ArgumentParser
import shutil

parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
args = parser.parse_args()
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
use_gpu = 1 if not args.no_gpu else 0

# Single database and sparse directory
database_path = os.path.join(args.source_path, "distorted", "database.db")
sparse_path = os.path.join(args.source_path, "distorted", "sparse")
os.makedirs(sparse_path, exist_ok=True)

image_dir = os.path.join(args.source_path, "input")
json_dir = os.path.join(args.source_path, "json")

image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))]

for i, image_file in enumerate(image_files):
    # Load corresponding camera parameters JSON file
    json_file = os.path.join(json_dir, os.path.splitext(image_file)[0] + ".json")
    
    with open(json_file, 'r') as f:
        camera_params = json.load(f)
    
    fx = camera_params['intrinsic']['intrinsic_matrix'][0]
    fy = camera_params['intrinsic']['intrinsic_matrix'][4]
    cx = camera_params['intrinsic']['intrinsic_matrix'][6]
    cy = camera_params['intrinsic']['intrinsic_matrix'][7]

    # Add distortion parameters if available, or set to 0
    k1 = camera_params.get('distortion', {}).get('k1', 0.0)
    k2 = camera_params.get('distortion', {}).get('k2', 0.0)
    p1 = camera_params.get('distortion', {}).get('p1', 0.0)
    p2 = camera_params.get('distortion', {}).get('p2', 0.0)

    camera_params_str = f"{fx},{fy},{cx},{cy},{k1},{k2},{p1},{p2}"

    # Create a temporary image list file for the current image
    temp_image_list = os.path.join(args.source_path, "temp_image_list.txt")
    with open(temp_image_list, 'w') as f:
        f.write(f"{os.path.join(image_dir, image_file)}\n")

    ## Feature extraction for the current image only
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + database_path + " \
        --image_path " + image_dir + " \
        --image_list_path " + temp_image_list + " \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --ImageReader.camera_params " + camera_params_str + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed for {image_file} with code {exit_code}. Exiting.")
        exit(exit_code)

    # Remove the temporary image list file
    os.remove(temp_image_list)

if not args.skip_matching:
    ## Feature matching across all images
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + database_path + " \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Bundle adjustment (mapper)
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + database_path + " \
        --image_path " + args.source_path + "/input \
        --output_path " + sparse_path + " \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion (if needed)
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + sparse_path + "/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Image undistortion failed with code {exit_code}. Exiting.")
    exit(exit_code)

print("Reconstruction complete. All images processed and results merged.")