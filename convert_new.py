import os
import logging
import shutil
import open3d as o3d
import numpy as np
import cv2
from argparse import ArgumentParser

# 解析命令行参数
parser = ArgumentParser("PLY to COLMAP converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--ply_file", "-p", required=True, type=str)
parser.add_argument("--output_path", "-o", required=True, type=str)
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
args = parser.parse_args()

colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
use_gpu = 1 if not args.no_gpu else 0

# 读取 PLY 文件
ply_file = args.ply_file
output_path = args.output_path
point_cloud = o3d.io.read_point_cloud(ply_file)

# 设置虚拟相机参数 (可根据需要调整)
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=1920, height=1080, fx=2304, fy=2304, cx=960, cy=540
)

# 生成不同视角的图像并保存
os.makedirs(output_path + "/input", exist_ok=True)
angles = np.linspace(0, 360, num=10, endpoint=False)
for i, angle in enumerate(angles):
    # 设置视角，只绕Y轴旋转
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle([0, angle * np.pi / 180, 0])

    # 创建 PinholeCameraParameters 对象
    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = intrinsic
    camera_params.extrinsic = extrinsic

    # 渲染虚拟图像
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=1920, height=1080)
    vis.add_geometry(point_cloud)
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    vis.poll_events()
    vis.update_renderer()
    img = vis.capture_screen_float_buffer(do_render=True)
    img = np.array(img) * 255
    img = img.astype(np.uint8)
    cv2.imwrite(f"{output_path}/input/view_{i:03d}.png", img)
    vis.destroy_window()


if not args.skip_matching:
    os.makedirs(output_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + output_path + "/distorted/database.db \
        --image_path " + output_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + output_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + output_path + "/distorted/database.db \
        --image_path "  + output_path + "/input \
        --output_path "  + output_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### Image undistortion
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + output_path + "/input \
    --input_path " + output_path + "/distorted/sparse/0 \
    --output_path " + output_path + "\
    --output_type COLMAP")
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Image undistorter failed with code {exit_code}. Exiting.")
    exit(exit_code)

print("Done.")
