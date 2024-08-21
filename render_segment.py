#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state, build_rotation
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos
from utils.rotate_sh import RotateSH, rotation_matrix_to_quaternion, quaternion_multiply
import taichi as ti
import numpy as np

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: trunstackion value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    print("iteration: ", iteration)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    # new_gaussians = GaussianModel(dataset.sh_degree)
    new_xyz = []
    new_features_dc = []
    new_features_rest = []
    new_opacity = []
    new_scaling = []
    new_rotation = []

    p1 = np.array([-0.184466, 1.931121, 2.687635])
    p2 = np.array([1.995424, 2.209363, 2.318887])
    p3 = np.array([2.606919, -0.480207, 3.382995])
    p4 = np.array([0.365852, -0.714891, 3.877035])
    mask = np.zeros(gaussians._xyz.shape[0])
    normal = np.cross(p2 - p1, p3 - p1)
    normal = normal / np.linalg.norm(normal)
    print("gaussians._features_dc.shape: ", gaussians._features_dc.shape)
    print("gaussians._features_rest.shape: ", gaussians._features_rest.shape)
    m = -0.1
    for i in range(0, gaussians._xyz.shape[0]):
        p = gaussians._xyz[i, :].cpu().detach().numpy()
        if (p - p1).dot(p2 - p1) > m and (p - p2).dot(p1 - p2) > m and (p - p3).dot(p4 - p3) > m and (p - p4).dot(p3 - p4) > m and (p - p2).dot(p3 - p2) > m and (p - p3).dot(p2 - p3) > m and (p - p1).dot(p4 - p1) > m and (p - p4).dot(p1 - p4) > m:
            if (np.abs((p - p1).dot(normal)) < 0.18):
                new_xyz.append(gaussians._xyz[i, :])
                new_features_dc.append(gaussians._features_dc[i, :, :])
                new_features_rest.append(gaussians._features_rest[i, :, :])
                new_opacity.append(gaussians._opacity[i, :])
                new_scaling.append(gaussians._scaling[i, :])
                new_rotation.append(gaussians._rotation[i, :])
                mask[i] = 1
    print("mask.sum(): ", mask.sum())
    file_path = r"C:\ren\code\2d-gaussian-splatting\output\paper\iter_-2"
    os.makedirs(file_path, exist_ok=True)
    np.save(r"C:\ren\code\2d-gaussian-splatting\output\paper\iter_-2\mask.npy", mask)
    new_xyz = torch.stack(new_xyz, dim=0)
    new_features_dc = torch.stack(new_features_dc, dim=0)
    new_features_rest = torch.stack(new_features_rest, dim=0)
    new_opacity = torch.stack(new_opacity, dim=0)
    new_scaling = torch.stack(new_scaling, dim=0)
    new_rotation = torch.stack(new_rotation, dim=0)
    new_xyz = new_xyz.to(dtype=torch.float32, device="cuda")
    new_features_dc = new_features_dc.to(dtype=torch.float32, device="cuda")
    new_features_rest = new_features_rest.to(dtype=torch.float32, device="cuda")
    new_opacity = new_opacity.to(dtype=torch.float32, device="cuda")
    new_scaling = new_scaling.to(dtype=torch.float32, device="cuda")
    new_rotation = new_rotation.to(dtype=torch.float32, device="cuda")
    
    gaussians._xyz = new_xyz
    gaussians._features_dc = new_features_dc
    gaussians._features_rest = new_features_rest
    gaussians._opacity = new_opacity
    gaussians._scaling = new_scaling
    gaussians._rotation = new_rotation / torch.norm(new_rotation, dim=1, keepdim=True)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    scene.save(-2)
    
    if not os.path.exists(file_path + "/new_gaussians_xyz.npy"):
        np.save(file_path + "/new_gaussians_xyz.npy", gaussians._xyz.cpu().detach().numpy())
    new_scaling = torch.cat([gaussians._scaling, torch.ones((gaussians._scaling.shape[0], 1), device="cuda")], dim=1)
    if not os.path.exists(file_path + "/new_gaussians_scaling.npy"):
        np.save(file_path + "/new_gaussians_scaling.npy", new_scaling.cpu().detach().numpy())
    new_scaling = new_scaling[:, :2]