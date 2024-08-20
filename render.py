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
from utils.rotate_sh import RotateSH
import taichi as ti
import numpy as np

import open3d as o3d
def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    
    Args:
        quaternion (torch.Tensor): A tensor of shape (4,) representing the quaternion (w, x, y, z).
        
    Returns:
        torch.Tensor: A tensor of shape (3, 3) representing the rotation matrix.
    """
    w, x, y, z = quaternion
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    
    rotation_matrix = torch.tensor([
        [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
    ])
    
    return rotation_matrix

def multiplt_quaternian(q1, q2):
    # input: pytorch tensor of shape (n, 4)
    # output: pytorch tensor of shape (n, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=1)


def rotation_matrix_to_quaternion(rotation_matrix):
    """
    Convert a 3x3 rotation matrix to a quaternion.
    
    Args:
        rotation_matrix (torch.Tensor): A tensor of shape (3, 3) representing the rotation matrix.

    Returns:
        torch.Tensor: A tensor of shape (4,) representing the quaternion (w, x, y, z).
    """
    trace = rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]
    w = torch.sqrt(1 + trace) / 2
    x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4*w)
    y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4*w)
    z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4*w)
    
    return torch.tensor([w, x, y, z])
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
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians_xyz_np = gaussians._xyz.cpu().detach().numpy()
    gaussians_rotation_np = np.zeros((gaussians._rotation.shape[0], 3, 3))
    print("len(gaussians._xyz): ", gaussians._xyz.shape)
    
    new_gaussians = GaussianModel(dataset.sh_degree)
    new_xyz = []
    new_features_dc = []
    new_features_rest = []
    new_opacity = []
    new_scaling = []
    new_rotation = []

    p1 = np.array([-4.451480, 0.974875, 5.581665])
    p2 = np.array([-1.992872, -0.037731, 6.155551])
    p3 = np.array([-0.514574, 2.460062, 4.277073])
    p4 = np.array([-2.974283, 3.445327, 3.594432])
    paper1 = np.load(r"C:\ren\code\2d-gaussian-splatting\output\input\regular.npy")
    # paper1 = np.zeros(gaussians._xyz.shape[0])
    normal = np.cross(p2 - p1, p3 - p1)
    normal = normal / np.linalg.norm(normal)
    print("gaussians._features_dc.shape: ", gaussians._features_dc.shape)
    print("gaussians._features_rest.shape: ", gaussians._features_rest.shape)
    for i in range(0, gaussians._xyz.shape[0]):
        p = gaussians._xyz[i, :].cpu().detach().numpy()
        # if (p - p1).dot(p2 - p1) > 0 and (p - p2).dot(p1 - p2) > 0 and (p - p3).dot(p4 - p3) > 0 and (p - p4).dot(p3 - p4) > 0 and (p - p2).dot(p3 - p2) > 0 and (p - p3).dot(p2 - p3) > 0 and (p - p1).dot(p4 - p1) > 0 and (p - p4).dot(p1 - p4) > 0:
        #     if (np.abs((p - p1).dot(normal)) < 0.14):
        if(np.abs(paper1[i] - 1) < 0.1):
                new_xyz.append(gaussians._xyz[i, :])
                new_features_dc.append(gaussians._features_dc[i, :, :])
                new_features_rest.append(gaussians._features_rest[i, :, :])
                new_opacity.append(gaussians._opacity[i, :])
                new_scaling.append(gaussians._scaling[i, :])
                new_rotation.append(gaussians._rotation[i, :])
                paper1[i] = 1
    
    # np.save(r"C:\ren\code\2d-gaussian-splatting\output\input\regular", paper1)
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

    # np.save(r"C:\Users\bowen ren\Desktop\gaussian_xyz_regular", np.array(new_xyz.cpu().detach().numpy()))
    new_rotation_ = build_rotation(new_rotation)
    # np.save(r"C:\Users\bowen ren\Desktop\gaussian_rotation_regular", np.array(new_rotation_.cpu().detach().numpy()))
    new_scaling = torch.cat([new_scaling, torch.ones((new_scaling.shape[0], 1), device="cuda")], dim=1)
    # np.save(r"C:\Users\bowen ren\Downloads\gaussian_scaling_regular", new_scaling.cpu().detach().numpy())
    new_scaling = new_scaling[:, :2]

    new_gaussians._xyz = new_xyz
    new_gaussians._features_dc = new_features_dc
    new_gaussians._features_rest = new_features_rest
    new_gaussians._opacity = new_opacity
    new_gaussians._scaling = new_scaling
    new_gaussians._rotation = new_rotation
    new_gaussians.active_sh_degree = gaussians.max_sh_degree
    print("gaussians._xyz.shape: ", gaussians._xyz.shape)
    print("new_gaussians._xyz.shape: ", new_gaussians._xyz.shape)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    # bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    new_gaussians_xyz_np_list = np.load(r"C:\ren\code\2d-gaussian-splatting\data\gaussian_xyz_list_paper2_rot.npy")
    new_gaussians_rotation_np_list = np.load(r"C:\ren\code\2d-gaussian-splatting\data\gaussian_rotation_list_paper2_rot.npy")
    new_gaussians_scaling_np_list = np.load(r"C:\ren\code\2d-gaussian-splatting\data\gaussian_scaling_list_paper2_rot.npy")
    mean_xyz = torch.mean(gaussians._xyz, axis=0)
    for i in range(20, 200, 10):
        print("i: ", i)
        new_gaussians._xyz = torch.from_numpy(new_gaussians_xyz_np_list[i]).to(dtype=torch.float32, device="cuda")
        new_gaussians_rotation_np = new_gaussians_rotation_np_list[i]
        new_gaussians_rotation = np.zeros((new_gaussians_rotation_np.shape[0], 4))
        ti.init(arch = 'cuda')
        @ti.kernel
        def RotateSHs(rotation: ti.types.matrix(3, 3, float), features: ti.types.matrix(16, 3, float)) -> ti.types.matrix(16, 3, float):
            return RotateSH(rotation, features)
        for j in range(0, new_gaussians_rotation_np.shape[0]):
            new_gaussians_rotation[j] = rotation_matrix_to_quaternion(torch.from_numpy(new_gaussians_rotation_np[j]).to(dtype=torch.float32, device="cuda"))
            rotation_matrix = (torch.from_numpy(new_gaussians_rotation_np[j]) @ quaternion_to_rotation_matrix(new_gaussians._rotation[j]).T).to(dtype=torch.float32, device="cuda")
            new_gaussians_features_rest = torch.cat((new_gaussians._features_rest[j], torch.zeros(1, 3).to(dtype=torch.float32, device="cuda")), dim=0)
            Matrix_3x3 = ti.types.matrix(3, 3, float)
            rotation_matrix_ti = Matrix_3x3(rotation_matrix.cpu().detach().numpy().tolist())
            Matrix_16x3 = ti.types.matrix(16, 3, float)
            new_gaussians_features_rest_ti = Matrix_16x3(new_gaussians_features_rest.cpu().detach().numpy().tolist())
            new_gaussians_features_rest_ti = RotateSHs(rotation_matrix_ti, new_gaussians_features_rest_ti)
            new_gaussians_features_rest = torch.tensor(new_gaussians_features_rest_ti.to_numpy()).to(dtype=torch.float32, device="cuda")
            new_gaussians._features_rest[j] = new_gaussians_features_rest[:15, :]



        new_gaussians_rotation = torch.from_numpy(new_gaussians_rotation).to(dtype=torch.float32, device="cuda")
        # print("new_gaussians_rotation: ", new_gaussians_rotation[0])
        # print("new_gaussians._rotation: ", new_gaussians._rotation[0])
        # print("new_gaussians_rotation: ", new_gaussians_rotation[0] @ new_gaussians._rotation[0].T)
        new_gaussians._rotation = new_gaussians_rotation
        # new_gaussians._scaling = torch.from_numpy(new_gaussians_scaling_np_list[i][:, :2]).to(dtype=torch.float32, device="cuda")
        # as xyz_mean to be center, rotation 20 degree around z axis
        # rotation_matrix = torch.tensor([[np.cos(20 * np.pi / 180), -np.sin(20 * np.pi / 180), 0],
        #                                 [np.sin(20 * np.pi / 180), np.cos(20 * np.pi / 180), 0],
        #                                 [0, 0, 1]]).to(dtype=torch.float32, device="cuda")
        # new_gaussians._xyz = (new_gaussians._xyz - mean_xyz) @ rotation_matrix.T + mean_xyz
        # new_gaussians._rotation = multiplt_quaternian(new_gaussians._rotation, rotation_matrix_to_quaternion(rotation_matrix.to(dtype=torch.float32, device="cuda")))


        train_dir = os.path.join(args.model_path, 'train', "ours_{}_{}".format(scene.loaded_iter, i))
        test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
        gaussExtractor = GaussianExtractor(new_gaussians, render, pipe, bg_color=bg_color)    

        if not args.skip_train:
            # print("export training images ...")
            os.makedirs(train_dir, exist_ok=True)
            gaussExtractor.reconstruction(scene.getTrainCameras())
            gaussExtractor.export_image(train_dir)
            
        
        if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
            print("export rendered testing images ...")
            os.makedirs(test_dir, exist_ok=True)
            gaussExtractor.reconstruction(scene.getTestCameras())
            gaussExtractor.export_image(test_dir)
        

        if args.render_path:
            print("render videos ...")
            traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
            os.makedirs(traj_dir, exist_ok=True)
            n_fames = 240
            cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
            gaussExtractor.reconstruction(cam_traj)
            gaussExtractor.export_image(traj_dir)
            create_videos(base_dir=traj_dir,
                        input_dir=traj_dir, 
                        out_name='render_traj', 
                        num_frames=n_fames)
        
        if not args.skip_mesh:
            print("export mesh ...")
            os.makedirs(train_dir, exist_ok=True)
            # set the active_sh to 0 to export only diffuse texture
            gaussExtractor.gaussians.active_sh_degree = 0
            gaussExtractor.reconstruction(scene.getTrainCameras())
            # extract the mesh and save
            if args.unbounded:
                name = 'fuse_unbounded.ply'
                mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
            else:
                name = 'fuse.ply'
                depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
                voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
                sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
                mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
            
            o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
            print("mesh saved at {}".format(os.path.join(train_dir, name)))
            # post-process the mesh and save, saving the largest N clusters
            mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
            o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
            print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))
        