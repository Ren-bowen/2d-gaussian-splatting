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

import open3d as o3d
def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    
    Args:
        quaternion (torch.Tensor): A tensor of shape (4,) representing the quaternion (w, x, y, z).
        
    Returns:
        torch.Tensor: A tensor of shape (3, 3) representing the rotation matrix.
    """
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    
    rotation_matrix = torch.tensor([
        [1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy)],
        [2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy)]
    ])
    
    return rotation_matrix

'''
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
'''
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
    new_gaussians = GaussianModel(dataset.sh_degree)
    print("iteration: ", iteration)
    scene = Scene(dataset, new_gaussians, load_iteration=iteration, shuffle=False)

    file_path = os.path.join(args.model_path, 'iter_{}'.format(iteration))
    os.makedirs(file_path, exist_ok=True)
    if not os.path.exists(file_path + "/new_gaussians_xyz.npy"):
        np.save(file_path + "/new_gaussians_xyz.npy", new_gaussians._xyz.cpu().detach().numpy())
    new_scaling = torch.cat([new_gaussians._scaling, torch.ones((new_gaussians._scaling.shape[0], 1), device="cuda")], dim=1)
    if not os.path.exists(file_path + "/new_gaussians_scaling.npy"):
        np.save(file_path + "/new_gaussians_scaling.npy", new_scaling.cpu().detach().numpy())
    print("new_gaussians._xyz.shape: ", new_gaussians._xyz.shape)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    # bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # new_gaussians_xyz_np_list = np.load(os.path.join(file_path, "gaussian_xyz_list_rot.npy"))
    # F_np_list = np.load(os.path.join(file_path, "rotation_list_rot.npy"))
    # new_gaussians_scaling_np_list = np.load(os.path.join(file_path, "gaussian_scaling_list_rot.npy"))
    mean_xyz = torch.mean(new_gaussians._xyz, axis=0)
    num_gaussians = new_gaussians._xyz.shape[0]
    
    for i in range(0, 40, 1):
        print("i: ", i)
        # as xyz_mean to be center, rotation 20 degree around z axis
        rotation_matrix = np.array([[np.cos(20 * np.pi / 180), -np.sin(20 * np.pi / 180), 0],
                                        [np.sin(20 * np.pi / 180), np.cos(20 * np.pi / 180), 0],
                                        [0, 0, 1]])
        new_gaussians._xyz = (new_gaussians._xyz - mean_xyz) @ (torch.from_numpy(rotation_matrix).to(dtype=torch.float32, device="cuda")).T + mean_xyz
        # new_gaussians._xyz = torch.from_numpy(new_gaussians_xyz_np_list[i]).to(dtype=torch.float32, device="cuda")
        # repeat the rotation matrix for all gaussians
        rotation_matrix = np.repeat(rotation_matrix.reshape(1, 3, 3), num_gaussians, axis=0)
        # F_np = F_np_list[i]
        F_np = rotation_matrix
        new_gaussians_rotation = np.zeros((num_gaussians, 4))
        ti.init(arch = ti.gpu)
        @ti.kernel
        def Do_Rotate(rotation_matrixs: ti.types.ndarray(ndim=1), rotations: ti.types.ndarray(ndim=1), features: ti.types.ndarray(ndim=1)):
            print("rotation_matrixs[0]: ", rotation_matrixs[0])
            print("rotation_matrix_to_quaternion(rotation_matrixs[0]): ", rotation_matrix_to_quaternion(rotation_matrixs[0]))
            for i in range(features.shape[0]):
                rotation_matrix = rotation_matrixs[i]
                rotation_quaternion = rotation_matrix_to_quaternion(rotation_matrix)
                rotations[i] = quaternion_multiply(rotation_quaternion, rotations[i])
                features[i] = RotateSH(rotation_matrix, features[i])
        rotation_ti = ti.Matrix.ndarray(3, 3, ti.f32, shape=(num_gaussians))
        rotation_ti.from_numpy(F_np)
        new_gaussians_rotation_ti = ti.Vector.ndarray(4, ti.f32, shape=(num_gaussians))
        new_gaussians_rotation_ti.from_numpy(new_gaussians._rotation.cpu().detach().numpy())
        new_gaussians_features_rest_ti = ti.Matrix.ndarray(16, 3, ti.f32, shape=(num_gaussians))
        new_gaussians_features_rest_ti.from_numpy(torch.cat((new_gaussians._features_rest.cpu().detach(), torch.zeros(num_gaussians, 1, 3)), dim=1).numpy())
        # Do_Rotate(rotation_ti, new_gaussians_rotation_ti, new_gaussians_features_rest_ti)
        # print("new_gaussians_rotation_ti[0]: ", new_gaussians_rotation_ti[0])
        # print("quaternion_to_rotation_matrix(torch.from_numpy(new_gaussians_rotation_ti[0].to_numpy())): ", quaternion_to_rotation_matrix(torch.from_numpy(new_gaussians_rotation_ti[0].to_numpy())))
        # rotation_matrix_trans = quaternion_to_rotation_matrix(torch.from_numpy(new_gaussians_rotation_ti[0].to_numpy())).to(dtype=torch.float32, device="cuda") @ (quaternion_to_rotation_matrix(new_gaussians._rotation[0]).T).to(dtype=torch.float32, device="cuda")
        # print("rotation_matrix_trans: ", rotation_matrix_trans)
        new_gaussians._rotation = torch.from_numpy(new_gaussians_rotation_ti.to_numpy()).to(dtype=torch.float32, device="cuda")
        new_gaussians._features_rest = torch.from_numpy(new_gaussians_features_rest_ti.to_numpy()).to(dtype=torch.float32, device="cuda")[:, :15, :]

        train_dir = os.path.join(args.model_path, 'train', "ours_{}_{}_rot_".format(scene.loaded_iter, i))
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
        