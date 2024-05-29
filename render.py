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
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos
from phys_engine.mass_spring import simulation

import open3d as o3d

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
    print("len(gaussians._xyz): ", gaussians._xyz.shape)
    cloth_gaussians = GaussianModel(dataset.sh_degree)
    cloth_xyz = []
    cloth_features_dc = []
    cloth_features_rest = []
    cloth_opacity = []
    cloth_scaling = []
    cloth_rotation = []
    for i in range(0, gaussians._xyz.shape[0]):
        if (gaussians._xyz[i, 2] < 3 and gaussians._xyz[i, 2] > 2 and gaussians._xyz[i, 0] > -1.2 and gaussians._xyz[i, 0] < 0.5 and gaussians._xyz[i, 1] > 1.5 and gaussians._xyz[i, 1] < 3.5):
        #if (gaussians._xyz[i, 0] > 0 and gaussians._xyz[i, 2] > 0.1 and gaussians._xyz[i, 1] < 0):
            cloth_xyz.append(gaussians._xyz[i, :])
            cloth_features_dc.append(gaussians._features_dc[i, :, :])
            cloth_features_rest.append(gaussians._features_rest[i, :, :])
            cloth_opacity.append(gaussians._opacity[i, :])
            cloth_scaling.append(gaussians._scaling[i, :])
            cloth_rotation.append(gaussians._rotation[i, :])
    cloth_xyz = torch.stack(cloth_xyz, dim=0) 
    cloth_features_dc = torch.stack(cloth_features_dc, dim=0)
    cloth_features_rest = torch.stack(cloth_features_rest, dim=0)
    cloth_opacity = torch.stack(cloth_opacity, dim=0)
    cloth_scaling = torch.stack(cloth_scaling, dim=0)
    cloth_rotation = torch.stack(cloth_rotation, dim=0)
    cloth_xyz = cloth_xyz.to(dtype=torch.float32, device="cuda")
    cloth_features_dc = cloth_features_dc.to(dtype=torch.float32, device="cuda")
    cloth_features_rest = cloth_features_rest.to(dtype=torch.float32, device="cuda")
    cloth_opacity = cloth_opacity.to(dtype=torch.float32, device="cuda")
    cloth_scaling = cloth_scaling.to(dtype=torch.float32, device="cuda")
    cloth_rotation = cloth_rotation.to(dtype=torch.float32, device="cuda")
    cloth_gaussians._xyz = cloth_xyz
    cloth_gaussians._features_dc = cloth_features_dc
    cloth_gaussians._features_rest = cloth_features_rest
    cloth_gaussians._opacity = cloth_opacity
    cloth_gaussians._scaling = cloth_scaling
    cloth_gaussians._rotation = cloth_rotation
    cloth_gaussians.active_sh_degree = gaussians.max_sh_degree
    print("gaussians._xyz.shape: ", gaussians._xyz.shape)
    print("cloth_gaussians._xyz.shape: ", cloth_gaussians._xyz.shape)
    # for i in range(0, cloth_gaussians._xyz.shape[0]):
        # print("cloth_gaussians._xyz[{}]".format(i), cloth_gaussians._xyz[i, :])
    #bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    x_history = simulation(cloth_xyz, cloth_gaussians.get_covariance())
    for i in range(len(x_history)):
        print("i: ", i)
        cloth_gaussians._xyz = torch.from_numpy(x_history[i]).to(dtype=torch.float32, device="cuda")

        train_dir = os.path.join(args.model_path, 'train', "ours_{}_{}".format(scene.loaded_iter, i))
        test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
        gaussExtractor = GaussianExtractor(cloth_gaussians, render, pipe, bg_color=bg_color)    

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
    '''
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
    '''