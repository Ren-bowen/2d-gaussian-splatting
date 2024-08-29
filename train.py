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

import os
import torch
import trimesh
from random import randint
from utils.loss_utils import l1_loss, ssim, surfel_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.mesh_utils import post_process_mesh
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from render import GaussianExtractor
from gaussian_renderer import render
import torch.nn.functional as F
import open3d as o3d
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.loss import point_mesh_face_distance
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def point_to_triangle_distance(points, triangles):
    N = points.shape[0]
    M = triangles.shape[0]
    # points: (N, 3)
    # triangles: (M, 3, 3)

    # 将点扩展到与三角形匹配的形状 (N, 1, 3)
    p = points[:, None, :]  # (N, 1, 3)
    
    # 拆分三角形顶点
    v0 = triangles[:, 0, :]  # (M, 3)
    v1 = triangles[:, 1, :]  # (M, 3)
    v2 = triangles[:, 2, :]  # (M, 3)

    # 计算边向量
    edge0 = v1 - v0  # (M, 3)
    edge1 = v2 - v0  # (M, 3)
    
    # 计算法向量和平面的垂直距离
    normal = torch.cross(edge0, edge1, dim=1)  # (M, 3)
    normal_norm = torch.norm(normal, dim=1, keepdim=True)  # (M, 1)
    
    # 计算每个点到每个三角形平面的距离
    # 这里需要扩展 normal 的第一个维度，以匹配 p 和 v0_to_p 的形状
    normal_expanded = normal.unsqueeze(0).expand(N, M, 3)  # (N, M, 3)
    normal_norm_expanded = normal_norm.unsqueeze(0).expand(N, M, 1)  # (N, M, 1)

    v0_to_p = p - v0[None, :, :]  # (N, M, 3)
    distance_to_plane = torch.abs(torch.sum(v0_to_p * normal_expanded, dim=2) / normal_norm_expanded.squeeze(2))  # (N, M)
    
    # 投影点到平面上
    projection = p - distance_to_plane[..., None] * (normal[None, :, :] / normal_norm)  # (N, M, 3)

    # 判断投影点是否在三角形内
    def is_point_in_triangle(pt, v0, v1, v2):
        edge0 = v1 - v0
        edge1 = v2 - v0
        v0_to_pt = pt - v0

        c0 = torch.cross(edge0, v0_to_pt, dim=2)
        c1 = torch.cross(v2 - v1, pt - v1, dim=2)
        c2 = torch.cross(v0 - v2, pt - v2, dim=2)

        inside = (torch.sum(c0 * c1, dim=2) >= 0) & (torch.sum(c0 * c2, dim=2) >= 0)
        return inside

    # 检查投影点是否在三角形内
    inside_triangle = is_point_in_triangle(projection, v0[None, :, :], v1[None, :, :], v2[None, :, :])  # (N, M)
    
    # 如果在三角形内，距离就是垂直距离
    distances = torch.where(inside_triangle, distance_to_plane, torch.full_like(distance_to_plane, float('inf')))  # (N, M)

    # 如果不在三角形内，计算点到三角形边的最小距离
    def point_to_edge_distance(pt, v0, v1):
        edge = v1 - v0  # (M, 3)
        t = torch.sum((pt - v0) * edge, dim=2) / torch.sum(edge * edge, dim=1)  # (N, M)
        t = torch.clamp(t, 0, 1)  # (N, M)
        closest_point = v0[None, :, :] + t[..., None] * edge[None, :, :]  # (N, M, 3)
        return torch.norm(pt - closest_point, dim=2)  # (N, M)

    dist_to_edge0 = point_to_edge_distance(p, v0, v1)
    dist_to_edge1 = point_to_edge_distance(p, v1, v2)
    dist_to_edge2 = point_to_edge_distance(p, v2, v0)
    
    edge_distances = torch.min(torch.stack([dist_to_edge0, dist_to_edge1, dist_to_edge2], dim=2), dim=2).values  # (N, M)

    # 最终距离是垂直距离和边距离的最小值
    final_distances = torch.min(distances, edge_distances)  # (N, M)
    
    # 对每个点，找到最近的三角形距离
    min_distances = torch.min(final_distances, dim=1).values  # (N,)
    
    return min_distances

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=0)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    print("gaussians._xyz.shape",gaussians._xyz.shape)
    gaussians.training_setup(opt)
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    # print("gaussians.max_radii2D.shape",gaussians.max_radii2D.shape)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    print("Background color: ", background)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, opacity, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["rend_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        gt_opacity = viewpoint_cam.gt_alpha_mask.cuda()
        rgba = torch.cat([image, opacity], dim=0)
        image = image * opacity
        gt_image = gt_image * gt_opacity
        Ll1 = l1_loss(image, gt_image)
        opacity_loss = l1_loss(opacity, gt_opacity) * 0.1
        # for the conv loss, use the original rgb image
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']

        # 预计算 surf_normal 的内积
        surf_normal_pad = F.pad(surf_normal, (2, 2, 2, 2))  # 填充以便处理边界条件
        normal_diff = torch.zeros(1, dtype=surf_normal.dtype, device=surf_normal.device)  # 初始化 normal_diff

        # 生成相邻像素的位移
        offsets = torch.tensor([(k, l) for k in range(-2, 3) for l in range(-2, 3)], device=surf_normal.device)

        # 遍历所有位移，计算对应的内积并累积 normal_diff
        '''
        for offset in offsets:
            k, l = offset
            # 获取位移后的surf_normal
            shifted_surf_normal = surf_normal_pad[:, 2+k:2+k+surf_normal.shape[1], 2+l:2+l+surf_normal.shape[2]]
            
            # 计算点积并累积
            dot_product = (surf_normal * shifted_surf_normal).sum(dim=0)
            
            # 对gt_opacity进行掩码操作
            # mask = (gt_opacity[0] > 0.5) & (F.pad(gt_opacity[0], (2, 2, 2, 2))[2+k:2+k+surf_normal.shape[1], 2+l:2+l+surf_normal.shape[2]] > 0.5)
            
            normal_diff += (1 - dot_product[mask]).sum()

        # 最终 normal_diff 结果
        normal_diff = normal_diff / (surf_normal.shape[1] * surf_normal.shape[2])
        '''     
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        Sp=gaussians.get_scaling
        r = 2
        Laniso = torch.mean(torch.maximum(torch.max(Sp, dim=1)[0] / torch.min(Sp, dim=1)[0], torch.tensor(r)) - r)
        lambda_aniso = 50

        # 新增的体积比损失
        # print("Sp.shape",Sp.shape)
        volumes = torch.prod(Sp, dim=1) # 计算每个高斯的面积
        sorted_indices = torch.argsort(volumes, descending=True) # 按面积降序排序
        alpha = 0.3 # ratio of top and bottom
        n = len(volumes)
        top_k = int(n * alpha)

        # planar loss
        if (iteration >= 7000):
            if (iteration % 1000 == 0):
                gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color) 
                gaussExtractor.gaussians.active_sh_degree = 0
                gaussExtractor.reconstruction(scene.getTrainCameras())
                depth_trunc = gaussExtractor.radius * 2.0
                voxel_size = depth_trunc / 128
                sdf_trunc = 5.0 * voxel_size
                mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
                mesh_post = post_process_mesh(mesh, cluster_to_keep=1)
                o3d.io.write_triangle_mesh(scene.model_path + "/mesh_{:04d}.ply".format(iteration), mesh_post)
                mesh_vertices = torch.tensor(mesh_post.vertices, dtype=torch.float32, device="cuda")
                mesh_faces = torch.tensor(mesh_post.triangles, dtype=torch.int64, device="cuda")
                
            # Compute distance of each Gaussian to the mesh as planar loss
            triangles = mesh_vertices[mesh_faces]
            # distances = torch.tensor(point_mesh_face_distance(mesh, triangles), dtype=torch.float32, device="cuda")
            distances = point_to_triangle_distance(gaussians.get_xyz, triangles)
            planar_loss = torch.mean(torch.abs(distances)) * 10
        else:
            planar_loss = torch.tensor(0, dtype=torch.float32, device="cuda")
        # surface_loss = surfel_loss(gaussians.get_covariance().squeeze())
        top_mean_volume = torch.mean(volumes[sorted_indices[:top_k]])
        with torch.no_grad():
            bottom_mean_volume = torch.mean(volumes[sorted_indices[-top_k:]])

        min_volum_ratio = 16.0
        Lvol_ratio = torch.max(top_mean_volume / bottom_mean_volume, torch.tensor(min_volum_ratio))-min_volum_ratio
        lambda_vol_ratio = 0.02

        plane_vector = torch.tensor([-0.0028569559637713245, -0.8715965537089316, -0.490216], device='cuda')
        mesh_distance = torch.matmul(gaussians.get_xyz, plane_vector) + 1.5171937252269088
        mesh_loss = torch.mean(mesh_distance) * 10
        regular_loss = lambda_aniso * Laniso+ lambda_vol_ratio * Lvol_ratio

        # total_loss = loss + dist_loss + normal_loss + normal_loss1 * 3 + surface_loss * 3 + regular_loss
        # normal_diff = torch.tensor(normal_diff, dtype=torch.float32, device="cuda") * 0.1
        total_loss = loss + regular_loss + dist_loss + normal_loss + opacity_loss + planar_loss
        if (iteration % 100 == 0):
            
            print("iter: ", iteration, "loss: ", loss.item(), "dist_loss: ", dist_loss.item(), "normal_loss: ", normal_loss.item(), "regular_loss: ", regular_loss.item())
            print("Ll1: ", Ll1.item(), "1 - ssim: ", 1 - ssim(image, gt_image).item(), "opacity_loss: ", opacity_loss.item(), "mesh_loss: ", mesh_loss.item(), "planar_loss: ", planar_loss.item())
            # print("Ll1: ", Ll1.item(), "1 - ssim: ", 1 - ssim(image, gt_image).item(), "opacity_loss: ", opacity_loss.item(), "normal_diff: ", normal_diff.item())
            # print("iter: ", iteration, "loss: ", loss.item(), "dist_loss: ", dist_loss.item(), "normal_loss: ", normal_loss.item(), "regular_loss: ", regular_loss.item())
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            if iteration == -1:
                num = 0
                diff_min = 0
                for i in range(gaussians.get_xyz.shape[0]):
                    mask = torch.zeros((n, 1, 3), dtype=bool)
                    mask[i][:][:] = True 
                    gaussians_new = GaussianModel(dataset.sh_degree)
                    gaussians_new._xyz = torch.cat([gaussians._xyz[:i], gaussians._xyz[i+1:]], dim=0)
                    gaussians_new._opacity = torch.cat([gaussians._opacity[:i], gaussians._opacity[i+1:]], dim=0)
                    gaussians_new._scaling = torch.cat([gaussians._scaling[:i], gaussians._scaling[i+1:]], dim=0)
                    gaussians_new._rotation = torch.cat([gaussians._rotation[:i], gaussians._rotation[i+1:]], dim=0)
                    gaussians_new._features_dc = torch.cat([gaussians._features_dc[:i], gaussians._features_dc[i+1:]], dim=0)
                    gaussians_new._features_rest = torch.cat([gaussians._features_rest[:i], gaussians._features_rest[i+1:]], dim=0)
                    loss_diff = 0
                    for j in range(len(viewpoint_stack)):
                        viewpoint_cam = viewpoint_stack[j]
            
                        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                        render_pkg_new = render(viewpoint_cam, gaussians_new, pipe, background)
                        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                        image_new, viewspace_point_tensor_new, visibility_filter_new, radii_new = render_pkg_new["render"], render_pkg_new["viewspace_points"], render_pkg_new["visibility_filter"], render_pkg_new["radii"]

                        gt_image = viewpoint_cam.original_image.cuda()
                        gt_image = gt_image[:3, :, :]
                        Ll1 = l1_loss(image, gt_image)
                        Ll1_new = l1_loss(image_new, gt_image)
                        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
                        loss_new = (1.0 - opt.lambda_dssim) * Ll1_new + opt.lambda_dssim * (1.0 - ssim(image_new, gt_image))
                        loss_diff += loss_new - loss
                    print("i: ", i, "loss_diff: ", loss_diff)
                    if loss_diff > diff_min:
                        diff_min = loss_diff
                    if loss_diff < 0.002:
                        gaussians._opacity[i] = -1000
                        num += 1
                print("diff_min: ", diff_min)
                print("prune points: ", num)
                size_threshold = 20 
                gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):     
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            gt_image = gt_image[:3, :, :]
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 3_000, 7_000, 8000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1, 3_000, 7_000, 8000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")