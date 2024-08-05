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
from random import randint
from utils.loss_utils import l1_loss, ssim, loss_cls_3d
from gaussian_renderer import render, network_gui
from os import makedirs
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import wandb
import json
import torchvision
from sklearn.decomposition import PCA
from PIL import Image

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def feature_to_rgb(features_gpu):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    features = features_gpu.detach().cpu()
    print("features_gpu.device: ", features_gpu.device)
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, use_wandb):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    print("rotation: ", gaussians.get_rotation)
    gaussians.training_setup(opt)
    num_classes = dataset.num_classes
    print("Num classes: ",num_classes)
    classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)
    classifier.cuda()
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

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
        
        # print("rotation: ", gaussians.get_rotation)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        torch.cuda.synchronize()
        image, viewspace_point_tensor, visibility_filter, radii, objects = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["render_object"]
        test_dir = "C:\\ren\code\\2d-gaussian-splatting\\test\\input"
        makedirs(test_dir, exist_ok=True)
        if (iteration % 10 == 0):        
            rgb_mask = feature_to_rgb(objects)
            Image.fromarray(rgb_mask).save(os.path.join(test_dir, '{0:05d}'.format(iteration) + ".png"))
        # print("viewspace_point_tensor.norm: ", torch.norm(viewspace_point_tensor.grad[visibility_filter], dim=-1, keepdim=True))
        torch.cuda.synchronize()
        # Object Loss
        gt_obj = viewpoint_cam.objects.cuda().long()
        logits = classifier(objects)
        loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
        loss_obj = loss_obj / torch.log(torch.tensor(num_classes))  # normalize to (0,1)

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss_obj_3d = None
        # check nan
        assert not torch.isnan(Ll1).any(), "L1 loss is nan"
        assert not torch.isnan(loss_obj).any(), "Object loss is nan"
        assert not torch.isnan(gaussians._xyz).any(), "XYZ is nan"
        assert not torch.isnan(gaussians._objects_dc).any(), "Objects is nan"
        assert not torch.isnan(gaussians._features_dc).any(), "Features is nan"
        assert not torch.isnan(gaussians._opacity).any(), "Opacity is nan"
        assert not torch.isnan(gaussians.max_radii2D).any(), "Radii is nan"
        assert not torch.isnan(gaussians._scaling).any(), "SH is nan"
        assert not torch.isnan(gaussians.xyz_gradient_accum).any(), "SH weights is nan"
        assert not torch.isnan(gaussians._rotation).any(), "SH offsets is nan"

        if iteration % opt.reg3d_interval == 0:
            # regularize at certain intervals
            logits3d = classifier(gaussians._objects_dc.permute(2,0,1))
            prob_obj3d = torch.softmax(logits3d,dim=0).squeeze().permute(1,0)
            loss_obj_3d = loss_cls_3d(gaussians._xyz.squeeze().detach(), prob_obj3d, opt.reg3d_k, opt.reg3d_lambda_val, opt.reg3d_max_points, opt.reg3d_sample_size)
            # print ("Ll1: ", Ll1, "Loss_obj: ", loss_obj, "Loss_obj_3d: ", loss_obj_3d)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj * 1 + loss_obj_3d * 1
        else:
            # print ("Ll1: ", Ll1, "Loss_obj: ", loss_obj)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + loss_obj * 1
   
        assert not torch.isnan(Ll1).any(), "L1 loss is nan"
        assert not torch.isnan(loss_obj).any(), "Object loss is nan"
        assert not torch.isnan(gaussians._xyz).any(), "XYZ is nan"
        assert not torch.isnan(gaussians._objects_dc).any(), "Objects is nan"
        assert not torch.isnan(gaussians._features_dc).any(), "Features is nan"
        assert not torch.isnan(gaussians._opacity).any(), "Opacity is nan"
        assert not torch.isnan(gaussians.max_radii2D).any(), "Radii is nan"
        assert not torch.isnan(gaussians._scaling).any(), "SH is nan"
        assert not torch.isnan(gaussians.xyz_gradient_accum).any(), "SH weights is nan"
        assert not torch.isnan(gaussians._rotation).any(), "SH offsets is nan"

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss
        total_loss.backward()
        # print("viewspace_point_tensor.norm: ", torch.norm(viewspace_point_tensor.grad[visibility_filter], dim=-1, keepdim=True))
        iter_end.record()
        assert not torch.isnan(Ll1).any(), "L1 loss is nan"
        assert not torch.isnan(loss_obj).any(), "Object loss is nan"
        assert not torch.isnan(gaussians._xyz).any(), "XYZ is nan"
        assert not torch.isnan(gaussians._objects_dc).any(), "Objects is nan"
        assert not torch.isnan(gaussians._features_dc).any(), "Features is nan"
        assert not torch.isnan(gaussians._opacity).any(), "Opacity is nan"
        assert not torch.isnan(gaussians.max_radii2D).any(), "Radii is nan"
        assert not torch.isnan(gaussians._scaling).any(), "SH is nan"
        assert not torch.isnan(gaussians.xyz_gradient_accum).any(), "SH weights is nan"
        assert not torch.isnan(gaussians._rotation).any(), "SH offsets is nan"
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
            assert not torch.isnan(Ll1).any(), "L1 loss is nan"
            assert not torch.isnan(loss_obj).any(), "Object loss is nan"
            assert not torch.isnan(gaussians._xyz).any(), "XYZ is nan"
            assert not torch.isnan(gaussians._objects_dc).any(), "Objects is nan"
            assert not torch.isnan(gaussians._features_dc).any(), "Features is nan"
            assert not torch.isnan(gaussians._opacity).any(), "Opacity is nan"
            assert not torch.isnan(gaussians.max_radii2D).any(), "Radii is nan"
            assert not torch.isnan(gaussians._scaling).any(), "SH is nan"
            assert not torch.isnan(gaussians.xyz_gradient_accum).any(), "SH weights is nan"
            assert not torch.isnan(gaussians._rotation).any(), "SH offsets is nan"
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), loss_obj_3d, use_wandb)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.save(classifier.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}".format(iteration),'classifier.pth'))

            assert not torch.isnan(gaussians.xyz_gradient_accum).any(), "SH weights is nan"
            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                assert not torch.isnan(gaussians.xyz_gradient_accum).any(), "SH weights is nan"
                # print("viewspace_point_tensor.norm: ", torch.norm(viewspace_point_tensor.grad[visibility_filter], dim=-1, keepdim=True))
                assert not torch.isnan(viewspace_point_tensor.grad[visibility_filter]).any(), "viewspace_point_tensor is nan"
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                assert not torch.isnan(gaussians.xyz_gradient_accum).any(), "SH weights is nan"

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                max_norm = 1.0
                torch.nn.utils.clip_grad_norm_(gaussians._xyz, max_norm, norm_type=2)
                assert not torch.isnan(Ll1).any(), "L1 loss is nan"
                assert not torch.isnan(loss_obj).any(), "Object loss is nan"
                assert not torch.isnan(gaussians._xyz).any(), "XYZ is nan"
                assert not torch.isnan(gaussians._objects_dc).any(), "Objects is nan"
                assert not torch.isnan(gaussians._features_dc).any(), "Features is nan"
                assert not torch.isnan(gaussians._opacity).any(), "Opacity is nan"
                assert not torch.isnan(gaussians.max_radii2D).any(), "Radii is nan"
                assert not torch.isnan(gaussians._scaling).any(), "SH is nan"
                assert not torch.isnan(gaussians.xyz_gradient_accum).any(), "SH weights is nan"
                assert not torch.isnan(gaussians._rotation).any(), "SH offsets is nan"
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                assert not torch.isnan(Ll1).any(), "L1 loss is nan"
                assert not torch.isnan(loss_obj).any(), "Object loss is nan"
                assert not torch.isnan(gaussians._xyz).any(), "XYZ is nan"
                assert not torch.isnan(gaussians._objects_dc).any(), "Objects is nan"
                assert not torch.isnan(gaussians._features_dc).any(), "Features is nan"
                assert not torch.isnan(gaussians._opacity).any(), "Opacity is nan"
                assert not torch.isnan(gaussians.max_radii2D).any(), "Radii is nan"
                assert not torch.isnan(gaussians._scaling).any(), "SH is nan"
                assert not torch.isnan(gaussians.xyz_gradient_accum).any(), "SH weights is nan"
                assert not torch.isnan(gaussians._rotation).any(), "SH offsets is nan"
                cls_optimizer.step()
                cls_optimizer.zero_grad()
                assert not torch.isnan(Ll1).any(), "L1 loss is nan"
                assert not torch.isnan(loss_obj).any(), "Object loss is nan"
                assert not torch.isnan(gaussians._xyz).any(), "XYZ is nan"
                assert not torch.isnan(gaussians._objects_dc).any(), "Objects is nan"
                assert not torch.isnan(gaussians._features_dc).any(), "Features is nan"
                assert not torch.isnan(gaussians._opacity).any(), "Opacity is nan"
                assert not torch.isnan(gaussians.max_radii2D).any(), "Radii is nan"
                assert not torch.isnan(gaussians._scaling).any(), "SH is nan"
                assert not torch.isnan(gaussians.xyz_gradient_accum).any(), "SH weights is nan"
                assert not torch.isnan(gaussians._rotation).any(), "SH offsets is nan"


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, loss_obj_3d, use_wandb):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
  
    if use_wandb:
        if loss_obj_3d:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(), "train_loss_patches/loss_obj_3d": loss_obj_3d.item(), "iter_time": elapsed, "iter": iteration})
        else:
            wandb.log({"train_loss_patches/l1_loss": Ll1.item(), "train_loss_patches/total_loss": loss.item(), "iter_time": elapsed, "iter": iteration})

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
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if use_wandb:
                    wandb.log({config['name'] + "/loss_viewpoint - l1_loss": l1_test, config['name'] + "/loss_viewpoint - psnr": psnr_test})
        if use_wandb:
            wandb.log({"scene/opacity_histogram": scene.gaussians.get_opacity, "total_points": scene.gaussians.get_xyz.shape[0], "iter": iteration})
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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000, 7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # Add an argument for the configuration file
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--use_wandb", action='store_true', default=False, help="Use wandb to record loss value")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # Read and parse the configuration file
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the JSON configuration file: {e}")
        exit(1)

    args.densify_until_iter = config.get("densify_until_iter", 15000)
    args.num_classes = config.get("num_classes", 20)
    args.reg3d_interval = config.get("reg3d_interval", 2)
    args.reg3d_k = config.get("reg3d_k", 5)
    args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
    args.reg3d_max_points = config.get("reg3d_max_points", 300000)
    args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)

    print("Optimizing " + args.model_path)

    if args.use_wandb:
        wandb.init(project="gaussian-splatting")
        wandb.config.args = args
        wandb.run.name = args.model_path

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.use_wandb)

    # All done
    print("\nTraining complete.")