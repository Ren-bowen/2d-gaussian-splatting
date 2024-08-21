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

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def render_net_image(render_pkg, render_items, render_mode, camera):
    output = render_items[render_mode].lower()
    if output == 'alpha':
        net_image = render_pkg["alpha"]
        net_image = (net_image - net_image.min()) / (net_image.max() - net_image.min())
    elif output == 'depth':
        net_image = render_pkg["mean_depth"]
        net_image = (net_image - net_image.min()) / (net_image.max() - net_image.min())
    elif output == 'normal':
        net_image = depth_to_normal(render_pkg["mean_depth"], camera).permute(2,0,1)
        net_image = (net_image+1)/2
    elif output == 'edge':
        net_image = gradient_map(render_pkg["render"])
    elif output == 'curvature':
        net_image = gradient_map(depth_to_normal(render_pkg["mean_depth"], camera).permute(2,0,1))
    else:
        net_image = render_pkg["render"]
    
    #Make sure the rendering image is Shape of (3, H, W) or (1, H, W)

    if net_image.shape[0]==1:
        net_image = colormap(net_image)
    return net_image 