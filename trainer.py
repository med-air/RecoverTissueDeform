import torch.optim as optim
import random
import torch as th
import torch.nn.functional as F
import sys

sys.path.append('../')  # add relative path
from torch import nn as nn
import numpy as np
from pytorch3d.structures.volumes import Volumes
from pytorch3d.ops import add_pointclouds_to_volumes
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    SfMPerspectiveCameras,
    PerspectiveCameras,
    SfMOrthographicCameras,
    OpenGLPerspectiveCameras,
    FoVOrthographicCameras,
    FoVPerspectiveCameras
)
from loss import first_order_smoothness_loss, warp, compute_cen, compute_occ, smooth_loss, robust_l1
from dataloader import Surgical_dataset
from dataloader import Surgical_dataset_eval
from torch.utils.data import DataLoader
import logging
import utils
from loss import SSIM
from network import VecInt

device = "cuda"
K1_inv = th.tensor(np.linalg.inv([[732.24990637, 0., 372.81334305],
                                  [0., 732.24990637, 276.87692261],
                                  [0., 0., 1.]])).float().cuda().unsqueeze(0)
K1 = th.tensor([[732.24990637, 0., 372.81334305],
                                  [0., 732.24990637, 276.87692261],
                                  [0., 0., 1.]]).float().cuda().unsqueeze(0)
R1 = th.tensor([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]).float().cuda().unsqueeze(0)
R2 = th.tensor([[0.999930, -0.010532, -0.005401],
                [0.010504, 0.999932, -0.005049],
                [0.005454, 0.004992, 0.999973]]).float().cuda().unsqueeze(0)
T1 = th.tensor([[0, 0, 0]]).float().cuda()
T2 = th.tensor([[-4.551925, -0.015196, -0.042822]]).float().cuda()

focal = 732.24990637
baseline = 4.552

fx_screen = 732.24990637 * (740 / 540)
fy_screen = 732.24990637
px_screen = 372.81334305
py_screen = 276.87692261
image_width = 740
image_height = 540
fx = fx_screen * 2.0 / image_width
fy = fy_screen * 2.0 / image_height

px = - (px_screen - image_width / 2.0) * 2.0 / image_width
py = - (py_screen - image_height / 2.0) * 2.0 / image_height

cameras_right = PerspectiveCameras(focal_length=-th.Tensor([[fx, fy]]), principal_point=th.Tensor([[px, py]]),
                                   R=R2, T=T2, device=device)

cameras_left = PerspectiveCameras(focal_length=-th.Tensor([[fx, fy]]), principal_point=th.Tensor([[px, py]]),
                                  R=R1, T=T1, device=device)
image_size = (image_height, image_width)
raster_settings = PointsRasterizationSettings(
    image_size=image_size,
    radius=0.01,
    points_per_pixel=10,
    bin_size=100
)

renderer_left = PointsRenderer(
    rasterizer=PointsRasterizer(cameras=cameras_left, raster_settings=raster_settings),
    compositor=AlphaCompositor(-0.01)
)

from skimage.metrics import structural_similarity
import cv2
from math import log10, sqrt

def l1_norm(img1, img2):
    mask = (img1 > 0) * (img2 > 0)
    return np.sum(mask * np.abs(img1 - img2)) / np.sum(mask) * 255
ssim = SSIM()
def ssim_sim(img1, img2):
    ssim = structural_similarity(img1, img2, channel_axis=2, full=True)[1]
    mask = (img1 > 0) * (img2 > 0)
    ssim = np.sum(ssim * mask) / np.sum(mask)
    return ssim
def PSNR(original, compressed):
    mask = (original > 0) * (compressed > 0)
    mse = np.sum(mask * (original * 255 - compressed * 255) ** 2)/np.sum(mask)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
def jacobian(deformation):
    mask = (th.sum(deformation[:,:,:-1,:-1,:-1],dim=1) != 0).reshape(-1)
    t1 = (deformation[0,:,1:,:-1,:-1] + th.tensor([1,0,0]).reshape(1,3,1,1,1).cuda() - deformation[0,:,:-1,:-1,:-1]).reshape(1, 3,-1)
    t2 = (deformation[0,:,:-1,1:,:-1] + th.tensor([0,1,0]).reshape(1,3,1,1,1).cuda() - deformation[0,:,:-1,:-1,:-1]).reshape(1, 3,-1)
    t3 = (deformation[0,:,:-1,:-1,1:] + th.tensor([0,0,1]).reshape(1,3,1,1,1).cuda() - deformation[0,:,:-1,:-1,:-1]).reshape(1, 3,-1)
    res = th.linalg.det(th.cat([t1,t2,t3], dim=0).permute(2,0,1))
    return ((th.sum(mask * res>0))/th.sum(mask)).cpu().numpy()

def trainer_surgery(args, flow_net, refine_net, snapshot_path, init_epoch=0):
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    batch_size = args.batch_size * args.n_gpu
    device = "cuda" if th.cuda.is_available() else "cpu"
    K1_inv = th.tensor(np.linalg.inv([[732.24990637, 0., 372.81334305],
                                      [0., 732.24990637, 276.87692261],
                                      [0., 0., 1.]])).float().cuda().unsqueeze(0)

    focal = 732.24990637
    baseline = 4.552
    image_width = 512
    image_height = 512

    base_lr = args.base_lr

    db_train = Surgical_dataset(args.train_data)
    db_val = Surgical_dataset_eval(args.eval_data)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False)
    if args.n_gpu > 1:
        flow_net = nn.DataParallel(flow_net)

    flow_optimizer = optim.Adam(params=flow_net.parameters(), lr=base_lr)
    flow_scheduler = optim.lr_scheduler.LinearLR(flow_optimizer, start_factor=1., end_factor=0.1, total_iters=20,
                                                   last_epoch=- 1, verbose=False)
    if refine_net is not None:
        refine_optimizer = optim.Adam(params=refine_net.parameters(), lr=base_lr)
        refine_scheduler = optim.lr_scheduler.LinearLR(refine_optimizer, start_factor=1., end_factor=0.1, total_iters=20,
                                               last_epoch=- 1, verbose=False)
    max_epoch = args.max_epochs

    best_loss = np.inf
    vecint = VecInt([64, 64, 64], 7).to(device)


    def compute_loss(img1, img2, flow_fwd, flow_bwd, occ1, occ2):
        img1_warp = warp(img2, flow_fwd)
        occ2_warp = (warp(occ2, flow_fwd).squeeze(1).detach()==1).long()
        occ_map = compute_occ(flow_bwd)
        l_photo = compute_cen(img1, img1_warp, occ_map * occ1 * occ2_warp)
        l_smooth = first_order_smoothness_loss(img1, flow_fwd)
        loss = l_photo + 4 * l_smooth
        return loss, l_photo.detach().cpu(), l_smooth.detach().cpu()

    for epoch_num in range(init_epoch, max_epoch):

        flow_net.train()
        if refine_net is not None:
            refine_net.train()
        loss_summary = []
        loss_flow_summary = []
        loss_smooth_summary = []
        loss_pixel_summary = []
        loss_dif_summary = []
        loss_cycle_summary = []
        loss_depth_summary = []
        for i_batch, sampled_batch in enumerate(train_loader):
            left_0 = sampled_batch["left_t0"].cuda()
            right_0 = sampled_batch["right_t0"].cuda()
            disp_0 = sampled_batch["disp_t0"].cuda()
            tool_mask_0 = sampled_batch["tool_t0"].cuda()
            tool_right_0 = sampled_batch["tool_t0_r"].cuda()

            left_1 = sampled_batch["left_t1"].cuda()
            right_1 = sampled_batch["right_t1"].cuda()
            disp_1 = sampled_batch["disp_t1"].cuda()
            tool_mask_1 = sampled_batch["tool_t1"].cuda()
            tool_right_1 = sampled_batch["tool_t1_r"].cuda()

            left_2 = sampled_batch["left_t2"].cuda()
            right_2 = sampled_batch["right_t2"].cuda()
            disp_2 = sampled_batch["disp_t2"].cuda()
            tool_mask_2 = sampled_batch["tool_t2"].cuda()
            tool_right_2 = sampled_batch["tool_t2_r"].cuda()

            left_3 = sampled_batch["left_t3"].cuda()
            right_3 = sampled_batch["right_t3"].cuda()
            disp_3 = sampled_batch["disp_t3"].cuda()
            tool_mask_3 = sampled_batch["tool_t3"].cuda()
            tool_right_3 = sampled_batch["tool_t3_r"].cuda()

            left_4 = sampled_batch["left_t4"].cuda()
            right_4 = sampled_batch["right_t4"].cuda()
            disp_4 = sampled_batch["disp_t4"].cuda()
            tool_mask_4 = sampled_batch["tool_t4"].cuda()
            tool_right_4 = sampled_batch["tool_t4_r"].cuda()

            left_t0 = th.cat([left_0, left_1, left_2, left_3, left_4, left_3, left_2, left_1], dim=0)
            left_t1 = th.cat([left_1, left_2, left_3, left_4, left_3, left_2, left_1, left_0], dim=0)

            right_t1 = th.cat([right_1, right_2, right_3, right_4], dim=0)
            tool_mask1_r = th.cat([tool_right_1, tool_right_2, tool_right_3, tool_right_4], dim=0)

            disp_t0 = th.cat([disp_0, disp_1, disp_2, disp_3, disp_4, disp_3, disp_2, disp_1], dim=0)
            disp_t1 = th.cat([disp_1, disp_2, disp_3, disp_4, disp_3, disp_2, disp_1, disp_0], dim=0)

            tool_mask0 = th.cat(
                [tool_mask_0, tool_mask_1, tool_mask_2, tool_mask_3, tool_mask_4, tool_mask_3, tool_mask_2,
                 tool_mask_1], dim=0)
            tool_mask1 = th.cat(
                [tool_mask_1, tool_mask_2, tool_mask_3, tool_mask_4, tool_mask_3, tool_mask_2, tool_mask_1,
                 tool_mask_0], dim=0)

            l = len(left_t0)

            left_t0 = left_t0 * tool_mask0
            left_t1 = left_t1 * tool_mask1
            x_base = th.linspace(226, 737, image_width).repeat(l, image_height, 1) \
                .float().reshape(l, 1, -1).cuda()
            y_base = th.linspace(0, image_height - 1, image_height).repeat(l, image_width, 1).transpose(1, 2) \
                .float().reshape(l, 1, -1).cuda()

            fwd_flow = flow_net(left_t0, left_t1)

            with th.no_grad():
                bwd_flow = flow_net(left_t1, left_t0)
            loss_list = [compute_loss(left_t0, left_t1, i, j, tool_mask0.squeeze(1), tool_mask1) for i, j in
                         zip(fwd_flow, bwd_flow)]
            loss_flow = sum([0.8 ** (11 - j) * i[0] for i, j in zip(loss_list, range(12))])

            fwd_final = fwd_flow[-1]

            depth_gt_t0 = focal * baseline / (disp_t0 + 0.000001)
            a = (th.abs(depth_gt_t0 - th.roll(depth_gt_t0, 3, 2)) <= 1).float()
            b = (th.abs(depth_gt_t0 - th.roll(depth_gt_t0, -3, 2)) <= 1).float()
            c = (th.abs(depth_gt_t0 - th.roll(depth_gt_t0, 3, 3)) <= 1).float()
            d = (th.abs(depth_gt_t0 - th.roll(depth_gt_t0, -3, 3)) <= 1).float()
            inline_t0 = (a + b + c + d) > 1
            inline_t0 = inline_t0.reshape(l, -1)
            depth_gt_t0 = depth_gt_t0.reshape(l, 1, -1)
            outlier_t0 = depth_gt_t0[:, 0, :] < 128
            outlier_t0 = outlier_t0 * inline_t0
            outlier_t0 = (outlier_t0 * tool_mask0.reshape(l, -1))
            xy_gt_t0 = th.cat([x_base, y_base, th.ones_like(x_base)], dim=1) * depth_gt_t0
            points_t0_gt = th.bmm(K1_inv.repeat(l, 1, 1), xy_gt_t0).transpose(1, 2)

            depth_gt_t1 = focal * baseline / (disp_t1 + 0.000001)
            a = (th.abs(depth_gt_t1 - th.roll(depth_gt_t1, 3, 2)) <= 1).float()
            b = (th.abs(depth_gt_t1 - th.roll(depth_gt_t1, -3, 2)) <= 1).float()
            c = (th.abs(depth_gt_t1 - th.roll(depth_gt_t1, 3, 3)) <= 1).float()
            d = (th.abs(depth_gt_t1 - th.roll(depth_gt_t1, -3, 3)) <= 1).float()
            inline_t1 = (a + b + c + d) > 1
            inline_t1 = inline_t1.reshape(l, -1)
            depth_gt_t1 = depth_gt_t1.reshape(l, 1, -1)
            outlier_t1 = depth_gt_t1[:, 0, :] < 128
            outlier_t1 = outlier_t1 * inline_t1
            outlier_t1 = (outlier_t1.reshape(l, 1, 512, 512) * tool_mask1)
            xy_gt_t1 = th.cat([x_base, y_base, th.ones_like(x_base)], dim=1) * depth_gt_t1
            points_t1_gt = th.bmm(K1_inv.repeat(l, 1, 1), xy_gt_t1).transpose(1, 2)

            new_coords = warp(points_t1_gt.permute(0, 2, 1).reshape(l, 3, 512, 512),
                              fwd_final).reshape(l, 3, -1).permute(0, 2, 1)
            filter_t1 = warp(outlier_t1.reshape(l, 1, 512, 512), fwd_final).reshape(l, -1)
            delta_deform = (new_coords - points_t0_gt)
            delta_deform_filter = [i[(j == 1) & (k == 1)] for i, j, k in zip(delta_deform, outlier_t0, filter_t1)]
            points_filter = [i[(j == 1) & (k == 1)] for i, j, k in zip(points_t0_gt, outlier_t0, filter_t1)]
            mid_x = th.median(points_filter[0][:, 0]).detach().cpu().numpy()
            mid_y = th.median(points_filter[0][:, 1]).detach().cpu().numpy()
            mid_z = th.median(points_filter[0][:, 2]).detach().cpu().numpy()

            points_filter_complete = [i[(j == 1)] for i, j in zip(points_t0_gt, outlier_t0)]
            rgb_filter_complete = [i[(j == 1)] for i, j in zip(left_t0.reshape(l, 3, -1).permute(0, 2, 1), outlier_t0)]

            initial_volumes_t0 = Volumes(
                features=th.zeros(l, 3, 64, 64, 64),
                densities=th.zeros(l, 1, 64, 64, 64),
                volume_translation=[-mid_x, -mid_y, -mid_z],
                voxel_size=1.0,
            ).cuda()
            deform_cloud_fwd = Pointclouds(points=points_filter, features=delta_deform_filter)
            flow_volume_tri = add_pointclouds_to_volumes(
                pointclouds=deform_cloud_fwd,
                initial_volumes=initial_volumes_t0,
                mode="trilinear",
            ).features()

            semantic_volumes_t0 = Volumes(
                features=th.zeros(l, 3, 64, 64, 64),
                densities=th.zeros(l, 1, 64, 64, 64),
                volume_translation=[-mid_x, -mid_y, -mid_z],
                voxel_size=1.0,
            ).cuda()
            semantic_cloud_fwd = Pointclouds(points=points_filter_complete, features=rgb_filter_complete)
            semantic_volume_tri = add_pointclouds_to_volumes(
                pointclouds=semantic_cloud_fwd,
                initial_volumes=semantic_volumes_t0,
                mode="trilinear",
            ).features()

            flow_volume_previous = th.zeros(1, 3, 64, 64, 64).cuda()
            l_smooth = 0.
            l_dif = 0.
            l_pixel = 0.
            l_depth = 0.
            # l_temporal = 0.
            point_previous = points_filter_complete[0]

            previous_history = []
            for i in range(l):
                if i == l // 2:
                    flow_volume_previous = th.zeros(1, 3, 64, 64, 64).cuda()
                velocity = refine_net(flow_volume_tri[[i]], semantic_volume_tri[[i]], flow_volume_previous)
                l_smooth += th.mean(smooth_loss(velocity))
                flow_volume_refine = vecint(velocity)
                point = points_filter[i]
                dif_out = F.grid_sample(flow_volume_refine,
                                        ((point + th.tensor(
                                            [32 - mid_x, 32 - mid_y, 32 - mid_z]).float().cuda()) / 32. - 1) \
                                        .unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                        align_corners=True).squeeze().transpose(0, 1)

                l_dif += th.mean(robust_l1(dif_out - delta_deform_filter[i].detach()))

                if i < l // 2:
                    dif_out_previous = F.grid_sample(flow_volume_refine,
                                                     ((point_previous + th.tensor([32 - mid_x, 32 - mid_y,
                                                                                   32 - mid_z]).float().cuda()) / 32. - 1) \
                                                     .unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                                     align_corners=True).squeeze().transpose(0, 1)
                    point_previous = dif_out_previous + point_previous

                    previous_history.append(point_previous)
                    point_deform_2d = K1[0] @ point_previous.transpose(0, 1)
                    point_deform_2d = point_deform_2d[:2] / point_deform_2d[2]
                    point_deform_2d[0] = point_deform_2d[0] - 226
                    point_deform_2d = point_deform_2d / 256. - 1.
                    point_deform_2d_rgb = F.grid_sample(left_t1[[i]],
                                                        point_deform_2d.unsqueeze(0).transpose(1, 2).unsqueeze(0))
                    deform_mask = F.grid_sample(tool_mask1[[i]],
                                                point_deform_2d.unsqueeze(0).transpose(1, 2).unsqueeze(0)).detach()

                    occ = outlier_t0[0].clone().detach()
                    occ[(outlier_t0[0] == 1)] *= deform_mask[:, 0].squeeze()
                    occ[(outlier_t0[0] == 1)][th.max(th.abs(point_deform_2d), dim=0)[0] > 1] *= 0

                    out_img = th.zeros(1, 3, 512, 512).cuda()
                    out_img = out_img.reshape(3, -1)
                    out_img[:, (outlier_t0[0] == 1)] += point_deform_2d_rgb.squeeze()
                    out_img = out_img.reshape(3, 512, 512)
                    l_pixel += compute_cen(left_t0[[0]], out_img.unsqueeze(0), occ.reshape(1, 1, 512, 512))

                    point_deform_2d = K1[0] @ th.linalg.inv(R2)[0] @ (point_previous + T2).transpose(0, 1)
                    point_deform_2d = point_deform_2d[:2] / point_deform_2d[2]
                    point_deform_2d = point_deform_2d.transpose(0, 1)
                    point_deform_2d = point_deform_2d / th.tensor([370., 270.]).cuda() - 1.
                    point_deform_2d_rgb = F.grid_sample(right_t1[[i]],
                                                        point_deform_2d.unsqueeze(0).unsqueeze(0))
                    deform_mask = F.grid_sample(tool_mask1_r[[i]],
                                                point_deform_2d.unsqueeze(0).unsqueeze(0)).detach()
                    occ = outlier_t0[0].clone().detach()
                    occ[(outlier_t0[0] == 1)] *= deform_mask[:, 0].squeeze()
                    occ[(outlier_t0[0] == 1)][th.max(th.abs(point_deform_2d), dim=1)[0] > 1] *= 0

                    out_img = th.zeros(1, 3, 512, 512).cuda()
                    out_img = out_img.reshape(3, -1)
                    out_img[:, (outlier_t0[0] == 1)] += point_deform_2d_rgb.squeeze()
                    out_img = out_img.reshape(3, 512, 512)
                    l_depth += th.sum(occ.reshape(512, 512) *
                                     th.arccos(th.sum(left_t0[0] * out_img, dim=0)
                                               / (th.linalg.norm(left_t0[0], dim=0, ord=2)
                                                  * th.linalg.norm(out_img, dim=0, ord=2) + 0.00001))) / th.sum(occ)

                else:
                    dif_out_previous = [F.grid_sample(flow_volume_refine,
                                                     ((point_previous + th.tensor([32 - mid_x, 32 - mid_y,
                                                                                   32 - mid_z]).float().cuda()) / 32. - 1) \
                                                     .unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                                     align_corners=True).squeeze().transpose(0, 1) for point_previous in previous_history[-1-(i-(l//2)):]]
                    for j in range(1+(i-(l//2))):
                        previous_history[-1 - (i - (l // 2)):][j] += dif_out_previous[j]

                flow_volume_previous = velocity

            l_smooth /= l
            l_dif /= l
            l_pixel /= (l//2)
            l_depth /= (l//2)


            l_cycle = sum([th.mean(robust_l1(point_previous - points_filter_complete[0])) for point_previous in previous_history]) / (l//2)


            loss = loss_flow + (l_dif + l_pixel + l_depth + 0.1 * l_cycle + 0.1 * l_smooth)

            loss_summary.append(loss.detach().cpu().numpy())
            loss_flow_summary.append(loss_flow.detach().cpu().numpy())
            loss_smooth_summary.append(l_smooth.detach().cpu().numpy())
            loss_dif_summary.append(l_dif.detach().cpu().numpy())
            loss_pixel_summary.append(l_pixel.detach().cpu().numpy())
            loss_cycle_summary.append(l_cycle.detach().cpu().numpy())
            loss_depth_summary.append(l_depth.detach().cpu().numpy())

            flow_optimizer.zero_grad()
            refine_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(flow_net.parameters(), 12.0)
            nn.utils.clip_grad_norm_(refine_net.parameters(), 12.0)
            flow_optimizer.step()
            refine_optimizer.step()

        logging.info("- Train metrics: " + str(np.mean(loss_summary)))
        logging.info("- Train metrics flow: " + str(np.mean(loss_flow_summary)))
        logging.info("- Train metrics smooth: " + str(np.mean(loss_smooth_summary)))
        logging.info("- Train metrics dif: " + str(np.mean(loss_dif_summary)))
        logging.info("- Train metrics pixel: " + str(np.mean(loss_pixel_summary)))
        logging.info("- Train metrics depth: " + str(np.mean(loss_depth_summary)))
        logging.info("- Train metrics cycle: " + str(np.mean(loss_cycle_summary)))
        refine_scheduler.step()
        flow_scheduler.step()

        flow_net.eval()
        if refine_net is not None:
            refine_net.eval()

        l1_loss_left = []
        sim_loss_left = []
        psr_left = []
        jacobian_left = []
        for i_batch, sampled_batch in enumerate(val_loader):
            length = len(sampled_batch["left"])
            flow_volume_previous = th.zeros(1, 3, 64, 64, 64).cuda()
            with th.no_grad():
                for t in range(length-1):
                    left_t0 = sampled_batch["left"][t].cuda()
                    disp_t0 = sampled_batch["disp"][t].cuda()
                    tool_mask0 = sampled_batch["tool"][t].cuda()

                    left_t1 = sampled_batch["left"][t+1].cuda()
                    disp_t1 = sampled_batch["disp"][t+1].cuda()
                    tool_mask1 = sampled_batch["tool"][t+1].cuda()

                    l = len(left_t0)

                    left_t0 = left_t0 * tool_mask0
                    left_t1 = left_t1 * tool_mask1
                    x_base = th.linspace(226, 737, image_width).repeat(l, image_height, 1) \
                        .float().reshape(l, 1, -1).cuda()
                    y_base = th.linspace(0, image_height - 1, image_height).repeat(l, image_width, 1).transpose(1, 2) \
                        .float().reshape(l, 1, -1).cuda()

                    fwd_final = flow_net(left_t0, left_t1)[-1]

                    depth_gt_t0 = focal * baseline / (disp_t0 + 0.000001)
                    a = (th.abs(depth_gt_t0 - th.roll(depth_gt_t0, 3, 2)) <= 1).float()
                    b = (th.abs(depth_gt_t0 - th.roll(depth_gt_t0, -3, 2)) <= 1).float()
                    c = (th.abs(depth_gt_t0 - th.roll(depth_gt_t0, 3, 3)) <= 1).float()
                    d = (th.abs(depth_gt_t0 - th.roll(depth_gt_t0, -3, 3)) <= 1).float()
                    inline_t0 = (a + b + c + d) > 1
                    inline_t0 = inline_t0.reshape(l, -1)
                    depth_gt_t0 = depth_gt_t0.reshape(l, 1, -1)
                    outlier_t0 = depth_gt_t0[:, 0, :] < 128
                    outlier_t0 = outlier_t0 * inline_t0
                    outlier_t0 = (outlier_t0 * tool_mask0.reshape(l, -1))
                    xy_gt_t0 = th.cat([x_base, y_base, th.ones_like(x_base)], dim=1) * depth_gt_t0
                    points_t0_gt = th.bmm(K1_inv.repeat(l, 1, 1), xy_gt_t0).transpose(1, 2)

                    depth_gt_t1 = focal * baseline / (disp_t1 + 0.000001)
                    a = (th.abs(depth_gt_t1 - th.roll(depth_gt_t1, 3, 2)) <= 1).float()
                    b = (th.abs(depth_gt_t1 - th.roll(depth_gt_t1, -3, 2)) <= 1).float()
                    c = (th.abs(depth_gt_t1 - th.roll(depth_gt_t1, 3, 3)) <= 1).float()
                    d = (th.abs(depth_gt_t1 - th.roll(depth_gt_t1, -3, 3)) <= 1).float()
                    inline_t1 = (a + b + c + d) > 1
                    inline_t1 = inline_t1.reshape(l, -1)
                    depth_gt_t1 = depth_gt_t1.reshape(l, 1, -1)
                    outlier_t1 = depth_gt_t1[:, 0, :] < 128
                    outlier_t1 = outlier_t1 * inline_t1
                    outlier_t1 = (outlier_t1.reshape(l, 1, 512, 512) * tool_mask1)
                    xy_gt_t1 = th.cat([x_base, y_base, th.ones_like(x_base)], dim=1) * depth_gt_t1
                    points_t1_gt = th.bmm(K1_inv.repeat(l, 1, 1), xy_gt_t1).transpose(1, 2)


                    new_coords = warp(points_t1_gt.permute(0, 2, 1).reshape(l, 3, 512, 512),
                                      fwd_final).reshape(l, 3, -1).permute(0, 2, 1)
                    filter_t1 = warp(outlier_t1.reshape(l, 1, 512, 512), fwd_final).reshape(l, -1)
                    delta_deform = new_coords - points_t0_gt
                    delta_deform_filter = [i[(j == 1) & (k == 1)] for i, j, k in zip(delta_deform, outlier_t0, filter_t1)]
                    points_filter = [i[(j == 1) & (k == 1)] for i, j, k in zip(points_t0_gt, outlier_t0, filter_t1)]

                    points_t1_filter = [i[(j == 1)] for i, j in zip(points_t1_gt, outlier_t1.reshape(l, -1))]
                    rgb_t1_filter = [i[(j == 1)] for i, j in
                                     zip(left_t1.reshape(1, 3, -1).permute(0, 2, 1), outlier_t1.reshape(l, -1))]
                    points_filter_complete = [i[(j == 1)] for i, j in zip(points_t0_gt, outlier_t0)]
                    rgb_filter_complete = [i[(j == 1)] for i, j in
                                           zip(left_t0.reshape(l, 3, -1).permute(0, 2, 1), outlier_t0)]

                    if t == 0:
                        mid_x = th.median(points_filter[0][:, 0]).detach().cpu().numpy()
                        mid_y = th.median(points_filter[0][:, 1]).detach().cpu().numpy()
                        mid_z = th.median(points_filter[0][:, 2]).detach().cpu().numpy()

                    initial_volumes_t0 = Volumes(
                        features=th.zeros(l, 3, 64, 64, 64),
                        densities=th.zeros(l, 1, 64, 64, 64),
                        volume_translation=[-mid_x, -mid_y, -mid_z],
                        voxel_size=1.0,
                    ).cuda()
                    deform_cloud_fwd = Pointclouds(points=points_filter, features=delta_deform_filter)
                    flow_volume_tri = add_pointclouds_to_volumes(
                        pointclouds=deform_cloud_fwd,
                        initial_volumes=initial_volumes_t0,
                        mode="trilinear",
                    ).features()

                    semantic_volumes_t0 = Volumes(
                        features=th.zeros(l, 3, 64, 64, 64),
                        densities=th.zeros(l, 1, 64, 64, 64),
                        volume_translation=[-mid_x, -mid_y, -mid_z],
                        voxel_size=1.0,
                    ).cuda()
                    semantic_cloud_fwd = Pointclouds(points=points_filter_complete, features=rgb_filter_complete)
                    semantic_volume_tri = add_pointclouds_to_volumes(
                        pointclouds=semantic_cloud_fwd,
                        initial_volumes=semantic_volumes_t0,
                        mode="trilinear",
                    ).features()

                    velocity = refine_net(flow_volume_tri, semantic_volume_tri, flow_volume_previous)
                    flow_volume_refine = vecint(velocity)
                    flow_volume_previous = velocity
                    if t == 0:
                        point = points_filter_complete[0]
                        rgb = left_t0.reshape(1, 3, -1).permute(0, 2, 1)[0][outlier_t0[0]==1]
                    else:
                        point = newpoint
                    dif_out = F.grid_sample(flow_volume_refine,
                                            ((point + th.tensor(
                                                [32 - mid_x, 32 - mid_y, 32 - mid_z]).float().cuda()) / 32. - 1) \
                                            .unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                            align_corners=True).squeeze().transpose(0, 1)
                    newpoint = point + dif_out
                    point_cloud_fwd = Pointclouds(points=newpoint.unsqueeze(0),
                                                  features=rgb.unsqueeze(0))
                    point_cloud_t1 = Pointclouds(points=points_t1_filter[0].unsqueeze(0),
                                                 features=rgb_t1_filter[0].unsqueeze(0))
                    images_left_fwd = renderer_left(point_cloud_fwd)[0].cpu().numpy()
                    images_left_true = renderer_left(point_cloud_t1)[0].cpu().numpy()

                    l1_loss_left.append(l1_norm(images_left_fwd, images_left_true))
                    sim_loss_left.append(-ssim_sim(images_left_fwd, images_left_true))
                    psr_left.append(PSNR(images_left_true, images_left_fwd))
                    jacobian_left.append(jacobian(flow_volume_refine))

        logging.info("- Val metrics ssim: " + str(np.mean(sim_loss_left)))
        logging.info("- Val metrics l1: " + str(np.mean(l1_loss_left)))
        logging.info("- Val metrics psnr: " + str(np.mean(psr_left)))
        logging.info("- Val metrics jacobian: " + str(np.mean(jacobian_left)))
        loss_summary = sim_loss_left

        is_best = False
        if np.mean(loss_summary) < best_loss:
            best_loss = np.mean(loss_summary)
            is_best = True
        utils.save_checkpoint({"epoch": epoch_num + 1,
                               "best_val_loss": best_loss,
                               "flow_state_dict": flow_net.state_dict(),
                               "flow_optim_dict": flow_optimizer.state_dict(),
                               "refine_state_dict": refine_net.state_dict(),
                               "refine_optim_dict": refine_optimizer.state_dict()
                               },
                              is_best=is_best,
                              checkpoint=snapshot_path
                              )
    return "Training FInished!"
