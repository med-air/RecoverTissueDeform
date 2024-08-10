import argparse
import torch
from network import VecInt
from torchvision.models.optical_flow import raft_small
import torch as th
import torch.nn.functional as F
import sys
sys.path.append('../') # add relative path
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
    FoVPerspectiveCameras,
    PerspectiveCameras,
    OrthographicCameras,
    VolumeRenderer,
    NDCGridRaysampler,
    MonteCarloRaysampler,
    GridRaysampler,
    EmissionAbsorptionRaymarcher,
    AbsorptionOnlyRaymarcher,
    NDCMultinomialRaysampler
)
from loss import warp
from torch.utils.data import DataLoader
from loss import SSIM
from network import Unet_multimodal
import numpy as np
from skimage.metrics import structural_similarity
from dataloader import Surgical_dataset_eval
from math import log10, sqrt

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

def l1_norm(img1, img2):
    mask = (img1 > 0) * (img2 > 0)
    return np.sum(mask * np.abs(img1 - img2)) / np.sum(mask) * 255
ssim = SSIM()
def ssim_sim(img1, img2):
    mask = (img1 > 0) * (img2 > 0)
    ssim = structural_similarity(img1*mask, img2*mask, channel_axis=2, full=True)[1]
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
    res = torch.linalg.det(th.cat([t1,t2,t3], dim=0).permute(2,0,1))
    return ((th.sum(mask * res>0))/th.sum(mask)).cpu().numpy()
def laplacian_smooting(deformation):
    kernel_sub = th.zeros(3, 3, 3).cuda()
    kernel_sub[1, 0, 1] = 1 / 6
    kernel_sub[1, 2, 1] = 1 / 6
    kernel_sub[0, 1, 1] = 1 / 6
    kernel_sub[2, 1, 1] = 1 / 6
    kernel_sub[1, 1, 0] = 1 / 6
    kernel_sub[1, 1, 2] = 1 / 6
    kernel = th.zeros(3, 3, 3, 3, 3).cuda()
    kernel[0, 0] = kernel_sub
    kernel[1, 1] = kernel_sub
    kernel[2, 2] = kernel_sub
    deform_lap = F.conv3d(deformation, kernel,padding=1)
    return deform_lap
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str,
                        default='test_new.pkl', help='root dir for data')
    parser.add_argument('--model_path', type=str, default="../experiments/optical_v0/best.pth.tar", help='model_dir')

    args = parser.parse_args()
    device = "cuda"

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
    renderer_right = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras_right, raster_settings=raster_settings),
        compositor=AlphaCompositor(-0.01)
    )
    flow_net = raft_small(pretrained=True).cuda()
    flow_net.load_state_dict(th.load(args.model_path)["flow_state_dict"])
    refine_net = Unet_multimodal(inshape=[64, 64, 64], infeats=3, outfeats=3).cuda()
    refine_net.load_state_dict(th.load(args.model_path)["refine_state_dict"])
    focal = 732.24990637
    baseline = 4.552
    image_width = 512
    image_height = 512
    l1_loss_left = []
    sim_loss_left = []
    psr_left = []
    jacobian_left = []
    db_val = Surgical_dataset_eval(args.test_data)
    val_loader = DataLoader(db_val, batch_size=1, shuffle=False)
    vecint = VecInt([64, 64, 64], 7).to(device)
    flow_volume_previous = th.zeros(1, 3, 64, 64, 64).cuda()
    for i_batch, sampled_batch in enumerate(val_loader):
        length = len(sampled_batch["left"])
        flow_volume_previous = th.zeros(1, 3, 64, 64, 64).cuda()
        with th.no_grad():
            for t in range(length - 1):
                left_t0 = sampled_batch["left"][t].cuda()
                disp_t0 = sampled_batch["disp"][t].cuda()
                tool_mask0 = sampled_batch["tool"][t].cuda()

                left_t1 = sampled_batch["left"][t + 1].cuda()
                disp_t1 = sampled_batch["disp"][t + 1].cuda()
                tool_mask1 = sampled_batch["tool"][t + 1].cuda()

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
                rgb_filter = [i[(j == 1) & (k == 1)] for i, j, k in
                              zip(left_t0.reshape(1, 3, -1).permute(0, 2, 1), outlier_t0, filter_t1)]

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
                    point = points_t0_gt[0][outlier_t0[0]==1]
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
                sim_loss_left.append(ssim_sim(images_left_fwd, images_left_true))
                psr_left.append(PSNR(images_left_true, images_left_fwd))
                jacobian_left.append(jacobian(flow_volume_refine))
            print(i_batch, l1_loss_left[-1], psr_left[-1], sim_loss_left[-1], jacobian_left[-1])
    print(np.mean(l1_loss_left), np.mean(psr_left), np.mean(sim_loss_left), np.mean(jacobian_left))
