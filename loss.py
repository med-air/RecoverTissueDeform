from typing import Union
from pytorch3d.ops.knn import knn_gather, knn_points
import torch as th
import torch
import sys
sys.path.append('../') # add relative path
from pytorch3d.structures import Pointclouds
import torch.nn as nn
import torch.nn.functional as F
def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
) -> None:
    """Check the requested reductions are valid.

    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
                raise ValueError("Expected lengths to be of shape (N,)")
            if lengths.max() > X.shape[1]:
                raise ValueError("A length value was too long")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


def chamfer_distance(
        x,
        y,
        x_feature=None,
        y_feature=None,
        x_lengths=None,
        y_lengths=None,
        x_normals=None,
        y_normals=None,
        weights=None,
        weight_x = None,
        weight_y = None,
        batch_reduction: Union[str, None] = "mean",
        point_reduction: str = "mean",
        norm: int = 2,
):
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None
    return_features = x_feature is not None and y_feature is not None
    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, norm=norm, K=1)
    idx_x = x_nn[1]
    idx_y = y_nn[1]

    if y_feature is not None:
        cham_color_x = x_feature - torch.gather(y_feature, 1, idx_x.repeat(1, 1, 3))
        cham_color_y = y_feature - torch.gather(x_feature, 1, idx_y.repeat(1, 1, 3))
        cham_color_x = th.norm(cham_color_x, p=2, dim=-1) / 3
        cham_color_y = th.norm(cham_color_y, p=2, dim=-1) / 3

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    cham_x = th.exp(-10 * cham_color_x) * cham_x
    cham_y = th.exp(-10 * cham_color_y) * cham_y
    if weight_x is not None:
        cham_x = cham_x * weight_x
        cham_y = cham_y * weight_y
    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        x_lengths_clamped = x_lengths.clamp(min=1)
        y_lengths_clamped = y_lengths.clamp(min=1)
        cham_x /= x_lengths_clamped
        cham_y /= y_lengths_clamped
        if return_normals:
            cham_norm_x /= x_lengths_clamped
            cham_norm_y /= y_lengths_clamped

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else max(N, 1)
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist, 0., 0.

def smooth_loss(y_pred):
    dy = th.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
    dx = th.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])
    dz = th.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
    dy = dy * dy
    dx = dx * dx
    dz = dz * dz
    d = th.mean(dx.reshape(len(dx),-1), dim=-1) + th.mean(dy.reshape(len(dy),-1), dim=-1) + th.mean(dz.reshape(len(dz),-1), dim=-1)
    grad = d / 3.0
    return grad


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

from torchvision.transforms.functional import rgb_to_grayscale

def census_transform(image, patch_size = 3):
    intensities = rgb_to_grayscale(image) * 255
    kernel = th.eye(patch_size* patch_size).reshape(patch_size, patch_size,1, patch_size* patch_size).permute(3,2,0,1).to(image.device)
    neighbors = F.conv2d(intensities, kernel,stride=(1,1), padding="same")
    diff = neighbors - intensities
    diff_norm = diff / th.sqrt(.81 + th.square(diff))
    return diff_norm
def hamming_distance(t1, t2):
    dist = th.square(t1-t2)
    dist_norm = dist/(0.1+dist)
    dist_sum = th.sum(dist_norm, dim=1)
    return dist_sum
def abs_robust_loss(diff, eps=0.01, q=0.4):
    return th.pow((th.abs(diff) + eps), q)

def compute_cen(img1, img2, occ):
    img1 = census_transform(img1)
    img2 = census_transform(img2)
    distance = hamming_distance(img1, img2)
    loss = abs_robust_loss(distance)
    return th.sum(occ*loss) / (th.sum(occ)+1e-6)
def compute_occ(flow):
    l, _, image_height, image_width = flow.shape
    x_base = th.linspace(0, image_width - 1, image_width).repeat(l, image_height, 1) \
                .type_as(flow).reshape(l, 1, -1)
    y_base = th.linspace(0, image_height - 1, image_height).repeat(l, image_width, 1).transpose(1, 2) \
        .type_as(flow).reshape(l, 1, -1)
    point = th.cat([x_base, y_base], dim=1) + flow.reshape(l, 2, -1)
    point = point[:,[1,0],:]
    point_floor = th.floor(point)
    point_offset = point - point_floor
    coords_floor_flattened = point_floor.permute(0,2,1).reshape(-1,2)
    coords_offset_flattened = point_offset.permute(0,2,1).reshape(-1,2)
    batch_range = th.arange(0,l).reshape([l,1,1])
    idx_batch_offset = batch_range.repeat([1, image_height, image_width]) * image_height * image_width
    idx_batch_offset_flattened = idx_batch_offset.reshape(-1).to(flow.device)
    output_width = image_width
    idxs_list = []
    weights_list = []
    for di in range(2):
        for dj in range(2):
            idxs_i = coords_floor_flattened[:, 0] + di
            idxs_j = coords_floor_flattened[:, 1] + dj
            idxs = idx_batch_offset_flattened + idxs_i * output_width + idxs_j
            mask = th.where((idxs_i>=0) & (idxs_i < image_height) & (idxs_j>=0) & (idxs_j < image_width))[0].reshape(-1)
            valid_idxs = th.gather(idxs, 0, mask)
            valid_offsets = th.gather(coords_offset_flattened, 0, mask.reshape(-1,1).repeat(1,2))
            weights_i = (1. - di) - (-1)**di * valid_offsets[:, 0]
            weights_j = (1. - dj) - (-1)**dj * valid_offsets[:, 1]
            weights = weights_i * weights_j
            idxs_list.append(valid_idxs)
            weights_list.append(weights)
    idxs = th.cat(idxs_list, dim=0).long()
    weights = th.cat(weights_list, dim=0)
    occ_map = th.zeros(l*image_height*image_width).to(flow.device).scatter_add_(dim=0, index=idxs, src=weights).reshape(l, image_height, image_width)
    return occ_map

def robust_l1(x):
    return (x**2 + 0.001**2)**0.5

def image_grads(image_batch, stride=1):
    image_batch_gh = image_batch[:, :, stride:] - image_batch[:, :, :-stride]
    image_batch_gw = image_batch[:, :, :, stride:] - image_batch[:, :, :, :-stride]
    return image_batch_gh, image_batch_gw

def edge_weighting_fn(img, lamb = 50):
    return th.exp(-lamb * th.sum(th.abs(img), dim=1))

def first_order_smoothness_loss(image, flow):
    img_gx, img_gy = image_grads(image)
    weights_x = edge_weighting_fn(img_gx).unsqueeze(1)
    weights_y = edge_weighting_fn(img_gy).unsqueeze(1)

    # Compute second derivatives of the predicted smoothness.
    flow_gx, flow_gy = image_grads(flow)

    # Compute weighted smoothness
    return ((th.mean(weights_x * robust_l1(flow_gx)) +
           th.mean(weights_y * robust_l1(flow_gy))) / 2.)

def warp(src, flow):
    flow = flow[:,[1,0],:,:]
    size = flow.shape[2:]
    vectors = [th.arange(0, s) for s in size]
    grids = th.meshgrid(vectors)
    grid = th.stack(grids)
    grid = th.unsqueeze(grid, 0)
    grid = grid.type(th.FloatTensor).to(src.device)
    new_locs = grid + flow
    shape = flow.shape[2:]
    for i in range(len(shape)):
        new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
    if len(shape) == 2:
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
    elif len(shape) == 3:
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
    return F.grid_sample(src, new_locs, align_corners=True, mode="bilinear")

def laplacian(input):
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
    res = th.mean((F.conv3d(input, kernel) - input[:, :, 1:-1, 1:-1, 1:-1]) ** 2)
    return res

def Charbonnier(a,b, occ):
    return th.sum(occ * th.sqrt((a-b)**2 + 0.001**2)) / th.sum(occ)