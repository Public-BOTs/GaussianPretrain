import torch
import copy
from torch import nn
import torch.nn.functional as F
import numpy as np

from mmcv.runner.base_module import BaseModule
from torch.autograd import Variable
from math import exp
from mmdet.models.losses.utils import weight_reduce_loss
from scipy.spatial import KDTree

def find_corresponding_points(points, random_pixels, radius=2):
    points_xy = points[:, :2]
    kdtree = KDTree(points_xy)

    random_pixels_arr = np.array(random_pixels)

    _, indices = kdtree.query(random_pixels_arr, k=None, p=1, distance_upper_bound=radius)
    indices = np.concatenate(indices)
    
    return indices

def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # pred_sigmoid = pred.sigmoid()
    pred_sigmoid = pred         # 我修改了

    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def _scatter_max(values, inverse, num_groups):
    """Reduce values by group without requiring a newer PyTorch scatter API."""
    try:
        import torch_scatter
        reduced, _ = torch_scatter.scatter_max(
            values, inverse, dim=0, dim_size=num_groups
        )
        return reduced
    except ImportError:
        # The released project depends on torch-scatter, but this fallback keeps
        # the utility usable in lightweight environments and remains differentiable.
        reduced = []
        for group_idx in range(num_groups):
            group_values = values[inverse == group_idx]
            reduced.append(group_values.max())
        return torch.stack(reduced, dim=0)


def _voxelize_points(points, pc_range, voxel_size, voxel_shape):
    """Return linear voxel ids and an in-range mask for ``points``."""
    coords = torch.floor(
        (points[..., :3] - pc_range[:3]) / voxel_size
    ).long()
    shape = voxel_shape.view(*([1] * (coords.dim() - 1)), 3)
    inside = ((coords >= 0) & (coords < shape)).all(dim=-1)
    ids = (
        coords[..., 0] * (voxel_shape[1] * voxel_shape[2])
        + coords[..., 1] * voxel_shape[2]
        + coords[..., 2]
    )
    return ids, inside


def _lookup_sorted_ids(sorted_ids, query_ids):
    """Look up sorted integer ids while safely handling missing ids."""
    positions = torch.searchsorted(sorted_ids, query_ids)
    if sorted_ids.numel() == 0:
        return positions, torch.zeros_like(positions, dtype=torch.bool)
    safe_positions = positions.clamp(max=sorted_ids.numel() - 1)
    present = (positions < sorted_ids.numel()) & (
        sorted_ids[safe_positions] == query_ids
    )
    return positions, present


def voxel_max_opacity_loss(pred_opacity,
                           anchor_points,
                           hit_points,
                           target_one_hot,
                           pc_range,
                           voxel_size,
                           voxel_shape=None,
                           loss_type="l1",
                           alpha=0.25,
                           gamma=2.0):
    """Supervise opacity after center-based voxel-wise max aggregation.

    Each Gaussian anchor is assigned to the voxel containing its center and
    the predicted opacity in a voxel is the maximum opacity of its anchors.
    For a valid LiDAR ray, anchor voxels before the return are free (target
    zero), the voxel containing the measured return is occupied (target one),
    and anchors behind the return are unobserved and excluded. Occupied
    evidence takes precedence when free and occupied rays share a voxel.
    """
    target_one_hot = target_one_hot.float()
    if target_one_hot.dim() == 3 and target_one_hot.size(-1) == 1:
        target_one_hot = target_one_hot.squeeze(-1)
    if target_one_hot.dim() != 2:
        target_one_hot = target_one_hot.reshape(
            target_one_hot.shape[0], target_one_hot.shape[-1]
        )

    num_rays, num_samples = target_one_hot.shape
    pred = pred_opacity.reshape(num_rays, num_samples)
    points = anchor_points.reshape(num_rays, num_samples, 3).to(pred)
    hit_points = hit_points.reshape(num_rays, -1)[..., :3].to(pred)
    target_one_hot = target_one_hot.to(pred)

    pc_range = points.new_tensor(pc_range)
    voxel_size = points.new_tensor(voxel_size)
    if voxel_shape is None:
        voxel_shape = torch.ceil(
            (pc_range[3:] - pc_range[:3]) / voxel_size
        ).long()
    else:
        voxel_shape = points.new_tensor(voxel_shape).long()

    anchor_ids, anchor_inside = _voxelize_points(
        points, pc_range, voxel_size, voxel_shape
    )
    anchor_ids_flat = anchor_ids.reshape(-1)
    anchor_inside_flat = anchor_inside.reshape(-1)
    if not anchor_inside_flat.any():
        return pred.sum() * 0.0

    # Aggregate every in-range anchor, including anchors behind a return. A
    # voxel is masked only when no valid ray provides evidence for it.
    pred_flat = pred.reshape(-1)[anchor_inside_flat]
    anchor_ids_flat = anchor_ids_flat[anchor_inside_flat]
    all_ids, all_inverse = torch.unique(
        anchor_ids_flat, sorted=True, return_inverse=True
    )
    pred_voxel = _scatter_max(pred_flat, all_inverse, all_ids.numel())

    has_return = target_one_hot.sum(dim=-1) > 0
    return_bin = target_one_hot.argmax(dim=-1)
    sample_bin = torch.arange(
        num_samples, device=pred.device
    ).view(1, -1)

    # Only anchors strictly before the measured return provide free-space
    # evidence. Bins after the return, and the return bin when its center is
    # not in the hit voxel, remain unobserved.
    free_mask = has_return[:, None] & (
        sample_bin < return_bin[:, None]
    )
    free_ids = anchor_ids[free_mask & anchor_inside].reshape(-1)

    hit_ids, hit_inside = _voxelize_points(
        hit_points, pc_range, voxel_size, voxel_shape
    )
    occupied_ids = hit_ids[has_return & hit_inside].reshape(-1)

    if free_ids.numel() == 0 and occupied_ids.numel() == 0:
        return pred.sum() * 0.0

    target_ids = torch.unique(
        torch.cat((free_ids, occupied_ids), dim=0), sorted=True
    )
    target_positions, target_present = _lookup_sorted_ids(
        all_ids, target_ids
    )
    if not target_present.any():
        return pred.sum() * 0.0

    target_ids = target_ids[target_present]
    target_positions = target_positions[target_present]
    pred_voxel = pred_voxel[target_positions]

    # Occupied evidence overrides free evidence in a shared voxel.
    target_voxel = pred_voxel.new_zeros(target_ids.shape)
    occupied_ids = torch.unique(occupied_ids, sorted=True)
    occupied_positions, occupied_present = _lookup_sorted_ids(
        target_ids, occupied_ids
    )
    if occupied_present.any():
        target_voxel[occupied_positions[occupied_present]] = 1.0

    if loss_type.lower() in ("focal", "focal_loss"):
        return py_sigmoid_focal_loss(
            pred_voxel,
            target_voxel,
            alpha=alpha,
            gamma=gamma,
        )
    if loss_type.lower() in ("l1", "mae"):
        return F.l1_loss(pred_voxel, target_voxel)
    raise ValueError("Unsupported voxel opacity loss: {}".format(loss_type))
