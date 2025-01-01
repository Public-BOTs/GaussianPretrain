import torch
import copy
from torch import nn
import torch.nn.functional as F
import numpy as np
import pickle

from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import HEADS
from mmcv.cnn import xavier_init
from einops import rearrange
from torch.autograd import Variable
from math import exp
from mmdet.models.losses.utils import weight_reduce_loss
from scipy.spatial import KDTree

from projects.mmdet3d_plugin.models.utils import Uni3DViewTrans, sparse_utils
from .. import modules 
from .. import utils
from ..utils.gaussian_utils import *

class LiftChannel(BaseModule):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        lift_z=5,
        bev_h=None,
        bev_w=None,
    ):
        super().__init__()
        # self.lift_cond = nn.Sequential(
        #                 nn.conv2
        #                 BasicBlock(in_channels, mid_channels),
        #                 BasicBlock(mid_channels, out_channels*lift_z)
        #                 )
        self.lift_z = lift_z
        self.bev_h = bev_h
        self.bev_w = bev_w

        self.lift_cond = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels*lift_z, 1),
                nn.BatchNorm2d(mid_channels*lift_z),
                nn.ReLU(True),
                nn.Conv2d(mid_channels*lift_z, mid_channels*lift_z, 3, padding=1),
                nn.BatchNorm2d(mid_channels*lift_z),
                nn.ReLU(True),
                nn.Conv2d(mid_channels*lift_z, out_channels*lift_z, 1))

    def forward(self, bev_feats):
        if self.bev_h is not None and bev_feats.dim()==3:
            b, n, c = bev_feats.shape
            assert self.bev_h * self.bev_w == n

            bev_feats = bev_feats.reshape(b, self.bev_h, self.bev_w, c)
            bev_feats = bev_feats.permute(0, 3, 1, 2).contiguous()

        lift_feats = self.lift_cond(bev_feats)
        bs, c, h, w = lift_feats.shape
        lift_feats = lift_feats.reshape(bs, c // self.lift_z, self.lift_z, h, w)
        return lift_feats




@HEADS.register_module()
class GaussianEgoHead(BaseModule):
    def __init__(
        self,
        in_channels,
        unified_voxel_size,
        unified_voxel_shape,
        pc_range,
        render_conv_cfg,
        gs_cfg,
        loss_cfg=None,
        cam_nums=6,
        ray_sampler_cfg=None,
        all_depth=False,
        liftfeats_cfg=None,
        img_norm_cfg=dict(
                mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        **kwargs
    ):
        super().__init__()
        if kwargs.get("fp16_enabled", False):
            self.fp16_enabled = fp16_enabled
        self.all_depth = all_depth
        self.in_channels = in_channels
        self.pc_range = np.array(pc_range, dtype=np.float32)
        self.unified_voxel_shape = np.array(unified_voxel_shape, dtype=np.int32)
        self.unified_voxel_size = np.array(unified_voxel_size, dtype=np.float32)
        self.ray_sampler_cfg = ray_sampler_cfg
   
        self.gs_param_regresser = getattr(modules, gs_cfg['type'])(**gs_cfg)

        self.render_conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                render_conv_cfg["out_channels"],
                kernel_size=render_conv_cfg["kernel_size"],
                padding=render_conv_cfg["padding"],
                stride=1,
            ),
            nn.BatchNorm3d(render_conv_cfg["out_channels"]),
            nn.ReLU(inplace=True),
        )

        # self.voxel_grid = self.create_voxel_grid()
        self.create_voxel_grid()
        self.pts_mask = torch.rand_like(self.voxel_grid) > 0.95
        

        self.loss_cfg = loss_cfg
        self.cam_nums = cam_nums
        self.lift_feats = LiftChannel(**liftfeats_cfg) if liftfeats_cfg is not None else None
        self.img_norm_cfg = img_norm_cfg

    

    def create_voxel_grid(self,):
        x_range = np.linspace(self.pc_range[0], self.pc_range[3], self.unified_voxel_shape[0], endpoint=False) + self.unified_voxel_size[0] / 2
        y_range = np.linspace(self.pc_range[1], self.pc_range[4], self.unified_voxel_shape[1], endpoint=False) + self.unified_voxel_size[1] / 2
        z_range = np.linspace(self.pc_range[2], self.pc_range[5], self.unified_voxel_shape[2], endpoint=False) + self.unified_voxel_size[2] / 2

        xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        voxel_centers_coors = np.concatenate((xx[:, :, :, None], yy[:, :, :, None], zz[:, :, :, None]), axis=-1)
        voxel_centers_coors = torch.tensor(voxel_centers_coors, dtype=torch.float32)
        
        self.register_buffer('voxel_grid', voxel_centers_coors.permute(2, 0, 1, 3).contiguous().view(-1, 3))

        xx_2d, yy_2d = torch.meshgrid([ torch.arange(0, 1600, 4), torch.arange(0, 900, 4)] )
        frustum = torch.stack([xx_2d, yy_2d, torch.ones_like(yy_2d), torch.ones_like(yy_2d)], dim=-1).view(-1, 4).to(torch.float32)  # (W, H, D, 3)
        self.register_buffer('frustum', frustum)



    @force_fp32(apply_to=("preds_dict", "targets"))
    def loss(self, preds_dict, targets):

        depth_pred = preds_dict["img_depth"].permute(0, 1, 3, 4, 2)
        depth_gt = torch.stack( [ t["img_depth"] for t in targets], dim=0)

        rgb_pred = preds_dict["img_rgb"].permute(0, 1, 3, 4, 2)
        rgb_gt = torch.stack( [ t["img_rgb"] for t in targets], dim=0)
        valid_gt_mask = torch.stack( [ t["rgb_mask"] for t in targets], dim=0).squeeze(2)
        valid_depth_gt_mask = torch.stack( [ t["depth_mask"] for t in targets], dim=0).squeeze(2)[..., :1]

        bs, n, h, w, c = rgb_pred.shape

        rgb_gt = rgb_gt[:, :, :h, :w, ...]
        valid_gt_mask = valid_gt_mask[:, :, :h, :w, ...]
        depth_gt = depth_gt[:, :, :h, :w, ...]

        loss_dict = {}
        loss_weights = self.loss_cfg.weights

        if loss_weights.get("rgb_loss", 0.0) > 0:
            rgb_loss = torch.sum(
                valid_gt_mask * torch.abs(rgb_pred - rgb_gt)
            ) / torch.clamp(valid_gt_mask.sum(), min=1.0)
            loss_dict["rgb_loss"] = rgb_loss * loss_weights.rgb_loss


        if loss_weights.get("depth_loss", 0.0) > 0:
            depth_loss = torch.sum(
                valid_depth_gt_mask * torch.abs(depth_gt - depth_pred)
            ) / torch.clamp(valid_depth_gt_mask.sum(), min=1.0)
            loss_dict["depth_loss"] = depth_loss * loss_weights.depth_loss

        if loss_weights.get("opacity_loss", 0.0) > 0:
            gt_opacity = torch.stack([ t["opacity_sampled_gt"] for t in targets], dim=0)
            bs, ray_num, sample_num = gt_opacity.shape
            pred_oci = preds_dict["voxel_opacity"].reshape(bs, ray_num, sample_num)
            loss_dict["opacity_loss"] = loss_weights.opacity_loss * F.l1_loss(pred_oci,gt_opacity)

        if loss_weights.get("opacity_focal_loss", 0.0) > 0:
            gt_opacity = torch.stack([ t["opacity_sampled_gt"] for t in targets], dim=0)
            bs, ray_num, sample_num = gt_opacity.shape
            pred_oci = preds_dict["voxel_opacity"].reshape(bs, ray_num, sample_num)
            loss_dict["opacity_focal_loss"] = loss_weights.opacity_focal_loss * py_sigmoid_focal_loss(pred_oci, gt_opacity, alpha=loss_weights.get('alpha', 0.25), gamma=loss_weights.get('gamma', 2.0))
        return loss_dict

    @force_fp32(apply_to=("viewmatrix", "projmatrix", "rgb_i", 'rot_i', 'scale_i', 'opacity_i', 'offset'))
    def render(self, pts, viewmatrix, projmatrix, cam_pos, rgb_i, rot_i, scale_i, opacity_i, offset=None, height=None, width=None, fovx=None, fovy=None):
        render_img_rgb, _, render_img_depth= modules.render(
            viewmatrix, 
            projmatrix=projmatrix, 
            cam_pos=cam_pos, 
            pts_xyz=pts if offset is None else pts + offset, 
            pts_rgb=rgb_i, 
            rotations=rot_i, 
            scales=scale_i, 
            opacity=opacity_i,
            height=height, 
            width=width, 
            fovx=fovx, 
            fovy=fovy, 
            )

        return render_img_rgb, render_img_depth
 

    @auto_fp16(apply_to=("pts_feats", "uni_feats", "img_depth"))
    def forward(self, pts_feats, uni_feats, img_metas, img_depth=None, batch_rays=None):
        """
        Args:
            Currently only support single-frame, no 3D data augmentation, no 2D data augmentation
            ray_o: [(N*C*K, 3), ...]
            ray_d: [(N*C*K, 3), ...]
            img_feats: [(B, N*C, C', H, W), ...]
            img_depth: [(B*N*C, 64, H, W), ...]
        Returns:

        """
        if self.lift_feats is not None:
            uni_feats = self.lift_feats(uni_feats)

        uni_feats = self.render_conv(uni_feats)     # [1, 32, 5, 128, 128]
        batch_ret = []
        render_img_batch = {'img_rgb':[], 'img_depth':[], 'img_mask':[], 'voxel_opacity':None, 'pts_sampled':[]}
        for bs_idx in range(len(img_metas)):
            gs_param = self.gs_param_regresser(uni_feats[bs_idx], batch_rays[bs_idx])

            pts = batch_rays[bs_idx]['pts_sampled'].view(-1, 3)
            rot_i = gs_param['rot']
            scale_i = gs_param['scale']
            opacity_i = gs_param['opacity']
            rgb_i = gs_param['rgb']
            offset_i = None
            img_rgb = []
            img_depth = []
            for cam_idx in range(self.cam_nums):
                fovx = img_metas[bs_idx]['cam_params'][cam_idx]['fovx']
                fovy = img_metas[bs_idx]['cam_params'][cam_idx]['fovy']
                viewmatrix = img_metas[bs_idx]['cam_params'][cam_idx]['viewmatrix'].to(uni_feats.device)
                projmatrix = img_metas[bs_idx]['cam_params'][cam_idx]['projmatrix'].to(uni_feats.device)
                cam_pos = img_metas[bs_idx]['cam_params'][cam_idx]['cam_pos'].to(uni_feats.device)

                render_img_rgb, render_img_depth = self.render(pts, viewmatrix, projmatrix, cam_pos, rgb_i, rot_i, scale_i,opacity_i, offset_i, img_metas[bs_idx]['cam_params'][cam_idx]['height'], img_metas[bs_idx]['cam_params'][cam_idx]['width'], fovx, fovy)

                img_rgb.append(render_img_rgb)
                img_depth.append(render_img_depth)

            render_img_batch['img_rgb'].append(torch.stack(img_rgb, dim=0))
            render_img_batch['img_depth'].append(torch.stack(img_depth, dim=0))
            render_img_batch['pts_sampled'].append(pts if offset_i is None else pts + offset_i)

        render_img_batch['img_rgb'] = torch.clamp_max(torch.stack(render_img_batch['img_rgb'], dim=0), 1.0)
        render_img_batch['img_depth'] = torch.stack(render_img_batch['img_depth'], dim=0)
        render_img_batch['voxel_opacity'] = gs_param['opacity']
        render_img_batch['pts_sampled'] = torch.stack(render_img_batch['pts_sampled'], dim=0)

        # self.vis_pred_img(render_img_batch['img_rgb'])
        # self.vis_grid(img_metas, batch_rays)
        # self.vis_points(img_metas, pts_feats, batch_rays)
        return render_img_batch
    
    def vis_pred_img(self, img_rgb, ):
        import cv2
        img_rgb = img_rgb.detach().cpu()
        img_rgb = img_rgb.permute(0, 1, 3, 4, 2).numpy()
        bs, cam_num = img_rgb.shape[:2]
        for b in range(bs):
            for cam in range(cam_num):
                img = img_rgb[b, cam]
                img = img * 255
                img = img.astype(np.int)
                cv2.imwrite(f'./pred_img_715_bs.{b}_cam.{cam}.png', img)
    

    def vis_points(self, img_metas, points, batch_rays=None):
        import cv2
        eps = 1e-5
        lidar2img, lidar2cam = [], []
        ego2lidar = img_metas[0]['ego2lidar']
        lidar2ego = np.linalg.inv(ego2lidar)
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
            lidar2cam.append(img_meta["lidar2cam"])
        lidar2img = np.asarray(lidar2img)
        lidar2cam = np.asarray(lidar2cam)

        ego2img = np.dot(lidar2img, ego2lidar)
        ego2cam = np.dot(lidar2cam, ego2lidar)

        lidar2cam = ego2cam
        lidar2img = ego2img
        bs = len(img_metas)
        for b in range(bs):
            voxel_grid = points[b][:, :3]
            voxel_grid = torch.cat((voxel_grid.cpu(), torch.ones((voxel_grid.shape[0],1))),-1).to(torch.float32)

            i_lidar2ego = voxel_grid.new_tensor(lidar2ego)
            voxel_grid = torch.matmul(
                i_lidar2ego.unsqueeze(0), voxel_grid.view(-1, 4, 1)
            ).squeeze(-1)
            for cam in range(len(img_metas[b]['lidar2img'])):
                lidar2img_ = torch.tensor(lidar2img[b][cam], dtype=torch.float32)
                voxel_grid_cam = voxel_grid @ lidar2img_.T
                voxel_grid_cam[:, :2] /= torch.maximum(voxel_grid_cam[..., 2:3], torch.ones_like(voxel_grid_cam[..., 2:3]) * eps)
                voxel_grid_cam = voxel_grid_cam.cpu().numpy().astype(np.int)
                # img = np.zeros(img_metas[b]['ori_shape'][cam])
                img = (batch_rays[b]['img_rgb'][cam].detach().cpu().numpy() * 255).astype(np.uint8)

                for point in voxel_grid_cam:
                    if point[2] > 0:
                        cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
                cv2.imwrite(f'points_bs.{b}_cam.{cam}.png', img)


    def vis_grid(self, img_metas, batch_rays):
        import cv2
        eps = 1e-5
        bs = len(img_metas)
        for b in range(bs):
            voxel_grid = batch_rays[b]['pts_sampled'].view(-1, 3)
            voxel_grid = torch.cat((voxel_grid.cpu(), torch.ones((voxel_grid.shape[0],1))),-1).to(torch.float32)
            for cam in range(len(img_metas[b]['lidar2img'])):
                lidar2img = torch.tensor(img_metas[b]['lidar2img'][cam], dtype=torch.float32)
                voxel_grid_cam = voxel_grid @ lidar2img.T
                voxel_grid_cam[:, :2] /= torch.maximum(voxel_grid_cam[..., 2:3], torch.ones_like(voxel_grid_cam[..., 2:3]) * eps)
                voxel_grid_cam = voxel_grid_cam.cpu().numpy().astype(np.int)
                img = np.zeros(img_metas[b]['ori_shape'][cam])
                for point in voxel_grid_cam:
                    if point[2] > 0:
                        cv2.circle(img, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)
                cv2.imwrite(f'./7.9.voxel_grid_bs.{b}_cam.{cam}.png', img)
    
    def points2depthmap(self, points, img_depth):
        height, width = img_depth.shape[0], img_depth.shape[1]
        depth = points[:, 2]
        # coor = torch.round(points[:, :2])
        coor = points[:, :2].long().float()

        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.0).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept], depth[kept]
        coor = coor.long()
        img_depth[coor[:, 1], coor[:, 0], 0] = depth
        return img_depth, coor
        

    def sample_rays(self, pts, imgs, img_metas):
        lidar2img, lidar2cam = [], []
        ego2lidar = img_metas[0]['ego2lidar']
        lidar2ego = np.linalg.inv(ego2lidar)
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
            lidar2cam.append(img_meta["lidar2cam"])
        lidar2img = np.asarray(lidar2img)
        lidar2cam = np.asarray(lidar2cam)

        ego2img = np.dot(lidar2img, ego2lidar)
        ego2cam = np.dot(lidar2cam, ego2lidar)

        lidar2cam = ego2cam
        lidar2img = ego2img

        new_pts = []
        for bs_idx, i_pts in enumerate(pts):
            i_lidar2ego = i_pts.new_tensor(lidar2ego)
            i_pts_ = torch.cat([i_pts[:, :3], torch.ones_like(i_pts[:, :1])], dim=1)

            i_pts_ = torch.matmul(
                i_lidar2ego.unsqueeze(0), i_pts_.view(-1, 4, 1)
            ).squeeze(-1)

            i_pts = torch.cat([i_pts_[:, :3], i_pts[:, 3:]], dim=1)
            dis = i_pts[:, :2].norm(dim=-1)
            dis_mask = (dis > self.ray_sampler_cfg.close_radius) & (
                dis < self.ray_sampler_cfg.get("far_radius", 100.0)
            )
            new_pts.append(i_pts[dis_mask]) 
        pts = new_pts
        if (
            sparse_utils._cur_active_voxel is not None
            and self.ray_sampler_cfg.only_point_mask
        ):
            pc_range = torch.from_numpy(self.pc_range).to(pts[0])
            mask_voxel_size = (
                torch.from_numpy(self.unified_voxel_size).to(pts[0])
                / sparse_utils._cur_voxel_scale
            )
            mask_voxel_shape = (
                torch.from_numpy(self.unified_voxel_shape).to(pts[0].device)
                * sparse_utils._cur_voxel_scale
            )
            nonactive_voxel_mask = torch.zeros(
                (len(pts), *mask_voxel_shape.flip(dims=[0])),
                dtype=torch.bool,
                device=pts[0].device,
            )
            nonactive_voxel_mask[
                sparse_utils._cur_voxel_coords[~sparse_utils._cur_active_voxel]
                .long()
                .unbind(dim=1)
            ] = True
            new_pts = []
            for bs_idx in range(len(pts)):
                p_pts = pts[bs_idx]
                p_coords = (p_pts[:, :3] - pc_range[:3]) / mask_voxel_size
                kept = torch.all(
                    (p_coords >= torch.zeros_like(mask_voxel_shape))
                    & (p_coords < mask_voxel_shape),
                    dim=-1,
                )
                p_coords = F.pad(
                    p_coords[:, [2, 1, 0]].long(), (1, 0), mode="constant", value=bs_idx
                )
                p_coords, p_pts = p_coords[kept], p_pts[kept]
                p_nonactive_pts_mask = nonactive_voxel_mask[p_coords.unbind(dim=1)]
                new_pts.append(p_pts[p_nonactive_pts_mask])
            pts = new_pts      
        
        # sparse_utils._cur_active: [6, 1, 29, 50]
        # self.ray_sampler_cfg.only_img_mask: False
        if sparse_utils._cur_active is not None and self.ray_sampler_cfg.only_img_mask:
            active_mask = sparse_utils._get_active_ex_or_ii(imgs.shape[-2])
            assert (
                active_mask.shape[-2] == imgs.shape[-2]
                and active_mask.shape[-1] == imgs.shape[-1]
            )
            active_mask = active_mask.view(
                imgs.shape[0], -1, imgs.shape[-2], imgs.shape[-1]
            )

        batch_ret = []
        for bs_idx in range(len(pts)):
            i_imgs = imgs[bs_idx]
            i_pts = pts[bs_idx]
            i_lidar2img = i_pts.new_tensor(lidar2img[bs_idx])

            i_ego2img = i_pts.new_tensor(ego2img[bs_idx])
            i_img2ego = torch.inverse(
                i_ego2img
            )

            i_img2lidar = torch.inverse(
                i_lidar2img
            )  # TODO: Are img2lidar and img2cam consistent after image data augmentation?
            i_cam2lidar = torch.inverse(
                i_pts.new_tensor(lidar2cam[bs_idx])
            )

            i_pts = torch.cat([i_pts[..., :3], torch.ones_like(i_pts[..., :1])], -1)
            i_pts_cam = torch.matmul(
                i_lidar2img.unsqueeze(1), i_pts.view(1, -1, 4, 1)
            ).squeeze(-1)

            eps = 1e-5
            i_pts_mask = i_pts_cam[..., 2] > eps
            i_pts_cam[..., :2] = i_pts_cam[..., :2] / torch.maximum(
                i_pts_cam[..., 2:3], torch.ones_like(i_pts_cam[..., 2:3]) * eps
            )

            # (N*C, 3) [(H, W, 3), ...]
            pad_before_shape = torch.tensor(
                img_metas[bs_idx]["ori_shape"], device=i_pts_cam.device
            )
            Hs, Ws = pad_before_shape[:, 0:1], pad_before_shape[:, 1:2]

            # (N*C, M)
            i_pts_mask = (
                i_pts_mask
                & (i_pts_cam[..., 0] > 0)
                & (i_pts_cam[..., 0] < Ws - 1)
                & (i_pts_cam[..., 1] > 0)
                & (i_pts_cam[..., 1] < Hs - 1)
            )

            i_imgs = i_imgs.permute(0, 2, 3, 1)
            i_imgs = i_imgs * i_imgs.new_tensor(
                self.img_norm_cfg['std']
            ) + i_imgs.new_tensor(self.img_norm_cfg['mean'])
            if not self.img_norm_cfg["to_rgb"]:
                i_imgs[..., [0, 1, 2]] = i_imgs[..., [2, 1, 0]]  # bgr->rgb
            i_imgs = i_imgs[:, :img_metas[0]['ori_shape'][0][0], :img_metas[0]['ori_shape'][0][1]]
     
            i_sampled_rgb_gt, i_sampled_rgb_mask, i_sampled_pts, i_sampled_depth_gt, i_sampled_pts_gt, i_sampled_opacity_gt, i_sampled_depth_mask  = ([], [], [], [], [], [], [])
            for c_idx in range(len(i_pts_mask)):
                j_sampled_all_pts, j_sampled_all_pts_cam, j_sampled_all_depth_mask,  j_sampled_all_lidar= (
                    [],
                    [],
                    [],
                    [],
                )

                """ sample points """
                j_sampled_pts_idx = i_pts_mask[c_idx].nonzero(as_tuple=True)[0]
                j_sampled_pts_cam = i_pts_cam[c_idx][j_sampled_pts_idx]
                j_sampled_pts_cam_all = j_sampled_pts_cam
                if self.ray_sampler_cfg.only_img_mask:
                    j_sampled_pts_mask = ~active_mask[
                        bs_idx,
                        c_idx,
                        j_sampled_pts_cam[:, 1].long(),
                        j_sampled_pts_cam[:, 0].long(),
                    ]
                    j_sampled_pts_idx = j_sampled_pts_mask.nonzero(as_tuple=True)[0]
                else:
                    j_sampled_pts_idx = torch.arange(
                        len(j_sampled_pts_cam),
                        dtype=torch.long,
                        device=j_sampled_pts_cam.device,
                    )

                point_nsample = min(      
                    len(j_sampled_pts_idx),
                    int(len(j_sampled_pts_idx) * self.ray_sampler_cfg.point_ratio)
                    if self.ray_sampler_cfg.point_nsample == -1
                    else self.ray_sampler_cfg.point_nsample,
                )

                if point_nsample > 0:
                    replace_sample = (
                        True
                        if point_nsample > len(j_sampled_pts_idx)
                        else self.ray_sampler_cfg.replace_sample
                    )
                    j_sampled_pts_idx = j_sampled_pts_idx[
                        torch.from_numpy(
                            np.random.choice(
                                len(j_sampled_pts_idx),
                                point_nsample,
                                replace=replace_sample,
                            )
                        )
                        .long()
                        .to(j_sampled_pts_idx.device)
                    ]
                    j_sampled_pts_cam = j_sampled_pts_cam[j_sampled_pts_idx]

                    depth_bin = torch.linspace(1, 60, self.ray_sampler_cfg.get('anchor_gaussian_interval', 100)).to(j_sampled_pts_cam.device)
                    pixel_points_repeated=j_sampled_pts_cam[...,:2].repeat_interleave(self.ray_sampler_cfg.get('anchor_gaussian_interval', 100), dim=0).long().float()    # 否则会存在错位的情况

                    depths_repeated = depth_bin.repeat(j_sampled_pts_cam.shape[0])
                    pixel_3d = torch.cat((pixel_points_repeated, depths_repeated.unsqueeze(1)), dim=1)

                    lidar_3d_sampled = torch.matmul(
                                        i_img2lidar[c_idx : c_idx + 1],
                                        torch.cat([
                                            pixel_3d[..., :2] * pixel_3d[..., 2:3],
                                            pixel_3d[..., 2:3],
                                            torch.ones_like(pixel_3d[..., 2:3]),
                                        ], dim=-1).unsqueeze(-1)
                                         ).squeeze(-1)[..., :3].view(-1, self.ray_sampler_cfg.get('anchor_gaussian_interval', 100), 3)


                    i_sampled_pts.append(lidar_3d_sampled)
                    
                    j_sampled_pts = torch.matmul(
                        i_img2lidar[c_idx : c_idx + 1],
                        torch.cat(
                            [
                                j_sampled_pts_cam[..., :2]
                                * j_sampled_pts_cam[..., 2:3],
                                j_sampled_pts_cam[..., 2:],
                            ],
                            dim=-1,
                        ).unsqueeze(-1),
                    ).squeeze(-1)[..., :3]

                    i_sampled_pts_gt.append(j_sampled_pts)
                    j_sampled_all_pts.append(j_sampled_pts)
                    j_sampled_all_pts_cam.append(j_sampled_pts_cam[..., :3])

                    sampled_opacity_gt = ((j_sampled_pts_cam[:,2] - 1)/(60.0 / self.ray_sampler_cfg.get('anchor_gaussian_interval', 100))).long()
                    sampled_opacity_gt = F.one_hot(sampled_opacity_gt, num_classes=self.ray_sampler_cfg.get('anchor_gaussian_interval', 100))
                    i_sampled_opacity_gt.append(sampled_opacity_gt)

                    img_depth = torch.zeros_like(i_imgs[c_idx:c_idx+1][..., 0:1])
                    
                    if self.all_depth:
                        img_depth, _ = self.points2depthmap(j_sampled_pts_cam_all, img_depth[0])
                    else:
                        img_depth, _ = self.points2depthmap(j_sampled_pts_cam, img_depth[0])

                    i_sampled_depth_gt.append(img_depth)
                    depth_mask = torch.zeros_like(i_imgs[c_idx:c_idx+1])
        
                    depth_mask[
                            0,
                            j_sampled_pts_cam[:, 1].long(),
                            j_sampled_pts_cam[:, 0].long(), 
                        ] = 1
                    i_sampled_depth_mask.append(depth_mask)

                    img_mask = torch.zeros_like(i_imgs[c_idx:c_idx+1])
        
                    img_mask[
                            0,
                            j_sampled_pts_cam[:, 1].long(),
                            j_sampled_pts_cam[:, 0].long(), 
                        ] = 1


                    i_sampled_rgb_mask.append(
                        img_mask
                    )
                    i_sampled_rgb_gt.append(i_imgs[c_idx] / 255.0)


                else:
                    depth_mask = torch.zeros_like(i_imgs[c_idx:c_idx+1])
                    i_sampled_depth_mask.append(depth_mask)
                    img_depth = torch.zeros_like(i_imgs[c_idx:c_idx+1][..., 0:1])[0]
                    i_sampled_depth_gt.append(img_depth)
                    i_sampled_rgb_mask.append(torch.zeros_like(i_imgs[c_idx:c_idx+1]))
                    i_sampled_rgb_gt.append(i_imgs[c_idx] / 255.0)

            batch_ret.append(
                {
                    'rgb_mask':torch.stack(i_sampled_rgb_mask, dim=0).to(imgs.device),
                    'pts_sampled':torch.cat(i_sampled_pts, dim=0).to(imgs.device),
                    'img_rgb':torch.stack(i_sampled_rgb_gt, dim=0).to(imgs.device),
                    'img_depth':torch.stack(i_sampled_depth_gt, dim=0).to(imgs.device),
                    'pts_sampled_gt':torch.cat(i_sampled_pts_gt, dim=0).to(imgs.device),
                    'opacity_sampled_gt':torch.cat(i_sampled_opacity_gt, dim=0).to(imgs.device),
                    'depth_mask':torch.stack(i_sampled_depth_mask, dim=0).to(imgs.device),
                }
            )
        return batch_ret


    def sample_rays_test(self, pts, imgs, img_metas):
        lidar2img, lidar2cam = [], []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
            lidar2cam.append(img_meta["lidar2cam"])
        lidar2img = np.asarray(lidar2img)
        lidar2cam = np.asarray(lidar2cam)

        new_pts = []
        for bs_idx, i_pts in enumerate(pts):
            dis = i_pts[:, :2].norm(dim=-1)
            dis_mask = (dis > self.ray_sampler_cfg.close_radius) & (
                dis < self.ray_sampler_cfg.get("far_radius", 100.0)
            )
            new_pts.append(i_pts[dis_mask])
        pts = new_pts
        # if (
        #     sparse_utils._cur_active_voxel is not None
        #     and self.ray_sampler_cfg.only_point_mask
        # ):
        #     pc_range = torch.from_numpy(self.pc_range).to(pts[0])
        #     mask_voxel_size = (
        #         torch.from_numpy(self.unified_voxel_size).to(pts[0])
        #         / sparse_utils._cur_voxel_scale
        #     )
        #     mask_voxel_shape = (
        #         torch.from_numpy(self.unified_voxel_shape).to(pts[0].device)
        #         * sparse_utils._cur_voxel_scale
        #     )
        #     nonactive_voxel_mask = torch.zeros(
        #         (len(pts), *mask_voxel_shape.flip(dims=[0])),
        #         dtype=torch.bool,
        #         device=pts[0].device,
        #     )
        #     nonactive_voxel_mask[
        #         sparse_utils._cur_voxel_coords[~sparse_utils._cur_active_voxel]
        #         .long()
        #         .unbind(dim=1)
        #     ] = True
        #     new_pts = []
        #     for bs_idx in range(len(pts)):
        #         p_pts = pts[bs_idx]
        #         p_coords = (p_pts[:, :3] - pc_range[:3]) / mask_voxel_size
        #         kept = torch.all(
        #             (p_coords >= torch.zeros_like(mask_voxel_shape))
        #             & (p_coords < mask_voxel_shape),
        #             dim=-1,
        #         )
        #         p_coords = F.pad(
        #             p_coords[:, [2, 1, 0]].long(), (1, 0), mode="constant", value=bs_idx
        #         )
        #         p_coords, p_pts = p_coords[kept], p_pts[kept]
        #         p_nonactive_pts_mask = nonactive_voxel_mask[p_coords.unbind(dim=1)]
        #         new_pts.append(p_pts[p_nonactive_pts_mask])
        #     pts = new_pts 
        # # sparse_utils._cur_active: [6, 1, 29, 50]
        # # self.ray_sampler_cfg.only_img_mask: False
        # if sparse_utils._cur_active is not None and self.ray_sampler_cfg.only_img_mask:
        #     active_mask = sparse_utils._get_active_ex_or_ii(imgs.shape[-2])
        #     assert (
        #         active_mask.shape[-2] == imgs.shape[-2]
        #         and active_mask.shape[-1] == imgs.shape[-1]
        #     )
        #     active_mask = active_mask.view(
        #         imgs.shape[0], -1, imgs.shape[-2], imgs.shape[-1]
        #     )

        batch_ret = []
        for bs_idx in range(len(pts)):
            i_imgs = imgs[bs_idx]
            i_pts = pts[bs_idx]
            i_lidar2img = i_pts.new_tensor(lidar2img[bs_idx]).flatten(0, 1)
            i_img2lidar = torch.inverse(
                i_lidar2img
            )  # TODO: Are img2lidar and img2cam consistent after image data augmentation?
            i_cam2lidar = torch.inverse(
                i_pts.new_tensor(lidar2cam[bs_idx]).flatten(0, 1)
            )
            i_pts = torch.cat([i_pts[..., :3], torch.ones_like(i_pts[..., :1])], -1)
            i_pts_cam = torch.matmul(
                i_lidar2img.unsqueeze(1), i_pts.view(1, -1, 4, 1)
            ).squeeze(-1)

            eps = 1e-5
            i_pts_mask = i_pts_cam[..., 2] > eps
            i_pts_cam[..., :2] = i_pts_cam[..., :2] / torch.maximum(
                i_pts_cam[..., 2:3], torch.ones_like(i_pts_cam[..., 2:3]) * eps
            )

            # (N*C, 3) [(H, W, 3), ...]
            pad_before_shape = torch.tensor(
                img_metas[bs_idx]["pad_before_shape"], device=i_pts_cam.device
            )
            Hs, Ws = pad_before_shape[:, 0:1], pad_before_shape[:, 1:2]

            # (N*C, M)
            i_pts_mask = (
                i_pts_mask
                & (i_pts_cam[..., 0] > 0)
                & (i_pts_cam[..., 0] < Ws - 1)
                & (i_pts_cam[..., 1] > 0)
                & (i_pts_cam[..., 1] < Hs - 1)
            )

            i_imgs = i_imgs.permute(0, 2, 3, 1)
            i_imgs = i_imgs * i_imgs.new_tensor(
                img_metas[0]["img_norm_cfg"]["std"]
            ) + i_imgs.new_tensor(img_metas[0]["img_norm_cfg"]["mean"])
            if not img_metas[0]["img_norm_cfg"]["to_rgb"]:
                i_imgs[..., [0, 1, 2]] = i_imgs[..., [2, 1, 0]]  # bgr->rgb
            i_imgs = i_imgs[:, :900, :1600]
            i_sampled_ray_o, i_sampled_ray_d, i_sampled_rgb, i_sampled_depth = (
                [],
                [],
                [],
                [],
            )
            i_sampled_rgb_gt, i_sampled_rgb_mask, i_sampled_pts, i_sampled_dpeth = ([], [], [], [])
            for c_idx in range(len(i_pts_mask)):
                j_sampled_all_pts, j_sampled_all_pts_cam, j_sampled_all_depth_mask = (
                    [],
                    [],
                    [],
                )

                """ sample points """
                j_sampled_pts_idx = i_pts_mask[c_idx].nonzero(as_tuple=True)[0]
                j_sampled_pts_cam = i_pts_cam[c_idx][j_sampled_pts_idx]

                if self.ray_sampler_cfg.only_img_mask:
                    j_sampled_pts_mask = ~active_mask[
                        bs_idx,
                        c_idx,
                        j_sampled_pts_cam[:, 1].long(),
                        j_sampled_pts_cam[:, 0].long(),
                    ]
                    j_sampled_pts_idx = j_sampled_pts_mask.nonzero(as_tuple=True)[0]
                else:
                    j_sampled_pts_idx = torch.arange(
                        len(j_sampled_pts_cam),
                        dtype=torch.long,
                        device=j_sampled_pts_cam.device,
                    )

                point_nsample = min(
                    len(j_sampled_pts_idx),
                    int(len(j_sampled_pts_idx) * self.ray_sampler_cfg.point_ratio)
                    if self.ray_sampler_cfg.point_nsample == -1
                    else self.ray_sampler_cfg.point_nsample,
                )

                if point_nsample > 0:
                    replace_sample = (
                        True
                        if point_nsample > len(j_sampled_pts_idx)
                        else self.ray_sampler_cfg.replace_sample
                    )
                    j_sampled_pts_idx = j_sampled_pts_idx[
                        torch.from_numpy(
                            np.random.choice(
                                len(j_sampled_pts_idx),
                                point_nsample,
                                replace=replace_sample,
                            )
                        )
                        .long()
                        .to(j_sampled_pts_idx.device)
                    ]
                    j_sampled_pts_cam = j_sampled_pts_cam[j_sampled_pts_idx]

                    j_sampled_pts_cam = self.frustum            # inference 

                    depth_bin = torch.linspace(1, 60, 100).to(j_sampled_pts_cam.device)
                    pixel_points_repeated=j_sampled_pts_cam[...,:2].repeat_interleave(100, dim=0).long().float()
                    
                    depths_repeated = depth_bin.repeat(j_sampled_pts_cam.shape[0])
                    pixel_3d = torch.cat((pixel_points_repeated, depths_repeated.unsqueeze(1)), dim=1)

                    lidar_3d_sampled = torch.matmul(
                                        i_img2lidar[c_idx : c_idx + 1],
                                        torch.cat([
                                            pixel_3d[..., :2] * pixel_3d[..., 2:3],
                                            pixel_3d[..., 2:3],
                                            torch.ones_like(pixel_3d[..., 2:3]),
                                        ], dim=-1).unsqueeze(-1)
                                         ).squeeze(-1)[..., :3].view(-1, 100, 3)

                    i_sampled_pts.append(lidar_3d_sampled)
                    j_sampled_pts = torch.matmul(
                        i_img2lidar[c_idx : c_idx + 1],
                        torch.cat(
                            [
                                j_sampled_pts_cam[..., :2]
                                * j_sampled_pts_cam[..., 2:3],
                                j_sampled_pts_cam[..., 2:],
                            ],
                            dim=-1,
                        ).unsqueeze(-1),
                    ).squeeze(-1)[..., :3]
                    j_sampled_all_pts.append(j_sampled_pts)
                    j_sampled_all_pts_cam.append(j_sampled_pts_cam[..., :2])
                    j_sampled_all_depth_mask.append(
                        torch.ones_like(j_sampled_pts_cam[:, 0])
                    )
                """ sample pixels """
                if self.ray_sampler_cfg.merged_nsample - point_nsample > 0:
                    pixel_interval = self.ray_sampler_cfg.pixel_interval
                    sky_region = self.ray_sampler_cfg.sky_region
                    tx = torch.arange(
                        0,
                        Ws[c_idx, 0],
                        pixel_interval,
                        device=i_imgs.device,
                        dtype=i_imgs.dtype,
                    )
                    ty = torch.arange(
                        int(sky_region * Hs[c_idx, 0]),
                        Hs[c_idx, 0],
                        pixel_interval,
                        device=i_imgs.device,
                        dtype=i_imgs.dtype,
                    )
                    pixels_y, pixels_x = torch.meshgrid(ty, tx)
                    i_pixels_cam = torch.stack([pixels_x, pixels_y], dim=-1)

                    j_sampled_pixels_cam = i_pixels_cam.flatten(0, 1)
                    if self.ray_sampler_cfg.only_img_mask:
                        j_sampled_pixels_mask = ~active_mask[ 
                            bs_idx,
                            c_idx,
                            j_sampled_pixels_cam[:, 1].long(),
                            j_sampled_pixels_cam[:, 0].long(),
                        ]  # (Q,)
                        j_sampled_pixels_idx = j_sampled_pixels_mask.nonzero(
                            as_tuple=True
                        )[0]
                    else:
                        j_sampled_pixels_idx = torch.arange(
                            len(j_sampled_pixels_cam),
                            dtype=torch.long,
                            device=j_sampled_pixels_cam.device,
                        )

                    pixel_nsample = min(
                        len(j_sampled_pixels_idx),
                        self.ray_sampler_cfg.merged_nsample - point_nsample,
                    )
                    j_sampled_pixels_idx = j_sampled_pixels_idx[
                        torch.from_numpy(
                            np.random.choice(
                                len(j_sampled_pixels_idx),
                                pixel_nsample,
                                replace=self.ray_sampler_cfg.replace_sample,
                            )
                        )
                        .long()
                        .to(j_sampled_pixels_idx.device)
                    ]
                    j_sampled_pixels_cam = j_sampled_pixels_cam[j_sampled_pixels_idx]
                    j_sampled_pixels = torch.matmul(
                        i_img2lidar[c_idx : c_idx + 1],
                        torch.cat(
                            [
                                j_sampled_pixels_cam,
                                torch.ones_like(j_sampled_pixels_cam),
                            ],
                            dim=-1,
                        ).unsqueeze(-1),
                    ).squeeze(-1)[..., :3]
                    j_sampled_all_pts.append(j_sampled_pixels)
                    j_sampled_all_pts_cam.append(j_sampled_pixels_cam)
                    j_sampled_all_depth_mask.append(
                        torch.zeros_like(j_sampled_pixels_cam[:, 0])
                    )

                if len(j_sampled_all_pts) > 0:
                    """merge"""
                    j_sampled_all_pts = torch.cat(j_sampled_all_pts, dim=0)
                    j_sampled_all_pts_cam = torch.cat(j_sampled_all_pts_cam, dim=0)
                    j_sampled_all_depth_mask = torch.cat(
                        j_sampled_all_depth_mask, dim=0
                    )

                    unscaled_ray_o = i_cam2lidar[c_idx : c_idx + 1, :3, 3].repeat(
                        j_sampled_all_pts.shape[0], 1
                    )
                    i_sampled_ray_o.append(
                        unscaled_ray_o 
                    )
                    i_sampled_ray_d.append(
                        F.normalize(j_sampled_all_pts - unscaled_ray_o, dim=-1)
                    )
                    sampled_depth = (
                        torch.norm(
                            j_sampled_all_pts - unscaled_ray_o, dim=-1, keepdim=True
                        )
                        
                    )
                    sampled_depth[j_sampled_all_depth_mask == 0] = -1.0
                    i_sampled_depth.append(sampled_depth)
                    i_sampled_rgb.append(
                        i_imgs[
                            c_idx,
                            j_sampled_all_pts_cam[:, 1].long(),
                            j_sampled_all_pts_cam[:, 0].long(),
                        ]
                        / 255.0
                    )
                    img_mask = torch.zeros_like(i_imgs[c_idx:c_idx+1])
                    img_mask[
                            0,
                            j_sampled_all_pts_cam[:, 1].long(),
                            j_sampled_all_pts_cam[:, 0].long(), 
                        ] = 1

                    img_depth = torch.zeros_like(i_imgs[c_idx:c_idx+1])
                    img_depth, _ = self.points2depthmap(j_sampled_pts_cam, img_depth[0])

                    i_sampled_rgb_mask.append(
                        img_mask
                    )
                    i_sampled_rgb_gt.append(i_imgs[c_idx] / 255.0)
                    i_sampled_dpeth.append(img_depth)

            batch_ret.append(
                {
                    "ray_o": torch.cat(i_sampled_ray_o, dim=0),     # [3072, 3]
                    "ray_d": torch.cat(i_sampled_ray_d, dim=0),     # [3072, 3]
                    "rgb": torch.cat(i_sampled_rgb, dim=0),         # [3072, 3]
                    "depth": torch.cat(i_sampled_depth, dim=0),         # [3072, 1]    
                    "scaled_points": pts[bs_idx][:, :3] ,        # [25682, 3]
                    'rgb_mask':torch.stack(i_sampled_rgb_mask, dim=0),
                    'pts_sampled':torch.cat(i_sampled_pts, dim=0),
                    'img_rgb':torch.stack(i_sampled_rgb_gt, dim=0),
                    'img_depth':torch.stack(i_sampled_dpeth, dim=0),
                    
                }
            
            )
        return batch_ret

