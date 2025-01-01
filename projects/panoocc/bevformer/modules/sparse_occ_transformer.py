import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.utils import build_from_cfg
from typing import Optional

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_

from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention

from mmcv.runner import force_fp32, auto_fp16

from ipdb import set_trace


@TRANSFORMER.register_module()
class SparseOccupancyTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 cam_encoder=None,
                 temporal_encoder=None,
                 voxel_encoder = None,
                 seg_decoder = None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(SparseOccupancyTransformer, self).__init__(**kwargs)
        self.cam_encoder = build_transformer_layer_sequence(cam_encoder)
        self.temporal_encoder = build_from_cfg(temporal_encoder, ATTENTION)
        # self.decoder = build_transformer_layer_sequence(decoder)
        self.voxel_encoder = build_transformer_layer_sequence(voxel_encoder)
        self.seg_decoder = build_transformer_layer_sequence(seg_decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        # self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        # xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            bev_z,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            **kwargs):
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']])
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']])
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :]
        can_bus = self.can_bus_mlp(can_bus)[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.cam_encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_z=bev_z,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            shift=shift,
            **kwargs
        )

        return bev_embed

    def align_prev_bev(self, prev_bev, bev_h, bev_w, bev_z, **kwargs):
        if prev_bev is not None:
            pc_range = self.cam_encoder.pc_range
            ref_y, ref_x, ref_z = torch.meshgrid(
                    torch.linspace(0.5, bev_h - 0.5, bev_h, dtype=prev_bev.dtype, device=prev_bev.device),
                    torch.linspace(0.5, bev_w - 0.5, bev_w, dtype=prev_bev.dtype, device=prev_bev.device),
                    torch.linspace(0.5, bev_z - 0.5, bev_z, dtype=prev_bev.dtype, device=prev_bev.device),
                )
            ref_y = ref_y / bev_h
            ref_x = ref_x / bev_w
            ref_z = ref_z / bev_z

            GROUND_HEIGHT = -2
            grid = torch.stack(
                    (ref_x,
                    ref_y,
                    # ref_x.new_full(ref_x.shape, GROUND_HEIGHT),
                    ref_z,
                    ref_x.new_ones(ref_x.shape)), dim=-1)

            min_x, min_y, min_z, max_x, max_y, max_z = pc_range
            grid[..., 0] = grid[..., 0] * (max_x - min_x) + min_x
            grid[..., 1] = grid[..., 1] * (max_y - min_y) + min_y
            grid[..., 2] = grid[..., 2] * (max_z - min_z) + min_z
            grid = grid.reshape(-1, 4)

            bs = prev_bev.shape[0]
            len_queue = prev_bev.shape[1]
            assert bs == 1
            for i in range(bs):
                lidar_to_ego = kwargs['img_metas'][i]['lidar2ego_transformation']
                curr_ego_to_global = kwargs['img_metas'][i]['ego2global_transform_lst'][-1]

                curr_grid_in_prev_frame_lst = []
                for j in range(len_queue):
                    prev_ego_to_global = kwargs['img_metas'][i]['ego2global_transform_lst'][j]
                    prev_lidar_to_curr_lidar = np.linalg.inv(lidar_to_ego) @ np.linalg.inv(curr_ego_to_global) @ prev_ego_to_global @ lidar_to_ego
                    curr_lidar_to_prev_lidar = np.linalg.inv(prev_lidar_to_curr_lidar)
                    curr_lidar_to_prev_lidar = grid.new_tensor(curr_lidar_to_prev_lidar)

                    curr_grid_in_prev_frame = torch.matmul(curr_lidar_to_prev_lidar, grid.T).T.reshape(bev_h, bev_w, bev_z, -1)[..., :3]
                    curr_grid_in_prev_frame[..., 0] = (curr_grid_in_prev_frame[..., 0] - min_x) / (max_x - min_x)
                    curr_grid_in_prev_frame[..., 1] = (curr_grid_in_prev_frame[..., 1] - min_y) / (max_y - min_y)
                    curr_grid_in_prev_frame[..., 2] = (curr_grid_in_prev_frame[..., 2] - min_z) / (max_z - min_z)
                    curr_grid_in_prev_frame = curr_grid_in_prev_frame * 2.0 - 1.0
                    curr_grid_in_prev_frame_lst.append(curr_grid_in_prev_frame)

                curr_grid_in_prev_frame = torch.stack(curr_grid_in_prev_frame_lst, dim=0)

                prev_bev_warp_to_curr_frame = torch.nn.functional.grid_sample(
                    prev_bev[i].permute(0, 1, 4, 2, 3),  # [bs, dim, z, h, w]
                    curr_grid_in_prev_frame.permute(0, 3, 1, 2, 4),  # [bs, z, h, w, 3]
                    align_corners=False)
                prev_bev = prev_bev_warp_to_curr_frame.permute(0, 1, 3, 4, 2).unsqueeze(0) # add bs dim, [bs, dim, h, w, z]
            return prev_bev

    def bev_temporal_fuse(
        self,
        bev_embeds: torch.Tensor,
        prev_bev: Optional[torch.Tensor],
        bev_h,
        bev_w,
        bev_z,
        **kwargs
    ) -> torch.Tensor:
        # [bs, num_queue, embed_dims, bev_h, bev_w]
        prev_bev = self.align_prev_bev(prev_bev, bev_h, bev_w, bev_z, **kwargs)

        ref_2d = self.cam_encoder.get_reference_points(
            bev_h, bev_w, dim='2d', bs=bev_embeds.size(0), device=bev_embeds.device, dtype=bev_embeds.dtype)
        bev_pos = kwargs["bev_pos"].flatten(2).permute(0, 2, 1)
        bev_embeds = self.temporal_encoder(bev_embeds, prev_bev, ref_2d=ref_2d, bev_pos=bev_pos)

        return bev_embeds


    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                bev_z,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bev_feat = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            bev_z,
            grid_length=grid_length,
            bev_pos=bev_pos,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bev_embed = self.bev_temporal_fuse(bev_feat, prev_bev, bev_h, bev_w, bev_z, bev_pos=bev_pos, **kwargs)

        bev_embed_vox = bev_embed.view(1, bev_h*bev_w,bev_z, -1)

        outs = self.voxel_encoder(bev_embed_vox)

        voxel_feat = outs[-1]

        occupancy = self.seg_decoder(voxel_feat)

        occ_list = outs[:-1] + [occupancy,]

        return bev_feat, occ_list
