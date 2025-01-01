import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import mmcv
import cv2 as cv
from mmdet3d.ops import scatter_v2
import torch_scatter
from mmdet.models.builder import build_loss
from spconv.pytorch import SparseConvTensor 

@HEADS.register_module()
class SparseOccupancyHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 bev_z=5,
                 num_occ_classes=17,
                 voxel_lidar = [0.05, 0.05, 0.05],
                 voxel_det = [2.048,2.048,1],
                 loss_occupancy=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=5.0),
                loss_occupancy_layer0=None,
                loss_occupancy_aux=None,
                loss_occupancy_det=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=5.0),
                bg_weight=0.02,
                early_supervision_cfg=dict(),
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.voxel_lidar = voxel_lidar
        self.voxel_det = voxel_det
        self.fp16_enabled = False
        self.bg_weight = bg_weight
        self.num_occ_classes = num_occ_classes

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.span_x, self.span_y, self.span_z = self.real_w, self.real_h, self.pc_range[5] - self.pc_range[2]
        self.num_cls_fcs = num_cls_fcs - 1
        super(SparseOccupancyHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.lidar_seg_loss = build_loss(loss_occupancy)
        self.early_supervision_cfg = early_supervision_cfg
        self.build_early_loss()
        # self.lidar_det_loss = build_loss(loss_occupancy_det)
        if loss_occupancy_aux is not None:
            self.lidar_seg_aux_loss = build_loss(loss_occupancy_aux)
    
    def build_early_loss(self,):
        cfg = self.early_supervision_cfg
        num = cfg.get('num_early_loss_layers', 1)
        for i in range(num):
            if cfg.get(f'layer{i}_loss', None) is not None:
                setattr(self, f'occ_loss_layer{i}', build_loss(cfg[f'layer{i}_loss']))

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w * self.bev_z, self.embed_dims)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_bev=None,  only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w, self.bev_z),device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:

            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                self.bev_z,
                grid_length=(self.real_h / self.bev_h,
                                self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )
            bev_feat, occupancy = outputs
            return bev_feat, bev_feat

        else:
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                self.bev_z,
                grid_length=(self.real_h / self.bev_h,
                                self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev
            )
            bev_feat, occupancy = outputs
            outs = {
                'bev_embed': bev_feat,
                'occupancy': occupancy[-1],
                'early_occupancy': occupancy[:-1]
            }

        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan,
                                                               :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             pts_sem,
             preds_dicts,
             dense_occupancy=None,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        
        occupancy = preds_dicts['occupancy']

        if isinstance(occupancy, SparseConvTensor):
            return self.loss_sparse(preds_dicts, pts_sem)
        else:
            assert isinstance(occupancy, torch.Tensor)
        
        # GT voxel supervision
        pts = pts_sem[:,:3]
        pts_semantic_mask = pts_sem[:,3:4]
        if dense_occupancy is None:
            pts_coors,voxelized_data,voxel_coors = self.voxelize(pts, self.pc_range, self.voxel_lidar)
            voxel_label = self.label_voxelization(pts_semantic_mask, pts_coors, voxel_coors)


        occupancy_pred = occupancy.squeeze(0)
        # occupancy_det_pred = occupancy_det.squeeze(0)

        cls_num,occ_z,occ_h,occ_w = occupancy_pred.shape
        occupancy_label = (torch.ones(1,occ_z,occ_h,occ_w)*cls_num).to(occupancy_pred.device).long()
        
        # Matrix operation acceleration
        if dense_occupancy is None:
            voxel_coors[:,1] = voxel_coors[:,1].clip(min=0,max=occ_z-1)
            voxel_coors[:,2] = voxel_coors[:,2].clip(min=0,max=occ_h-1)
            voxel_coors[:,3] = voxel_coors[:,3].clip(min=0,max=occ_w-1)
            occupancy_label[0,voxel_coors[:,1],voxel_coors[:,2],voxel_coors[:,3]] = voxel_label
        else:
            dense_occupancy = dense_occupancy.long().squeeze(0)
            occupancy_label[0,dense_occupancy[:,0],dense_occupancy[:,1],dense_occupancy[:,2]]=dense_occupancy[:,3]
        
        losses_seg_aux = self.lidar_seg_aux_loss(occupancy_pred.unsqueeze(0),occupancy_label)

        # occupancy_det_label = occupancy_det_label.reshape(-1)
        occupancy_label = occupancy_label.reshape(-1) 

        assert occupancy_label.max()<=cls_num and occupancy_label.min()>=0
        occupancy_pred = occupancy_pred.reshape(cls_num,-1).permute(1,0)

        loss_dict = dict()
        
        # Lidar seg loss
        if dense_occupancy is None:
            num_total_pos = len(voxel_label)
        else:
            num_total_pos = len(dense_occupancy)
        num_total_neg = len(occupancy_label)-num_total_pos
        avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_weight
        if self.sync_cls_avg_factor:
            avg_factor = reduce_mean(
                occupancy_pred.new_tensor([avg_factor]))
        avg_factor = max(avg_factor, 1)

        losses_seg = self.lidar_seg_loss(occupancy_pred, occupancy_label, avg_factor=avg_factor)

        loss_dict['loss_seg'] = losses_seg
        loss_dict['loss_seg_aux'] = losses_seg_aux

        if self.early_supervision_cfg.get('layer0_loss', None) is not None:
            occ_pred_0 = preds_dicts['early_occupancy'][0]
            lidar_seg_loss_layer0 = self.get_layer0_loss(occ_pred_0, pts, pts_semantic_mask)
            loss_dict['losss_occ_layer0'] = lidar_seg_loss_layer0
        
        return loss_dict
    
    def loss_sparse(self, preds_dicts, pts_sem):
        pts = pts_sem[:,:3]
        pts_semantic_mask = pts_sem[:,3:4]

        occupancy = preds_dicts['occupancy']
        early_occ = preds_dicts['early_occupancy']

        dense_occ = early_occ[0]
        sparse_occ = early_occ[1:] + [occupancy,]

        loss_dict = {}
        loss_layer0 = self.get_layer0_loss(dense_occ, pts, pts_semantic_mask)
        loss_dict['loss_occ_layer0'] = loss_layer0

        final = False
        for i, occ in enumerate(sparse_occ):

            if i == len(sparse_occ) - 1:
                final = True

            this_loss_dict = self.get_sparse_occ_loss(occ, pts, pts_semantic_mask, i+1, final)
            loss_dict.update(this_loss_dict)
        
        return loss_dict
    
    def get_sparse_occ_loss(self, occ_sp, pts, pts_semantic_mask, index, final=False):
        assert isinstance(occ_sp, SparseConvTensor)
        occ = occ_sp.features
        loss_dict = {}

        occ_z, occ_h, occ_w = occ_sp.spatial_shape


        vs_x = self.span_x / occ_w
        vs_y = self.span_y / occ_h
        vs_z = self.span_z / occ_z
        voxel_size = (vs_x, vs_y, vs_z)

        pts_coors, _, voxel_coors = self.voxelize(pts, self.pc_range, voxel_size)
        voxel_label = self.label_voxelization(pts_semantic_mask, pts_coors, voxel_coors)
        voxel_coors = self.clip_coors(voxel_coors, occ_z, occ_h, occ_w)

        dense_label = torch.ones(1, occ_z, occ_h, occ_w).to(occ.device).long() * self.num_occ_classes
        dense_label[0, voxel_coors[:,1], voxel_coors[:,2], voxel_coors[:,3]] = voxel_label

        occ_coors = occ_sp.indices.long()

        sparse_label = dense_label[occ_coors[:, 0], occ_coors[:, 1], occ_coors[:, 2], occ_coors[:, 3]]

        if final:
            loss_lovasz = self.lidar_seg_aux_loss(occ, sparse_label)
            loss_dict[f'loss_sparse_lovasz_final'] = loss_lovasz

        num_total_pos = (sparse_label < self.num_occ_classes).sum()
        num_total_neg = len(sparse_label) - num_total_pos
        avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_weight
        if self.sync_cls_avg_factor:
            avg_factor = reduce_mean(occ.new_tensor([avg_factor]))
        avg_factor = max(avg_factor, 1)

        if final:
            loss_seg = self.lidar_seg_loss(occ, sparse_label, avg_factor=avg_factor)
            loss_dict[f'loss_sparse_seg_final'] = loss_seg
        else:
            assert index > 0, 'first layer has dense loss, calculated outside'
            loss_early = getattr(self, f'occ_loss_layer{index}')
            if occ.shape[-1] == 1:
                occ_label = (sparse_label == self.num_occ_classes).long()
                loss_dict[f'loss_sparse_seg_{index}'] = loss_early(occ, occ_label, avg_factor=avg_factor)
            else:
                assert occ.shape[-1] == 17, 'For nus, it is fine to delete this assertion'
                loss_dict[f'loss_sparse_seg_{index}'] = loss_early(occ, sparse_label, avg_factor=avg_factor)

        return loss_dict
    
    def clip_coors(self, coors, z, h, w):
        coors[:,1] = coors[:,1].clip(min=0, max=z-1)
        coors[:,2] = coors[:,2].clip(min=0, max=h-1)
        coors[:,3] = coors[:,3].clip(min=0, max=w-1)
        return coors

    
    def get_layer0_loss(self, occupancy_pred, pts, pts_semantic_mask):

        seg_loss = self.occ_loss_layer0

        occupancy_pred = occupancy_pred.squeeze(0)
        cls_num, occ_z, occ_h, occ_w = occupancy_pred.shape
        assert cls_num == 1, 'occupied or not occupied'

        vs_x = self.span_x / occ_w
        vs_y = self.span_y / occ_h
        vs_z = self.span_z / occ_z
        voxel_size = (vs_x, vs_y, vs_z)

        pts_coors, _, voxel_coors = self.voxelize(pts, self.pc_range, voxel_size)
        # voxel_label = self.label_voxelization(pts_semantic_mask, pts_coors, voxel_coors)

        # assert voxel_label.max().item() <= 16, 'A hard code num classes'

        occupancy_label = torch.ones(1, occ_z, occ_h, occ_w).to(occupancy_pred.device).long()

        voxel_coors[:,1] = voxel_coors[:,1].clip(min=0, max=occ_z-1)
        voxel_coors[:,2] = voxel_coors[:,2].clip(min=0, max=occ_h-1)
        voxel_coors[:,3] = voxel_coors[:,3].clip(min=0, max=occ_w-1)
        # occupancy_label[0, voxel_coors[:,1], voxel_coors[:,2], voxel_coors[:,3]] = voxel_label
        occupancy_label[0, voxel_coors[:,1], voxel_coors[:,2], voxel_coors[:,3]] = 0

        occupancy_pred = occupancy_pred.reshape(cls_num, -1).permute(1,0)
        occupancy_label = occupancy_label.reshape(-1)

        num_total_pos = len(voxel_coors)
        num_total_neg = len(occupancy_label)-num_total_pos
        avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_weight
        if self.sync_cls_avg_factor:
            avg_factor = reduce_mean(
                occupancy_pred.new_tensor([avg_factor]))
        avg_factor = max(avg_factor, 1)

        losses_seg = seg_loss(occupancy_pred, occupancy_label, avg_factor=avg_factor)
        return losses_seg

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        bboxes = torch.zeros(1, 7, dtype=torch.float32)
        bboxes = img_metas[0]['box_type_3d'](bboxes, 7)
        scores = torch.zeros(1, dtype=torch.float32)
        labels = torch.zeros(1, dtype=torch.long)
        ret_list = [[bboxes, scores, labels]]
        return ret_list
    
    def decode_lidar_seg(self,points,occupancy):

        pts_coors,voxelized_data,voxel_coors = self.voxelize(points,self.pc_range,self.voxel_lidar)
        
        # clip out-ranged points
        z_max = int((self.pc_range[5]-self.pc_range[2])/self.voxel_lidar[2])-1
        y_max = int((self.pc_range[4]-self.pc_range[1])/self.voxel_lidar[1])-1
        x_max = int((self.pc_range[3]-self.pc_range[0])/self.voxel_lidar[0])-1
        pts_coors[:,1] = pts_coors[:,1].clip(min=0,max=z_max)
        pts_coors[:,2] = pts_coors[:,2].clip(min=0,max=y_max)
        pts_coors[:,3] = pts_coors[:,3].clip(min=0,max=x_max)

        if isinstance(occupancy, SparseConvTensor):
            assert (z_max + 1, y_max + 1, x_max + 1) == tuple(occupancy.spatial_shape)
            occupancy = occupancy.dense().squeeze(0)
            padding_mask = (occupancy == 0).all(0)
            occupancy[0, padding_mask] = 1 # regarding all empty positions as the first class
            occupancy = occupancy[None, ...]

        pts_pred = occupancy[:,:,pts_coors[:,1],pts_coors[:,2],pts_coors[:,3]].squeeze(0).softmax(dim=0).argmax(dim=0).cpu().numpy()
        return pts_pred
    
    def voxelize(self, points,point_cloud_range,voxelization_size):
        """
        Input:
            points

        Output:
            coors [N,4]
            voxelized_data [M,3]
            voxel_coors [M,4]

        """

        voxel_size = torch.tensor(voxelization_size, device=points.device)
        pc_range = torch.tensor(point_cloud_range, device=points.device)
        coors = torch.div(points[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        coors = coors[:, [2, 1, 0]] # to zyx order

        new_coors, unq_inv  = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)

        voxelized_data, voxel_coors = scatter_v2(points, coors, mode='avg', return_inv=False, new_coors=new_coors, unq_inv=unq_inv)

        batch_idx_pts = torch.zeros(coors.size(0),1).to(device=points.device)
        batch_idx_vox = torch.zeros(voxel_coors.size(0),1).to(device=points.device)

        coors_batch = torch.cat([batch_idx_pts,coors],dim=1)
        voxel_coors_batch = torch.cat([batch_idx_vox,voxel_coors],dim=1)

        return coors_batch.long(),voxelized_data,voxel_coors_batch.long()
    
    def decode_lidar_seg_hr(self,points,occupancy):

        out_h = 512
        out_w = 512
        out_z = 160
        
        self.voxel_lidar = [102.4/out_h,102.4/out_w,8/out_z]

        pts_coors,voxelized_data,voxel_coors = self.voxelize(points,self.pc_range,self.voxel_lidar)
        
        # clip out-ranged points
        z_max = int((self.pc_range[5]-self.pc_range[2])/self.voxel_lidar[2])-1
        y_max = int((self.pc_range[4]-self.pc_range[1])/self.voxel_lidar[1])-1
        x_max = int((self.pc_range[3]-self.pc_range[0])/self.voxel_lidar[0])-1
        pts_coors[:,1] = pts_coors[:,1].clip(min=0,max=z_max)
        pts_coors[:,2] = pts_coors[:,2].clip(min=0,max=y_max)
        pts_coors[:,3] = pts_coors[:,3].clip(min=0,max=x_max)


        new_h = torch.linspace(-1, 1, out_h).view(1,out_h,1).expand(out_z,out_h,out_w)
        new_w = torch.linspace(-1, 1, out_w).view(1,1,out_w).expand(out_z,out_h,out_w)
        new_z = torch.linspace(-1, 1, out_z).view(out_z,1,1).expand(out_z,out_h,out_w)

        grid = torch.cat((new_w.unsqueeze(3),new_h.unsqueeze(3), new_z.unsqueeze(3)), dim=-1)

        grid = grid.unsqueeze(0).to(occupancy.device)

        out_logit = F.grid_sample(occupancy, grid=grid)
        
        pts_pred = out_logit[:,:,pts_coors[:,1],pts_coors[:,2],pts_coors[:,3]].squeeze(0).softmax(dim=0).argmax(dim=0).cpu().numpy()
        return pts_pred
    
    @torch.no_grad()
    def label_voxelization(self, pts_semantic_mask, pts_coors, voxel_coors):
        mask = pts_semantic_mask
        assert mask.size(0) == pts_coors.size(0)

        pts_coors_cls = torch.cat([pts_coors, mask], dim=1) #[N, 5]
        unq_coors_cls, unq_inv, unq_cnt = torch.unique(pts_coors_cls, return_inverse=True, return_counts=True, dim=0) #[N1, 5], [N], [N1]

        unq_coors, unq_inv_2, _ = torch.unique(unq_coors_cls[:, :4], return_inverse=True, return_counts=True, dim=0) #[N2, 4], [N1], [N2,]
        max_num, max_inds = torch_scatter.scatter_max(unq_cnt.float()[:,None], unq_inv_2, dim=0) #[N2, 1], [N2, 1]

        cls_of_max_num = unq_coors_cls[:, -1][max_inds.reshape(-1)] #[N2,]
        cls_of_max_num_N1 = cls_of_max_num[unq_inv_2] #[N1]
        cls_of_max_num_at_pts = cls_of_max_num_N1[unq_inv] #[N]

        assert cls_of_max_num_at_pts.size(0) == mask.size(0)

        cls_no_change = cls_of_max_num_at_pts == mask[:,0] # fix memory bug when scale up
        # cls_no_change = cls_of_max_num_at_pts == mask
        assert cls_no_change.any()

        max_pts_coors = pts_coors.max(0)[0]
        max_voxel_coors = voxel_coors.max(0)[0]
        assert (max_voxel_coors <= max_pts_coors).all()
        bsz, num_win_z, num_win_y, num_win_x = \
        int(max_pts_coors[0].item() + 1), int(max_pts_coors[1].item() + 1), int(max_pts_coors[2].item() + 1), int(max_pts_coors[3].item() + 1)

        canvas = -pts_coors.new_ones((bsz, num_win_z, num_win_y, num_win_x))

        canvas[pts_coors[:, 0], pts_coors[:, 1], pts_coors[:, 2], pts_coors[:, 3]] = \
            torch.arange(pts_coors.size(0), dtype=pts_coors.dtype, device=pts_coors.device)

        fetch_inds_of_points = canvas[voxel_coors[:, 0], voxel_coors[:, 1], voxel_coors[:, 2], voxel_coors[:, 3]]

        assert (fetch_inds_of_points >= 0).all(), '-1 should not be in it.'

        voxel_label = cls_of_max_num_at_pts[fetch_inds_of_points]

        voxel_label = torch.clamp(voxel_label,min=0).long()

        return voxel_label
    
    @torch.no_grad()
    def get_point_pred(self,occupancy,pts_coors,voxel_coors,voxel_label,pts_semantic_mask):
        
        voxel_pred = occupancy[:,:,voxel_coors[:,1],voxel_coors[:,2],voxel_coors[:,3]].squeeze(0).softmax(dim=0).argmax(dim=0).cpu()

        voxel_gt = voxel_label.long().cpu()

        accurate = voxel_pred==voxel_gt

        acc = accurate.sum()/len(voxel_gt)

        pts_pred = occupancy[:,:,pts_coors[:,1],pts_coors[:,2],pts_coors[:,3]].squeeze(0).softmax(dim=0).argmax(dim=0).cpu()
        pts_gt  = pts_semantic_mask.long().squeeze(1).cpu()

        pts_accurate = pts_pred==pts_gt
        pts_acc = pts_accurate.sum()/len(pts_gt)

        return pts_acc
