import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import time
import copy
import numpy as np
from mmdet3d.models import builder
import mmdet3d

from projects.panoocc.bevformer.detectors.pano_occ import PanoOcc

@DETECTORS.register_module()
class PanoOcc_pretrain(PanoOcc):
    """PanoOcc.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 render_head,
                 fuse_bev=False,
                 **kwargs
                 ):

        super(PanoOcc_pretrain,
              self).__init__(**kwargs)
        
        self.render_head = builder.build_head(render_head)
        self.fuse_bev = fuse_bev


    def forward_pts_train(self,
                          pts_feats,
                          img_metas,
                          points,
                          img,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        bev_feats = self.pts_bbox_head(pts_feats, img_metas, only_bev=True)

        bev_h = 100
        bev_w = 100
        bev_z = 16

        if self.fuse_bev:
            kwargs = {'img_metas' : img_metas}
            bev_embed = self.pts_bbox_head.transformer.bev_temporal_fuse(bev_feats, prev_bev, bev_h, bev_w, bev_z, **kwargs)

            bev_embed_vox = bev_embed.view(1,bev_h*bev_w,bev_z,-1)

            voxel_feat, _ = self.pts_bbox_head.transformer.voxel_decoder(bev_embed_vox)
        else:
            voxel_feat = bev_feats.reshape(len(bev_feats), bev_h, bev_w, bev_z, -1)
            voxel_feat = voxel_feat.permute(0, 4, 3, 1, 2)

        batch_rays = self.render_head.sample_rays(points, img.unsqueeze(0), img_metas)
        out_dict = self.render_head(
            points, voxel_feat, img_metas, img_depth=None, batch_rays=batch_rays
        )
        losses = self.render_head.loss(out_dict, batch_rays)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        is_training = self.training
        self.eval()

        prev_bev_lst = []
        with torch.no_grad():
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, only_bev=True)
                prev_bev = prev_bev.permute(0, 2, 1)
                prev_bev = prev_bev.reshape(prev_bev.shape[0], -1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, self.pts_bbox_head.bev_z)
                prev_bev_lst.append(prev_bev)
        if is_training:
            self.train()
        # (bs, num_queue, embed_dims, H, W)
        return torch.stack(prev_bev_lst, dim=1)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      voxel_semantics=None,
                      mask_lidar=None,
                      mask_camera=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """

        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        if prev_img.size(1)==0:
            prev_bev = None
        else:
            prev_img_metas = copy.deepcopy(img_metas)
            prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)
            # prev_bev = prev_bev.view(prev_bev.shape[0], prev_bev.shape[1], -1).permute(0, 2, 1)

        img_metas = [each[len_queue - 1] for each in img_metas]
        if not img_metas[0]['prev_bev_exists']:
            prev_bev = None
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, img_metas, points, img,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas,
                     img=None,
                     voxel_semantics=None,
                     mask_lidar=None,
                     mask_camera=None,
                     **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = []
            self.prev_frame_info["ego2global_transformation_lst"] = []
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = []
            self.prev_frame_info["ego2global_transformation_lst"] = []

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        self.prev_frame_info["ego2global_transformation_lst"].append(img_metas[0][0]["ego2global_transformation"])

        img_metas[0][0]["ego2global_transform_lst"] = self.prev_frame_info["ego2global_transformation_lst"][-1::-self.time_interval][::-1]
        prev_bev = self.prev_frame_info['prev_bev'][-self.time_interval:: -self.time_interval][:: -1]
        prev_bev = torch.stack(prev_bev, dim=1) if len(prev_bev) > 0 else None

        new_prev_bev, occ_results = self.simple_test(
            img_metas[0], img[0], prev_bev=prev_bev, **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.

        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        new_prev_bev = new_prev_bev.permute(0, 2, 1).reshape(1, -1, self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w, self.pts_bbox_head.bev_z)
        self.prev_frame_info['prev_bev'].append(new_prev_bev)

        while len(self.prev_frame_info["prev_bev"]) >= self.pts_bbox_head.transformer.temporal_encoder.num_bev_queue * self.time_interval:
            self.prev_frame_info["prev_bev"].pop(0)
            self.prev_frame_info["ego2global_transformation_lst"].pop(0)

        return occ_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, test=True)

        occ = self.pts_bbox_head.get_occ(
            outs, img_metas, rescale=rescale)

        return outs['bev_embed'], occ

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        # bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, occ = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        # for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
        #     result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, occ
