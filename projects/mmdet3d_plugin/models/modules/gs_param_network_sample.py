import torch
import torch.nn.functional as F
from torch import nn
from projects.mmdet3d_plugin.ops import SmoothSampler, grid_sample_3d
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.base_module import BaseModule
import numpy as np



class ParamDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size=256, n_blocks=5):
        super().__init__()

        dims = [hidden_size] + [hidden_size for _ in range(n_blocks)] + [out_dim]
        self.num_layers = len(dims)

        for l in range(self.num_layers - 1):
            lin = nn.Linear(dims[l], dims[l + 1])
            setattr(self, "lin" + str(l), lin)

        self.fc_c = nn.ModuleList(
            [nn.Linear(in_dim, hidden_size) for i in range(self.num_layers - 1)]
        )
        self.fc_p = nn.Linear(3, hidden_size)


    def forward(self, points, point_feats):
        x = self.fc_p(points)
        for l in range(self.num_layers - 1):
            x = x + self.fc_c[l](point_feats)
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
        return x



class GSRegresser_Sample(BaseModule):
    def __init__(
        self,
        voxel_size,
        pc_range,
        voxel_shape,
        max_scale,
        interpolate_cfg,
        param_decoder_cfg,
        split_dimensions,
        **kwargs
    ):
        super(GSRegresser_Sample, self).__init__()
        self.fp16_enabled = kwargs.get("fp16_enabled", False)
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.voxel_shape = voxel_shape
        self.interpolate_cfg = interpolate_cfg
        self.split_dimensions = split_dimensions
        self.max_scale = max_scale

        self.param_decoder = ParamDecoder(**param_decoder_cfg)

        self.rotation_activation = torch.nn.functional.normalize
        self.scaling_activation = nn.Softplus(beta=100)
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid


    def flatten_vector(self, x):
        # Gets rid of the image dimensions and flattens to a point list
        # N, Z, C -> N*Z, C 
        return x.reshape(-1, x.shape[-1])

    def interpolate_feats(self, pts, feats_volume):
        # pts: [3072, 72, 3]
        pc_range = pts.new_tensor(self.pc_range)
        norm_coords = (pts - pc_range[:3]) / (
            pc_range[3:] - pc_range[:3]
        )
        assert (
            self.voxel_shape[0] == feats_volume.shape[3]
            and self.voxel_shape[1] == feats_volume.shape[2]
            and self.voxel_shape[2] == feats_volume.shape[1]
        )
        norm_coords = norm_coords * 2 - 1
        if self.interpolate_cfg["type"] == "SmoothSampler":
            feats = (
                SmoothSampler.apply(
                    feats_volume.unsqueeze(0),
                    norm_coords[None, None, ...],
                    self.interpolate_cfg["padding_mode"],
                    True,
                    False,
                )
                .squeeze(0)
                .squeeze(1)
                .permute(1, 2, 0)
            )
        else:
            feats = (
                grid_sample_3d(feats_volume.unsqueeze(0), norm_coords[None, None, ...])
                .squeeze(0)
                .squeeze(1)
                .permute(1, 2, 0)
            )
        # [3072, 72, 32]
        return feats

    @auto_fp16(apply_to=("points", "feature_volume"))
    def get_param(self, points, feature_volume):
        """predict the sdf value for ray samples"""
        point_features = self.interpolate_feats(points, feature_volume)
        param = self.param_decoder(points, point_features)
        return param


    @auto_fp16(out_fp32=True)
    def forward(self, feature_volume, ray_samples, return_alphas=False):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        outputs = {}

        points = ray_samples['pts_sampled']
        gaussian_param = self.get_param(points, feature_volume)
        rot, scale, opacity, rgb = gaussian_param.split(self.split_dimensions, dim=-1)

        return {
            'rot':self.flatten_vector(self.rotation_activation(rot)),
            'scale':self.flatten_vector(torch.clamp_max(self.scaling_activation(scale), self.max_scale)), 
            'opacity':self.flatten_vector(self.opacity_activation(opacity)),
            'rgb':self.flatten_vector(self.rgb_activation(rgb)),
        }
