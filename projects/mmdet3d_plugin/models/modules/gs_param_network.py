
import torch
from torch import nn
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import builder
from mmcv.cnn.bricks.conv_module import ConvModule
from .. import modules

class GSRegresser(BaseModule):
    def __init__(self, head_dim=32, fp16_enabled=False, max_scale=0.2,  **kwargs):
        super(GSRegresser, self).__init__()
        if fp16_enabled:
            self.fp16_enabled = True

        self.head_dim = head_dim
        self.max_scale = max_scale

        self.rot_head = nn.Sequential(
            nn.Conv3d(
                self.head_dim,
                self.head_dim,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                self.head_dim,
                4,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )
        self.scale_head = nn.Sequential(
            nn.Conv3d(
                self.head_dim,
                self.head_dim,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                self.head_dim,
                3,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Softplus(beta=100)
        )

        self.opacity_head = nn.Sequential(
            nn.Conv3d(
                self.head_dim,
                self.head_dim,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                self.head_dim,
                1,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Sigmoid()
        )

        self.offset_head = nn.Sequential(
            nn.Conv3d(
                self.head_dim,
                self.head_dim,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                self.head_dim,
                3,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Sigmoid()
        )

        self.rgb_head = nn.Sequential(
            nn.Conv3d(
                self.head_dim,
                self.head_dim,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                self.head_dim,
                3,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            nn.Sigmoid()
        )


    def flatten_vector(self, x):
        # Gets rid of the image dimensions and flattens to a point list
        # B x C x H x W -> B x C x N -> B x N x C
        return x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

    @auto_fp16(apply_to=("uni_feats"))
    def forward(self, uni_feats):

        # rot head
        rot_out = self.rot_head(uni_feats)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        # scale head
        # scale_out = torch.clamp_max(self.scale_head(uni_feats), 0.01)
        scale_out = torch.clamp_max(self.scale_head(uni_feats), self.max_scale)

        # opacity head
        opacity_out = self.opacity_head(uni_feats)

        # offset head
        offset_out = self.offset_head(uni_feats)

        # rgb head
        rgb_out = self.rgb_head(uni_feats)


        return {
            'rot':self.flatten_vector(rot_out),
            'scale':self.flatten_vector(scale_out), 
            'opacity':self.flatten_vector(opacity_out),
            'rgb':self.flatten_vector(rgb_out),
            'offset': self.flatten_vector(offset_out),
        }


class GSRegresser_Resnet(BaseModule):
    def __init__(self, head_dim=32, fp16_enabled=False, gs_encoder_cfg=None, split_dimensions=[4, 3, 1, 3, 3], max_scale=0.20, bias=None, scale=None, **kwargs):
        super(GSRegresser_Resnet, self).__init__()
        # if fp16_enabled:
        self.fp16_enabled = fp16_enabled

        self.head_dim = head_dim
        self.split_dimensions = split_dimensions
        self.max_scale = max_scale

        self.gs_encoder = getattr(utils, gs_encoder_cfg.type)(
                **gs_encoder_cfg
            ) 

        self.final_conv = ConvModule(
                self.head_dim,
                sum(self.split_dimensions),       # 4 + 3 + 1 + 3 + 3 (rot, scale , opacity, offset, rgb)
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                conv_cfg=dict(type='Conv3d'))
        
        # Activation functions for different parameters
        self.rotation_activation = torch.nn.functional.normalize
        self.scaling_activation = nn.Softplus(beta=100)
        self.opacity_activation = torch.sigmoid
        self.rgb_activation = torch.sigmoid
        self.offset_activation = torch.sigmoid

        self.init(bias, scale)

    def init(self, bias, scale):
        start_channels = 0
        if bias == None or scale == None:
            return

        for out_channels, b, s in zip(self.split_dimensions, bias, scale):
            nn.init.xavier_uniform_(self.final_conv.conv.weight[start_channels:start_channels+out_channels,
                                :, :, :, :], s)
            nn.init.constant_(self.final_conv.conv.bias[start_channels:start_channels+out_channels,], b)
            start_channels += out_channels

    def flatten_vector(self, x):
        # Gets rid of the image dimensions and flattens to a point list
        # B x C x H x W -> B x C x N -> B x N x C
        return x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    

    @auto_fp16(apply_to=("uni_feats"))
    def forward(self, uni_feats):
        gs_feats = self.gs_encoder(uni_feats)
        out = self.final_conv(gs_feats[0])
        rot, scale, opacity, rgb, offset = out.split(self.split_dimensions, dim=1)

        return {
            'rot':self.flatten_vector(self.rotation_activation(rot)),
            'scale':self.flatten_vector(torch.clamp_max(self.scaling_activation(scale), self.max_scale)), 
            'opacity':self.flatten_vector(self.opacity_activation(opacity)),
            'rgb':self.flatten_vector(self.rgb_activation(rgb)),
            'offset': self.flatten_vector(self.offset_activation(offset)),
        }








class GSRegresser_Resnet(BaseModule):
    def __init__(self, head_dim=32, fp16_enabled=False, gs_encoder_cfg=None, split_dimensions=[4, 3, 1, 3, 3], max_scale=0.20, bias=None, scale=None, **kwargs):
        super(GSRegresser_Resnet, self).__init__()
        # if fp16_enabled:
        self.fp16_enabled = fp16_enabled





if __name__=="__main__":
    gs_r = GSRegresser()
    unifeats = torch.rand(1, 32, 5, 128, 128)
    rot_out, scale_out, opacity_out = gs_r(unifeats)
    import pdb; pdb.set_trace()
    
