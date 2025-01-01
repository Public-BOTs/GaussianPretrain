from mmcv.runner import BaseModule
from torch import nn as nn
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
import torch.nn.functional as F
import torch


from spconv.pytorch import SparseConvTensor, SparseSequential
from mmdet3d.ops import make_sparse_convmodule

from ipdb import set_trace


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class OccupancyOnlyDecoder(BaseModule):

    def __init__(self,
                 bev_h=50,
                 bev_w=50,
                 bev_z=8,
                 conv_up_layer=2,
                 embed_dim=256,
                 out_dim=64,
                 early_supervision_cfg=dict(),
                 ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.out_dim = out_dim
        self.conv_up_layer = conv_up_layer
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(embed_dim,embed_dim,(1,5,5),padding=(0,2,2)),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(embed_dim, embed_dim, (1, 4, 4), stride=(1, 2, 2), padding=(0,1,1)),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=True),

            nn.ConvTranspose3d(embed_dim, self.out_dim, (2, 4, 4), stride=(2, 2, 2),padding=(0,1,1)),
            nn.BatchNorm3d(self.out_dim),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)

        cfg = early_supervision_cfg
        if cfg.get('layer0_loss', None) is not None:
            self.mlp_decoder0 = build_transformer_layer_sequence(cfg['layer0_decoder'])
        

                
    def forward(self, inputs):
        out_list = []
        
        voxel_input = inputs.view(1,self.bev_w,self.bev_h,self.bev_z, -1).permute(0,4,3,1,2) #[bsz, c, z, w, h]

        if hasattr(self, 'mlp_decoder0'):
            occ0 = self.mlp_decoder0(voxel_input)
            out_list.append(occ0)

        voxel_feat = self.upsample(voxel_input)
        out_list.append(voxel_feat)
        
        return out_list

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class SparseOccupancyDecoder(BaseModule):

    def __init__(self,
                 bev_h=50,
                 bev_w=50,
                 bev_z=8,
                 conv_up_layer=2,
                 embed_dim=256,
                 out_dim=64,
                 norm_cfg=dict(),
                 early_supervision_cfg=dict(),
                 sparse_cfg=dict(),
                 ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z = bev_z
        self.out_dim = out_dim
        self.conv_up_layer = conv_up_layer
        self.sparse_cfg = sparse_cfg
        
        self.num_layers = len(sparse_cfg['strides'])

        cfg = early_supervision_cfg
        # if cfg.get('layer0_loss', None) is not None:
        #     self.mlp_decoder0 = build_transformer_layer_sequence(cfg['layer0_decoder'])


        for loss_i in range(cfg.get('num_early_loss_layers', 1)):
            if cfg.get(f'layer{loss_i}_loss', None) is not None:
                setattr(self, f'mlp_decoder{loss_i}', build_transformer_layer_sequence(cfg[f'layer{loss_i}_decoder']))

        for i in range(self.num_layers):
            stride = sparse_cfg['strides'][i]

            if max(stride) == 1:
                conv_type = 'SubMConv3d'
            else:
                conv_type = 'SparseConvTranspose3d'

            this_deconv = make_sparse_convmodule(
                sparse_cfg['in_channels'][i],
                sparse_cfg['out_channels'][i],
                kernel_size=sparse_cfg['kernel_sizes'][i],
                indice_key=f'transpose{i}',
                stride=stride,
                norm_cfg=sparse_cfg['norm_cfg'],
                padding=sparse_cfg['paddings'][i],
                conv_type=conv_type,
            )
            if sparse_cfg.get('num_attached_subm', None) is None:
                setattr(self, f'upsample_{i}', this_deconv)
            else:
                this_convs = [this_deconv,]
                for subm_i in range(sparse_cfg['num_attached_subm'][i]):
                    this_subm = make_sparse_convmodule(
                        sparse_cfg['out_channels'][i],
                        sparse_cfg['out_channels'][i],
                        kernel_size=sparse_cfg['subm_kernel_sizes'][i],
                        indice_key=f'subm_{i}_{subm_i}',
                        stride=1,
                        norm_cfg=sparse_cfg['norm_cfg'],
                        conv_type='SubMConv3d',
                    )
                    this_convs.append(this_subm)
                setattr(self, f'upsample_{i}', SparseSequential(*this_convs))

        if 'extra_layer0_conv' in cfg:
            num_extra_conv = cfg['extra_layer0_conv']['num_extra_conv']
            conv_list = []
            for i in range(num_extra_conv):

                conv_list += [
                    nn.Conv3d(embed_dim, embed_dim, (3,3,3), padding=(1,1,1)),
                    nn.BatchNorm3d(embed_dim),
                    nn.ReLU(inplace=True)
                ]
        
            self.extra_layer0_conv = nn.Sequential(*conv_list)
        

                
    def forward(self, inputs):
        out_list = []
        
        voxel_input = inputs.view(1, self.bev_h, self.bev_w, self.bev_z, -1).permute(0,4,3,1,2) #[bsz, c, z, h, w]

        if hasattr(self, 'extra_layer0_conv'):
            voxel_input = self.extra_layer0_conv(voxel_input)

        if hasattr(self, 'mlp_decoder0'):
            occ0 = self.mlp_decoder0(voxel_input)
            out_list.append(occ0)

        sparse_data = self.sparsify(voxel_input, occ0)

        for i in range(self.num_layers):
            this_conv = getattr(self, f'upsample_{i}')
            sparse_data = this_conv(sparse_data)
            if hasattr(self, f'mlp_decoder{i+1}'):
                this_mlp_decoder = getattr(self, f'mlp_decoder{i+1}')
                this_occ = this_mlp_decoder(sparse_data)
                out_list.append(this_occ)
                
                sparse_data = self.prune(sparse_data, this_occ, i+1)

        out_list.append(sparse_data)
        
        return out_list
    
    def prune(self, sparse_feats, this_occ, index):
        feats = sparse_feats.features
        coors = sparse_feats.indices
        old_len = len(feats)

        sp_shape = sparse_feats.spatial_shape
        bsz = sparse_feats.batch_size

        assert (this_occ.indices == coors).all()

        occ = this_occ.features

        thresh = self.sparse_cfg['pruning_thresh'][index]
        max_ratio = self.sparse_cfg['max_keep_ratio'][index]
        
        if max_ratio < 0: # conduct random sampling for debugging
            max_ratio = abs(max_ratio)
            keep_num = int(max_ratio * old_len)
            top_inds = torch.randperm(old_len, device=feats.device)[:keep_num]
            feats = feats[top_inds]
            coors = coors[top_inds]
            sparse_input = SparseConvTensor(feats, coors, sp_shape, bsz)
            return sparse_input

        occ_prob = occ.sigmoid().reshape(-1)
        keep_mask = occ_prob > thresh


        min_keep_num = self.sparse_cfg.get('min_keep_num', 500)
        if keep_mask.sum() < min_keep_num:
            print(f'Got too small number of occupied voxels at layer {index}!!!!')
            top_inds = torch.sort(occ_prob, descending=True)[1]
            top_inds = top_inds[:min_keep_num]
            feats = feats[top_inds]
            coors = coors[top_inds]
            pruned_data = SparseConvTensor(feats, coors, sp_shape, bsz)
            return pruned_data


        feats = feats[keep_mask]
        coors = coors[keep_mask]
        occ_prob = occ_prob[keep_mask]

        if len(feats) > old_len * max_ratio:

            top_inds = torch.sort(occ_prob, descending=True)[1]
            top_inds = top_inds[:int(old_len * max_ratio)]

            feats = feats[top_inds]
            coors = coors[top_inds]
            occ_prob = occ_prob[top_inds]
        
        pruned_data = SparseConvTensor(feats, coors, sp_shape, bsz)

        return pruned_data

    
    def sparsify(self, voxel_input, occ_prob):

        device = voxel_input.device

        occ_prob = occ_prob.sigmoid()
        occ_prob = occ_prob.permute(0, 2, 3, 4, 1)
        assert occ_prob.shape[-1] == 1
        occ_prob = occ_prob.reshape(-1)

        bsz, C, z, y, x = voxel_input.shape 
        voxel_input = voxel_input.permute(0, 2, 3, 4, 1).reshape(-1, C) # to [bsz, z, y, x, C]

        coors = -1 * torch.ones(bsz, z, y, x, 4, dtype=torch.int32, device=device)
        coors[..., 0] = torch.arange(bsz, dtype=torch.int32, device=device)[:, None, None, None]
        coors[..., 1] = torch.arange(z, dtype=torch.int32, device=device)[None, :, None, None]
        coors[..., 2] = torch.arange(y, dtype=torch.int32, device=device)[None, None, :, None]
        coors[..., 3] = torch.arange(x, dtype=torch.int32, device=device)[None, None, None, :]

        coors = coors.reshape(-1, 4)

        thresh = self.sparse_cfg['pruning_thresh']
        if isinstance(thresh, (list, tuple)):
            thresh = thresh[0]

        max_ratio = self.sparse_cfg['max_keep_ratio']
        if isinstance(max_ratio, (list, tuple)):
            max_ratio = max_ratio[0]

        keep_mask = occ_prob > thresh

        min_keep_num = self.sparse_cfg.get('min_keep_num', 500)
        if keep_mask.sum() < min_keep_num:
            print('Got too small number of occupied voxels !!!!')
            top_inds = torch.sort(occ_prob, descending=True)[1]
            top_inds = top_inds[:min_keep_num]
            voxel_input = voxel_input[top_inds]
            coors = coors[top_inds]
            sparse_input = SparseConvTensor(voxel_input, coors, (z, y, x), bsz)
            return sparse_input


        voxel_input = voxel_input[keep_mask]
        coors = coors[keep_mask]
        occ_prob = occ_prob[keep_mask]

        dense_num = bsz * z * y * x

        if len(voxel_input) > dense_num * max_ratio:

            top_inds = torch.sort(occ_prob, descending=True)[1]
            top_inds = top_inds[:int(dense_num * max_ratio)]

            voxel_input = voxel_input[top_inds]
            coors = coors[top_inds]
            occ_prob = occ_prob[top_inds]
        
        sparse_input = SparseConvTensor(voxel_input, coors, (z, y, x), bsz)

        return sparse_input
        

