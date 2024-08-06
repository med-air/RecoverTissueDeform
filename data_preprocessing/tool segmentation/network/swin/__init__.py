from .swin_transformer import SwinTransformer
from .fcn_head import FCNHead
from .uper_head import UPerHead
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from network.mmcv_custom import load_checkpoint

# the setting is swin base
class Swin(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # norm_cfg = dict(type='SyncBN', requires_grad=True)
        norm_cfg = dict(type='BN', requires_grad=True)
        self.backbone = SwinTransformer(embed_dim=128,
                                       depths=[2, 2, 18, 2],
                                       num_heads=[4, 8, 16, 32],
                                       window_size=7,
                                       mlp_ratio=4.,
                                       qkv_bias=True,
                                       qk_scale=None,
                                       drop_rate=0.,
                                       attn_drop_rate=0.,
                                       drop_path_rate=0.3,
                                       ape=False,
                                       patch_norm=True,
                                       out_indices=(0, 1, 2, 3),
                                       use_checkpoint=False)
        self.decode_head = UPerHead(in_channels=[128, 256, 512, 1024],
                                in_index=[0, 1, 2, 3],
                                pool_scales=(1, 2, 3, 6),
                                channels=512,
                                dropout_ratio=0.1,
                                num_classes=num_classes,
                                norm_cfg=norm_cfg,
                                align_corners=False)
        self.auxiliary_head = FCNHead(in_channels=512,
                                   in_index=2,
                                   channels=256,
                                   num_convs=1,
                                   concat_input=False,
                                   dropout_ratio=0.1,
                                   num_classes=num_classes,
                                   norm_cfg=norm_cfg,
                                   align_corners=False)
        self.init_weights(None)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger() # dont set logger currently.
            load_checkpoint(self, pretrained, strict=False, logger=None)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        feature = self.backbone(x)
        out1 = self.decode_head(feature)
        out2 = self.auxiliary_head(feature)
        # out2 = None
        return [out1, out2]



def get_swin(num_classes):
    net = Swin(num_classes)
    return net
