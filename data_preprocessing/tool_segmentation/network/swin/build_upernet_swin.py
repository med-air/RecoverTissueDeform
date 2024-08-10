from swin_transformer import SwinTransformer
from fcn_head import FCNHead
from uper_head import UPerHead
import torch
from mmcv.runner import build_optimizer

if __name__ == "__main__":
    a = SwinTransformer(embed_dim=128,
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
    b = UPerHead(in_channels=[128, 256, 512, 1024],
                 in_index=[0, 1, 2, 3],
                 pool_scales=(1, 2, 3, 6),
                 channels=512,
                 dropout_ratio=0.1,
                 num_classes=4,
                 # norm_cfg=norm_cfg,
                 align_corners=False)
    c = FCNHead(in_channels=512,
                in_index=2,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=4,
                # norm_cfg=norm_cfg,
                align_corners=False)
    inp = torch.zeros([1, 3, 1024, 1480])
    meta = a(inp)
    out1 = b(meta)
    out2 = c(meta)
    aa = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

    build_optimizer(a, aa)
    print(b)
    print(c)
