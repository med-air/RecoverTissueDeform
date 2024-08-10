from PIL import Image
import torch
import argparse
import numpy as np
import sys
sys.path.append('../') # add relative path
from module.sttr import STTR
from dataset.preprocess import normalization, compute_left_occ_region
from utilities.misc import NestedTensor
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str,
                        default=None, help='address of the data')
    parser.add_argument('--model_file_name', type=str,
                        default=None, help='address of the model checkpoint')
    parser.add_argument('--channel_dim', type=int, default=128)
    parser.add_argument('--position_encoding', type=str, default='sine1d_rel')
    parser.add_argument('--num_attn_layers', type=int, default=6)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--regression_head', type=str, default='ot')
    parser.add_argument('--context_adjustment_layer', type=str, default='cal')
    parser.add_argument('--cal_num_blocks', type=int, default=8)
    parser.add_argument('--cal_feat_dim', type=int, default=16)
    parser.add_argument('--cal_expansion_ratio', type=str, default=4)

    args = parser.parse_args()
    address = args.address
    model = STTR(args).cuda().eval()
    model_file_name = args.model_file_name
    checkpoint = torch.load(args.model_file_name)
    pretrained_dict = checkpoint['state_dict']
    model.load_state_dict(pretrained_dict, strict=False)
    print("Pre-trained model successfully loaded.")
    for clip in sorted(os.listdir(address)):
        video_path_left = address + clip + '/img_left/'
        video_path_right = address + clip + '/img_right/'
        save_dir2 = address + clip + '/disp/'
        if not os.path.exists(save_dir2):
            os.mkdir(save_dir2)
        left_paths = os.listdir(video_path_left)
        right_paths = os.listdir(video_path_right)
        left_paths = sorted(left_paths)
        right_paths = sorted(right_paths)
        assert len(left_paths) == len(right_paths)
        file_num = len(left_paths)
        count = 0
        flag = True
        for image_name in left_paths:
            count += 1
            print(count, "/", file_num)
            image_left_path = video_path_left + image_name
            image_right_path = video_path_right + image_name
            left = np.array(Image.open(image_left_path))
            right = np.array(Image.open(image_right_path))
            input_data = {'left': left, 'right': right}
            input_data = normalization(**input_data)
            h, w, _ = left.shape
            bs = 1
            downsample = 3
            col_offset = int(downsample / 2)
            row_offset = int(downsample / 2)
            sampled_cols = torch.arange(col_offset, w, downsample)[None,].expand(bs, -1).cuda()
            sampled_rows = torch.arange(row_offset, h, downsample)[None,].expand(bs, -1).cuda()

            input_data = NestedTensor(input_data['left'].cuda()[None,], input_data['right'].cuda()[None,],
                                      sampled_cols=sampled_cols, sampled_rows=sampled_rows)
            output = model(input_data)

            disp_pred = output['disp_pred'].data.cpu().numpy()[0]
            disp_pred[disp_pred < 0] = 0.
            save_name_depth = os.path.basename(image_name).split(".")[0] + ".npy"
            np.save(save_dir2 + save_name_depth, disp_pred)