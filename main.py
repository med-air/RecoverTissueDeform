import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from network import Unet_multimodal
from trainer import trainer_surgery
from torchvision.models.optical_flow import raft_small

parser = argparse.ArgumentParser()
parser.add_argument('--train_data', type=str,
                    default='train_new.pkl', help='root dir for data')
parser.add_argument('--eval_data', type=str,
                    default='val_new.pkl', help='root dir for data')
parser.add_argument('--output_dir', type=str, default='../experiments/optical_v10/', help='output dir')
parser.add_argument('--max_epochs', type=int,
                    default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--base_lr', type=float,  default=0.0002,
                    help='segmentation network learning rate')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    flow_net = raft_small(pretrained = True).cuda()
    refine_net = Unet_multimodal(inshape=[64,64,64], infeats=3, outfeats=3).cuda()
    trainer = trainer_surgery
    trainer(args, flow_net, refine_net, args.output_dir)