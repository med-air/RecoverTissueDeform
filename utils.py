from pytorch3d.structures.volumes import Volumes
from pytorch3d.ops import add_pointclouds_to_volumes
import torch as th
import shutil
import os
def save_checkpoint(state, is_best, checkpoint):
    filepath_last = os.path.join(checkpoint, "last.pth.tar")
    filepath_best = os.path.join(checkpoint, "best.pth.tar")
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Masking directory {}").format(checkpoint)
        os.mkdir(checkpoint)
    else:
        print("Checkpoint DIrectory exists!")
    th.save(state, filepath_last)
    if is_best:
        if os.path.isfile(filepath_best):
            os.remove(filepath_best)
        shutil.copyfile(filepath_last, filepath_best)



