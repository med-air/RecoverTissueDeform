import torch
import cv2
import os
from albumentations.pytorch.functional import img_to_tensor
from albumentations import Normalize
from data.utils import save_mask, save_img
from network.swin import get_swin
from network.swin.mmseg.ops import resize
import argparse
import torch.nn as nn
normalize = Normalize()
import os
import skimage
import cv2

def load_model(model_path):
    model = get_swin(2)
    assert (torch.cuda.is_available())
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(model_path))
    return model


def predict_from_files(model, filepath, savedir):
    if not os.path.exists(filepath):
        raise ("no such a folder!")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    filelist = os.listdir(filepath)
    filelist.sort()
    with torch.no_grad():
        for batch_itr, filename in enumerate(filelist):
            img = cv2.imread(filepath + filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # crop to such a size
            # img = img[60:1020, :, :]
            h, w = img.shape[:2]
            img = cv2.resize(img, dsize=(w, h))
            img = normalize.apply(img)
            input_im = img_to_tensor(img)
            input_im = input_im[None, :]
            input_im = input_im.cuda()
            out = model(input_im)[0]
            img_name = filename
            output_classes = out.detach().cpu().numpy().argmax(axis=1)
            save_mask(output_classes, None, savedir + img_name)
            save_img(input_im, savedir + img_name.replace('.jpg', 'ori.png'))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', type=str,
                        default=None, help='address of the data')
    parser.add_argument('--model_path', type=str,
                        default='pretrain/finetune_swin.pth', help='address of the model checkpoint')
    args = parser.parse_args()
    address = args.address
    model_path = args.model_path
    model = load_model(model_path)
    model = model.cuda()
    model.eval()
    for clip in os.listdir(address):
        print(clip)
        #for action in os.listdir(address + clip):
        #    for seq in os.listdir(address + clip + "/" + action):
        file_path = os.path.join(address, clip, "img_left")
        save_dir = os.path.join(address, clip, "tool_left")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        predict_from_files(model, file_path, save_dir)
        file_path = os.path.join(address, clip, "img_right")
        save_dir = os.path.join(address, clip, "tool_right")
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        predict_from_files(model, file_path, save_dir)

    for clip in os.listdir(address):
        address = os.path.join(address, clip, "tool_left")
        address_save = os.path.join(address, clip, "tool_mask_left")
        if not os.path.isdir(address_save):
            os.mkdir(address_save)
        adress = os.listdir(address)
        for i in adress:
            if i.split(".")[-1] == "jpg":
                tool_t0 = address + i
                tool_t0 = cv2.imread(tool_t0)
                tool_t0 = cv2.resize(tool_t0, [740, 540])
                tool_t0 = tool_t0[:, :, 1] > 150
                tool_t0 = 1. - tool_t0
                for _ in range(20):
                    tool_t0 = skimage.morphology.binary_erosion(tool_t0)
                tool_t0 = tool_t0.astype(int)
                cv2.imwrite(address_save + i.split(".")[0] + '.png', tool_t0)

    for clip in os.listdir(address):
        address = os.path.join(address, clip, "tool_right")
        address_save = os.path.join(address, clip, "tool_mask_right")
        if not os.path.isdir(address_save):
            os.mkdir(address_save)
        adress = os.listdir(address)
        for i in adress:
            if i.split(".")[-1] == "jpg":
                tool_t0 = address + i
                tool_t0 = cv2.imread(tool_t0)
                tool_t0 = cv2.resize(tool_t0, [740, 540])
                tool_t0 = tool_t0[:, :, 1] > 150
                tool_t0 = 1. - tool_t0
                for _ in range(20):
                    tool_t0 = skimage.morphology.binary_erosion(tool_t0)
                tool_t0 = tool_t0.astype(int)
                cv2.imwrite(address_save + i.split(".")[0] + '.png', tool_t0)
