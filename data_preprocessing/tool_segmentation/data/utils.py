import torch
import cv2
import numpy as np
from PIL import Image
from albumentations import Normalize
normalize = Normalize()


class_color=[[0,0,0], [0,255,0], [0,255,255], [255,255,255],
            [255,55,0], [24,55,125], [187,155,25], [0,255,125], [255,255,125],
            [123,15,175], [124,155,5], [12,255,141]]


def save_mask(tensor, gt_img=None, img_path='./'):
    img = tensor2mask(tensor)
    image_pil = Image.fromarray(img)
    image_pil.save(img_path)

    if gt_img is None:
        return

    gt_img = tensor2mask(gt_img)
    gt_pil = Image.fromarray(gt_img)
    img_path = img_path.replace('.png', '_gt.png')
    gt_pil.save(img_path)


def tensor2mask(tensor):
    if tensor.dtype in ['int64', 'uint8', 'float32', 'float64']:
        pass
    elif len(tensor.size()) == 4:
        tensor = tensor.cpu().float().detach().numpy().astype(np.uint8)
        tensor = np.argmax(tensor, axis=1)
    else:
        tensor = tensor.cpu().float().detach().numpy().astype(np.uint8)
    temp = tensor[0]
    h, w = temp.shape
    full_mask = np.zeros((h, w, 3))
    for mask_label, sub_color in enumerate(class_color):
        full_mask[temp == mask_label, 0] = sub_color[2]
        full_mask[temp == mask_label, 1] = sub_color[1]
        full_mask[temp == mask_label, 2] = sub_color[0]
        if mask_label > 3:
            break
    return full_mask.astype(np.uint8)


def tensor2img(tensor):
    temp = tensor[0].cpu().float().numpy().transpose((1, 2, 0))
    if temp.shape[2] == 1:
        temp = np.concatenate((temp, temp, temp), axis=2)
    mean = np.array(normalize.mean, dtype=np.float32)
    mean *= normalize.max_pixel_value

    std = np.array(normalize.std, dtype=np.float32)
    std *= normalize.max_pixel_value
    temp *= std
    temp += mean
    return temp.astype(np.uint8)


def save_img(img, img_pth='./'):
    img = tensor2img(img)
    # img_pil = Image.fromarray(img)
    img_pil = Image.fromarray(img)
    img_pil.save(img_pth)