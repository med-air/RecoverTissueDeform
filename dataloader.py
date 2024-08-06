from torch.utils.data import Dataset
import torch as th
from PIL import Image
import pickle
import numpy as np
import cv2

class Surgical_dataset(Dataset):
    def __init__(self, data_dir):
        with open(data_dir, "rb") as f:
            self.img_list = pickle.load(f)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        direct = self.img_list[idx]
        path = direct["path"]
        left_t0 = path + "img_left/" + direct["t0"] + ".jpg"
        left_t0 = np.array(Image.open(left_t0))
        left_t1 = path + "img_left/" + direct["t1"] + ".jpg"
        left_t1 = np.array(Image.open(left_t1))
        left_t2 = path + "img_left/" + direct["t2"] + ".jpg"
        left_t2 = np.array(Image.open(left_t2))
        left_t3 = path + "img_left/" + direct["t3"] + ".jpg"
        left_t3 = np.array(Image.open(left_t3))
        left_t4 = path + "img_left/" + direct["t4"] + ".jpg"
        left_t4 = np.array(Image.open(left_t4))

        right_t0 = path + "img_right/" + direct["t0"] + ".jpg"
        right_t0 = np.array(Image.open(right_t0))
        right_t1 = path + "img_right/" + direct["t1"] + ".jpg"
        right_t1 = np.array(Image.open(right_t1))
        right_t2 = path + "img_right/" + direct["t2"] + ".jpg"
        right_t2 = np.array(Image.open(right_t2))
        right_t3 = path + "img_right/" + direct["t3"] + ".jpg"
        right_t3 = np.array(Image.open(right_t3))
        right_t4 = path + "img_right/" + direct["t4"] + ".jpg"
        right_t4 = np.array(Image.open(right_t4))

        disp_t0 = path + "disp/" + direct["t0"] + ".npy"
        disp_t0 = np.load(disp_t0)
        disp_t1 = path + "disp/" + direct["t1"] + ".npy"
        disp_t1 = np.load(disp_t1)
        disp_t2 = path + "disp/" + direct["t2"] + ".npy"
        disp_t2 = np.load(disp_t2)
        disp_t3 = path + "disp/" + direct["t3"] + ".npy"
        disp_t3 = np.load(disp_t3)
        disp_t4 = path + "disp/" + direct["t4"] + ".npy"
        disp_t4 = np.load(disp_t4)


        tool_t0 = path + "tool_mask/" + direct["t0"] + ".png"
        tool_t0 = cv2.imread(tool_t0, -1)
        tool_t0 = tool_t0[:512, 226:-2]
        tool_t1 = path + "tool_mask/" + direct["t1"] + ".png"
        tool_t1 = cv2.imread(tool_t1, -1)
        tool_t1 = tool_t1[:512, 226:-2]
        tool_t2 = path + "tool_mask/" + direct["t2"] + ".png"
        tool_t2 = cv2.imread(tool_t2, -1)
        tool_t2 = tool_t2[:512, 226:-2]
        tool_t3 = path + "tool_mask/" + direct["t3"] + ".png"
        tool_t3 = cv2.imread(tool_t3, -1)
        tool_t3 = tool_t3[:512, 226:-2]
        tool_t4 = path + "tool_mask/" + direct["t4"] + ".png"
        tool_t4 = cv2.imread(tool_t4, -1)
        tool_t4 = tool_t4[:512, 226:-2]

        tool_t0_r = path + "tool_mask_right/" + direct["t0"] + ".png"
        tool_t0_r = cv2.imread(tool_t0_r, -1)
        tool_t1_r = path + "tool_mask_right/" + direct["t1"] + ".png"
        tool_t1_r = cv2.imread(tool_t1_r, -1)
        tool_t2_r = path + "tool_mask_right/" + direct["t2"] + ".png"
        tool_t2_r = cv2.imread(tool_t2_r, -1)
        tool_t3_r = path + "tool_mask_right/" + direct["t3"] + ".png"
        tool_t3_r = cv2.imread(tool_t3_r, -1)
        tool_t4_r = path + "tool_mask_right/" + direct["t4"] + ".png"
        tool_t4_r = cv2.imread(tool_t4_r, -1)



        res = {}

        for i in ["left_t0", "left_t1", "left_t2", "left_t3", "left_t4"]:
            val = vars()[i]
            val = val[:512, 226:-2, :]
            val = th.from_numpy(val.astype(np.float32)).permute(2, 0, 1) / 255.
            res[i] = val

        for i in ["right_t0", "right_t1", "right_t2", "right_t3", "right_t4"]:
            val = vars()[i]
            val = th.from_numpy(val.astype(np.float32)).permute(2, 0, 1) / 255.
            res[i] = val

        for i in ["disp_t0", "disp_t1", "disp_t2", "disp_t3", "disp_t4"]:
            val = vars()[i]
            val = val[:512, 226:-2]
            val = th.from_numpy(val.astype(np.float32)).unsqueeze(0)
            res[i] = val

        for i in ["tool_t0", "tool_t1", "tool_t2", "tool_t3", "tool_t4",
                  "tool_t0_r", "tool_t1_r", "tool_t2_r", "tool_t3_r", "tool_t4_r"]:
            val = vars()[i]
            val = th.from_numpy(val.astype(np.float32)).unsqueeze(0)
            res[i] = val


        return res

class Surgical_dataset_eval(Dataset):
    def __init__(self, data_dir):
        with open(data_dir, "rb") as f:
            self.img_list = pickle.load(f)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        direct = self.img_list[idx]
        path = direct["path"]
        left = []
        disp = []
        tool = []
        for i in direct['sequence']:
            left_t0 = path + "img_left/" + i + ".jpg"
            left_t0 = np.array(Image.open(left_t0))
            left_t0 = left_t0[:512, 226:-2, :]
            left_t0 = th.from_numpy(left_t0.astype(np.float32)).permute(2, 0, 1) / 255.
            left.append(left_t0)

            disp_t0 = path + "disp/" + i + ".npy"
            disp_t0 = np.load(disp_t0)
            disp_t0 = disp_t0[:512, 226:-2]
            disp_t0 = th.from_numpy(disp_t0.astype(np.float32)).unsqueeze(0)
            disp.append(disp_t0)

            tool_t0 = path + "tool_mask/" + i + ".png"
            tool_t0 = cv2.imread(tool_t0, -1)
            tool_t0 = tool_t0[:512, 226:-2]
            tool_t0 = th.from_numpy(tool_t0.astype(np.float32)).unsqueeze(0)
            tool.append(tool_t0)

        res = {}
        res["left"] = left
        res["disp"] = disp
        res["tool"] = tool

        return res