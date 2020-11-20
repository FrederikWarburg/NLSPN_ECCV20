"""
    TartanAir Depth Completion Dataset Helper
"""


import os
import numpy as np
import json
import random
from . import BaseDataset

from PIL import Image
import torch
import torchvision.transforms.functional as TF


"""
TartanAir Depth Completion json file has a following format:

{
    "train": [
        {
            "rgb": "train/P000/cam0/data/000562.png",
            "depth": "train/P000/depth_sparse0/data/000562.npy",
            "gt": "train/P000/ground_truth/depth0/data/000562.npy",
            #####"K": "train/2011_09_30_drive_0018_sync/calib_cam_to_cam.txt"
        }, ...
    ],
    "val": [
        {
            "rgb": "train/P000/cam0/data/000562.png",
            "depth": "train/P000/depth_sparse0/data/000562.npy",
            "gt": "train/P000/ground_truth/depth0/data/000562.npy",
            #####"K": "train/2011_09_30_drive_0018_sync/calib_cam_to_cam.txt"
        }, ...
    ],
    "test": [
        {
            "rgb": "train/P000/cam0/data/000562.png",
            "depth": "train/P000/depth_sparse0/data/000562.npy",
            "gt": "train/P000/ground_truth/depth0/data/000562.npy",
            #####"K": "train/2011_09_30_drive_0018_sync/calib_cam_to_cam.txt"
        }, ...
    ]
}

Reference : https://github.com/XinJCheng/CSPN/blob/master/nyu_dataset_loader.py
"""


def backproject_project(depth_im):
    
    T_sc_depth = np.asarray([[0.99998779, -0.00044588, 0.00492056, -0.03102],
                [0.00046182, 0.99999465, -0.00323781, 0.00563841],
                [-0.00491909, 0.00324004, 0.99998265, 0.01847428],
                [0.0, 0.0, 0.0, 1.0 ]])
    T_sc_visible = np.asarray([[0.99999116, -0.00059866, 0.00416182, 0.02802532],
                [0.00061067, 0.99999565, -0.00288555, 0.00568833],
                [-0.00416007, 0.00288807, 0.99998718, 0.0181785],
                [0.0, 0.0, 0.0, 1.0 ]])

    T_sc_depth_inverse = np.linalg.inv(T_sc_depth)
    
    im_proj = np.zeros_like(depth_im)
    for i in range(depth_im.shape[0]):
        for j in range(depth_im.shape[1]):
            z = depth_im[i,j]
            if z == 0: continue
            q1 = np.asarray([[i,j,z,1]]).T
            Q = np.matmul(T_sc_depth_inverse,q1)
            Q /= Q[-1]
            q2 = np.matmul(T_sc_visible, Q)
            q2 /= q2[-1]
            im_proj[i,j] = q2[2]
            
    return im_proj

def read_depth(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.array(Image.open(file_name))

    # Consider empty depth
    assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
        "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth

class REALSENSE(BaseDataset):
    def __init__(self, args, mode):
        super(REALSENSE, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.input_conf = 'binary'

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        self.height = args.patch_height
        self.width = args.patch_width

        with open(self.args.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        rgb, depth, confidence, gt, K = self._load_data(idx)

        rgb = TF.to_tensor(rgb)
        rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225), inplace=True)
        depth = TF.to_tensor(np.array(depth))
        confidence = TF.to_tensor(np.array(confidence))
        gt = TF.to_tensor(np.array(gt))

        output = {'rgb': rgb, 'dep': depth, 'confidence': confidence, 'gt': gt, 'K': torch.Tensor(K)}

        return output

    def _load_data(self, idx):

        #TODO: for now we just use input as GT
        path_gt = os.path.join(self.args.dir_data,
                               self.sample_list[idx]['depth'])

        gt = read_depth(path_gt)
        gt = backproject_project(gt)
        gt = Image.fromarray(gt.astype('float32'), mode='F')

        path_rgb = os.path.join(self.args.dir_data,
                                self.sample_list[idx]['rgb'])

        path_depth = os.path.join(self.args.dir_data,
                                self.sample_list[idx]['depth'])

        depth = read_depth(path_depth)
        depth = backproject_project(path_depth)

        if self.input_conf == 'binary':
            confidence = np.zeros_like(depth)
            confidence[depth > 0] = 1

        depth = Image.fromarray(depth.astype('float32'), mode='F')
        confidence = Image.fromarray(confidence.astype('float32'), mode='F')

        rgb = Image.open(path_rgb)

        w1, h1 = rgb.size
        
        #TODO: change to correct camera paramters 
        f, cx, cy = 0.25, 320, 240
        K = [f, f, cx, cy]

        w1, h1 = rgb.size
        w2, h2 = gt.size
        w3, h3 = depth.size

        assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3
        return rgb, depth, confidence, gt, K
