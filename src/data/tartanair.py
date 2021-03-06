"""
    TartanAir Depth Completion Dataset Helper
"""


import os
import numpy as np
import json
import random
from . import BaseDataset
import cv2

from PIL import Image
import torch
import torchvision.transforms.functional as TF
from scipy.cluster.vq import kmeans2


import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from PIL import Image

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from math import *

import math


def calc_normals(depth):

    n = 5
    depth = cv2.GaussianBlur(depth,(n,n),0)

    zy, zx = np.gradient(depth)  
    normal = np.dstack((-zx, -zy, np.ones_like(depth)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    return normal

def cart2sph_vec(xyz):
    
    H,W,C = xyz.shape
    xyz = xyz.reshape(H*W,C)

    #takes list xyz (single coord)
    x       = xyz[:,0]
    y       = xyz[:,1]
    z       = xyz[:,2]
    r       =  (x**2 + y**2+z**2)**0.5#np.linalg.norm(xyz, axis=-1)

    theta   =  np.arccos(z/r)*180.0/ pi #to degrees
    phi     =  np.arctan2(y,x)*180.0/ pi
    
    #print(theta, phi)
    #theta = roundup(theta, 10.0)
    #phi = roundup(phi, 40.0)
    
    out = np.stack([theta,phi])
    out = np.transpose(out,(1,0))
    out = out.reshape(H,W,2)
    
    n = 5
    out[:,:,0] = cv2.GaussianBlur(out[:,:,0],(n,n),0)
    out[:,:,1] = cv2.GaussianBlur(out[:,:,1],(n,n),0)

    return out

def cart2cmap_vec(xyz, cmap):
    """
        converts xyz coordinates to spherical coordinates and then displays in a specified colormap 
    """
    
    rthetaphi = cart2sph_vec(xyz)
    H,W,C = rthetaphi.shape
    rthetaphi = rthetaphi.reshape(H*W,C)
    
    phi = rthetaphi[:, 0].astype(int)
    theta = (rthetaphi[:, 1] + 179.0).astype(int)

    rgb = cmap[phi, theta]
    rgb = rgb.reshape(H,W,3)

    return rgb

def _calc_norm_masks(depth):

    normal = calc_normals(depth)
    
    """
    normal_spherical = np.zeros_like(normal)
    for x in range(normal.shape[0]):
        for y in range(normal.shape[1]):
            normal_spherical[x,y,:] = cart2sph(normal[x,y,:])
    print(time.time() - t1)
    """

    cmap = create_custom_colormap()
    normal_spherical = cart2cmap_vec(normal, cmap)
    
    H,W,C = normal_spherical.shape
    normal_spherical = normal_spherical.reshape(H*W,C)
    try:
        centroid, label = kmeans2(normal_spherical, 15, minit='random')
    except:
        label = np.zeros((H,W), dtype=np.int32)

    seg = label.reshape(H,W)

    return seg

def create_custom_colormap():
    """
    Creates a 2D colormap similar to 
    """

    cmap = np.zeros((180,360,3))

    x, y = np.mgrid[0:180:1, 0:360:1]
    pos = np.dstack((x, y))
    rv = multivariate_normal([0, 0], 10000* np.asarray([[2.0, 0.], [0., 0.5]])).pdf(pos)
    rv += multivariate_normal([0, 360], 10000* np.asarray([[2.0, -0.], [-0., 0.50]])).pdf(pos)
    cmap[:,:,2] = rv / np.max(rv)

    rv = multivariate_normal([0, 120], 10000* np.asarray([[2.5, 0.], [0., 0.5]])).pdf(pos)
    cmap[:,:,1] = rv / np.max(rv)

    rv = multivariate_normal([180, 120], 10000* np.asarray([[0.5, 0.], [0., 40]])).pdf(pos)
    cmap[:,:,0] = rv / np.max(rv)

    return cmap



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


def read_depth(file_name):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)
    image_depth = np.load(file_name)

    # Consider empty depth
    #assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
    #    "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    #image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth

def read_sparse_depth(file_name, size):
    # loads depth map D from 16 bits png file as a numpy array,
    # refer to readme file in KITTI dataset
    assert os.path.exists(file_name), "file not found: {}".format(file_name)

    image_depth = np.zeros(size, dtype=np.float32)
    image_confidence = np.zeros(size, dtype=np.float32)
    #print(file_name, os.stat(file_name).st_size)
    # check if file is empty
    if os.stat(file_name).st_size < 5:
        return image_depth, image_confidence
    
    file_depth = np.loadtxt(file_name, delimiter=',')
    file_depth = file_depth.reshape(-1,4) # [features, params]
    
    for (x,y,z,sigma) in file_depth:
        #print(x,y,z,sigma,size)
        image_depth[int(y), int(x)] = z
        image_confidence[int(y), int(x)] = sigma
    # Consider empty depth
    #assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
    #    "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    #image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth, image_confidence


# Reference : https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


class TARTANAIR(BaseDataset):
    def __init__(self, args, mode):
        super(TARTANAIR, self).__init__(args, mode)

        self.args = args
        self.mode = mode
        self.dep_src = args.dep_src
        self.constrain_sgbm = args.constrain_sgbm
        self.input_conf = args.input_conf

        if mode != 'train' and mode != 'val' and mode != 'test':
            raise NotImplementedError

        self.height = args.patch_height
        self.width = args.patch_width

        self.augment = self.args.augment

        with open(self.args.split_json) as json_file:
            json_data = json.load(json_file)
            self.sample_list = json_data[mode]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        rgb, seg, depth_features, confidence_features, depth_sgbm, confidence_sgbm, gt, K = self._load_data(idx)

        if self.augment and self.mode == 'train':
            # Top crop if needed
            if self.args.top_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.args.top_crop, 0,
                              height - self.args.top_crop, width)
                seg = TF.crop(seg, self.args.top_crop, 0,
                              height - self.args.top_crop, width)
                if depth_features is not None:
                    depth_features = TF.crop(depth_features, self.args.top_crop, 0,
                                    height - self.args.top_crop, width)
                    confidence_features = TF.crop(confidence_features, self.args.top_crop, 0,
                                    height - self.args.top_crop, width)
                if depth_sgbm is not None:
                    depth_sgbm = TF.crop(depth_sgbm, self.args.top_crop, 0,
                                    height - self.args.top_crop, width)
                    confidence_sgbm = TF.crop(confidence_sgbm, self.args.top_crop, 0,
                                    height - self.args.top_crop, width)
                gt = TF.crop(gt, self.args.top_crop, 0,
                             height - self.args.top_crop, width)
                K[3] = K[3] - self.args.top_crop

            # Left crop if needed
            if self.args.left_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, 0, self.args.left_crop, height, width - self.args.left_crop)
                seg = TF.crop(seg, 0, self.args.left_crop, height, width - self.args.left_crop)
                if depth_features is not None:
                    depth_features = TF.crop(depth_features, 0, self.args.left_crop, height, width - self.args.left_crop)
                    confidence_features = TF.crop(confidence_features, 0, self.args.left_crop, height, width - self.args.left_crop)
                if depth_sgbm is not None:
                    depth_sgbm = TF.crop(depth_sgbm, 0, self.args.left_crop, height, width - self.args.left_crop)
                    confidence_sgbm = TF.crop(confidence_sgbm, 0, self.args.left_crop, height, width - self.args.left_crop)
                gt = TF.crop(gt, 0, self.args.left_crop, height, width - self.args.left_crop)
                K[2] = K[2] - self.args.left_crop

            width, height = rgb.size

            _scale = np.random.uniform(1.0, 1.5)
            scale = np.int(height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            # Horizontal flip
            if flip > 0.5:
                rgb = TF.hflip(rgb)
                seg = TF.hflip(seg)
                if depth_features is not None:
                    depth_features = TF.hflip(depth_features)
                    confidence_features = TF.hflip(confidence_features)
                if depth_sgbm is not None:
                    depth_sgbm = TF.hflip(depth_sgbm)
                    confidence_sgbm = TF.hflip(confidence_sgbm)
                gt = TF.hflip(gt)
                K[2] = width - K[2]

            # Rotation
            rgb = TF.rotate(rgb, angle=degree, resample=Image.BICUBIC)
            seg = TF.rotate(seg, angle=degree, resample=Image.NEAREST)
            if depth_features is not None:
                depth_features = TF.rotate(depth_features, angle=degree, resample=Image.NEAREST)
                confidence_features = TF.rotate(confidence_features, angle=degree, resample=Image.NEAREST)
            if depth_sgbm is not None:
                depth_sgbm = TF.rotate(depth_sgbm, angle=degree, resample=Image.NEAREST)
                confidence_sgbm = TF.rotate(confidence_sgbm, angle=degree, resample=Image.NEAREST)
            gt = TF.rotate(gt, angle=degree, resample=Image.NEAREST)

            # Color jitter
            brightness = np.random.uniform(0.6, 1.4)
            contrast = np.random.uniform(0.6, 1.4)
            saturation = np.random.uniform(0.6, 1.4)

            rgb = TF.adjust_brightness(rgb, brightness)
            rgb = TF.adjust_contrast(rgb, contrast)
            rgb = TF.adjust_saturation(rgb, saturation)

            # Resize
            rgb = TF.resize(rgb, scale, Image.BICUBIC)
            seg = TF.resize(seg, scale, Image.NEAREST)
            if depth_features is not None:
                depth_features = TF.resize(depth_features, scale, Image.NEAREST)
                confidence_features = TF.resize(confidence_features, scale, Image.NEAREST)
            if depth_sgbm is not None:
                depth_sgbm = TF.resize(depth_sgbm, scale, Image.NEAREST)
                confidence_sgbm = TF.resize(confidence_sgbm, scale, Image.NEAREST)
            gt = TF.resize(gt, scale, Image.NEAREST)

            K[0] = K[0] * _scale
            K[1] = K[1] * _scale
            K[2] = K[2] * _scale
            K[3] = K[3] * _scale

            # Crop
            width, height = rgb.size

            assert self.height <= height and self.width <= width, \
                "patch size is larger than the input size"

            h_start = random.randint(0, height - self.height)
            w_start = random.randint(0, width - self.width)

            rgb = TF.crop(rgb, h_start, w_start, self.height, self.width)
            seg = TF.crop(seg, h_start, w_start, self.height, self.width)
            if depth_features is not None:
                depth_features = TF.crop(depth_features, h_start, w_start, self.height, self.width)
                confidence_features = TF.crop(confidence_features, h_start, w_start, self.height, self.width)
            if depth_sgbm is not None:
                depth_sgbm = TF.crop(depth_sgbm, h_start, w_start, self.height, self.width)
                confidence_sgbm = TF.crop(confidence_sgbm, h_start, w_start, self.height, self.width)
            gt = TF.crop(gt, h_start, w_start, self.height, self.width)

            K[2] = K[2] - w_start
            K[3] = K[3] - h_start

            rgb = TF.to_tensor(rgb)
            seg = TF.to_tensor(seg)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225), inplace=True)

            if depth_features is not None:
                depth_features = TF.to_tensor(np.array(depth_features))
                depth_features = depth_features / _scale
                confidence_features = TF.to_tensor(np.array(confidence_features))
                confidence_features = confidence_features / _scale
            
            if depth_sgbm is not None:
                depth_sgbm = TF.to_tensor(np.array(depth_sgbm))
                depth_sgbm = depth_sgbm / _scale
                confidence_sgbm = TF.to_tensor(np.array(confidence_sgbm))
                confidence_sgbm = confidence_sgbm / _scale

            gt = TF.to_tensor(np.array(gt))
            gt = gt / _scale

        elif self.mode in ['train', 'val']:
            # Top crop if needed
            if self.args.top_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.args.top_crop, 0,
                              height - self.args.top_crop, width)
                seg = TF.crop(seg, self.args.top_crop, 0,
                              height - self.args.top_crop, width)
                if depth_features is not None:
                    depth_features = TF.crop(depth_features, self.args.top_crop, 0,
                                    height - self.args.top_crop, width)
                    confidence_features = TF.crop(confidence_features, self.args.top_crop, 0,
                                    height - self.args.top_crop, width)
                if depth_sgbm is not None:
                    depth_sgbm = TF.crop(depth_sgbm, self.args.top_crop, 0,
                                    height - self.args.top_crop, width)
                    confidence_sgbm = TF.crop(confidence_sgbm, self.args.top_crop, 0,
                                    height - self.args.top_crop, width)
                gt = TF.crop(gt, self.args.top_crop, 0,
                             height - self.args.top_crop, width)
                K[3] = K[3] - self.args.top_crop

            # Left crop if needed
            if self.args.left_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, 0, self.args.left_crop, height, width - self.args.left_crop)
                seg = TF.crop(seg, 0, self.args.left_crop, height, width - self.args.left_crop)
                if depth_features is not None:
                    depth_features = TF.crop(depth_features, 0, self.args.left_crop, height, width - self.args.left_crop)
                    confidence_features = TF.crop(confidence_features, 0, self.args.left_crop, height, width - self.args.left_crop)
                if depth_sgbm is not None:
                    depth_sgbm = TF.crop(depth_sgbm, 0, self.args.left_crop, height, width - self.args.left_crop)
                    confidence_sgbm = TF.crop(confidence_sgbm, 0, self.args.left_crop, height, width - self.args.left_crop)
                gt = TF.crop(gt, 0, self.args.left_crop, height, width - self.args.left_crop)
                K[2] = K[2] - self.args.left_crop

            # Crop
            width, height = rgb.size

            assert self.height <= height and self.width <= width, \
                "patch size is larger than the input size"

            h_start = random.randint(0, height - self.height)
            w_start = random.randint(0, width - self.width)

            rgb = TF.crop(rgb, h_start, w_start, self.height, self.width)
            seg = TF.crop(seg, h_start, w_start, self.height, self.width)
            if depth_features is not None:
                depth_features = TF.crop(depth_features, h_start, w_start, self.height, self.width)
                confidence_features = TF.crop(confidence_features, h_start, w_start, self.height, self.width)
            if depth_sgbm is not None:
                depth_sgbm = TF.crop(depth_sgbm, h_start, w_start, self.height, self.width)
                confidence_sgbm = TF.crop(confidence_sgbm, h_start, w_start, self.height, self.width)
            gt = TF.crop(gt, h_start, w_start, self.height, self.width)

            K[2] = K[2] - w_start
            K[3] = K[3] - h_start

            rgb = TF.to_tensor(rgb)
            seg = TF.to_tensor(seg)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225), inplace=True)

            if depth_features is not None:
                depth_features = TF.to_tensor(np.array(depth_features))
                confidence_features = TF.to_tensor(np.array(confidence_features))
            if depth_sgbm is not None:
                depth_sgbm = TF.to_tensor(np.array(depth_sgbm))
                confidence_sgbm = TF.to_tensor(np.array(confidence_sgbm))

            gt = TF.to_tensor(np.array(gt))

        else: # test
            if self.args.top_crop > 0 and self.args.test_crop:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.args.top_crop, 0,
                              height - self.args.top_crop, width)
                seg = TF.crop(seg, self.args.top_crop, 0,
                              height - self.args.top_crop, width)
                if depth_features is not None:
                    depth_features = TF.crop(depth_features, self.args.top_crop, 0,
                                    height - self.args.top_crop, width)
                    confidence_features = TF.crop(confidence_features, self.args.top_crop, 0,
                                    height - self.args.top_crop, width)
                if depth_sgbm is not None:
                    depth_sgbm = TF.crop(depth_sgbm, self.args.top_crop, 0,
                                    height - self.args.top_crop, width)
                    confidence_sgbm = TF.crop(confidence_sgbm, self.args.top_crop, 0,
                                    height - self.args.top_crop, width)
                gt = TF.crop(gt, self.args.top_crop, 0,
                             height - self.args.top_crop, width)
                K[3] = K[3] - self.args.top_crop

            # Left crop if needed
            if self.args.left_crop > 0 and self.args.test_crop:
                width, height = rgb.size
                rgb = TF.crop(rgb, 0, self.args.left_crop, height, width - self.args.left_crop)
                seg = TF.crop(seg, 0, self.args.left_crop, height, width - self.args.left_crop)
                if depth_features is not None:
                    depth_features = TF.crop(depth_features, 0, self.args.left_crop, height, width - self.args.left_crop)
                    confidence_features = TF.crop(confidence_features, 0, self.args.left_crop, height, width - self.args.left_crop)
                if depth_sgbm is not None:
                    depth_sgbm = TF.crop(depth_sgbm, 0, self.args.left_crop, height, width - self.args.left_crop)
                    confidence_sgbm = TF.crop(confidence_sgbm, 0, self.args.left_crop, height, width - self.args.left_crop)
                gt = TF.crop(gt, 0, self.args.left_crop, height, width - self.args.left_crop)
                K[2] = K[2] - self.args.left_crop

            width, height = rgb.size

            rgb = TF.to_tensor(rgb)
            seg = TF.to_tensor(seg)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225), inplace=True)

            if depth_features is not None:
                depth_features = TF.to_tensor(np.array(depth_features))
                confidence_features = TF.to_tensor(np.array(confidence_features))
            if depth_sgbm is not None:
                depth_sgbm = TF.to_tensor(np.array(depth_sgbm))
                confidence_sgbm = TF.to_tensor(np.array(confidence_sgbm))

            gt = TF.to_tensor(np.array(gt))

        if self.args.num_sample > 0:
            depth_features, confidence_features = self.get_sparse_depth(depth_features, confidence_features, self.args.num_sample)
            depth_sgbm, confidence_sgbm = self.get_sparse_depth(depth_sgbm, confidence_sgbm, self.args.num_sample)

        if self.args.dep_src == 'slam':
            output = {'rgb': rgb, 'seg': seg, 'dep': depth_features, 'confidence': confidence_features, 'gt': gt, 'K': torch.Tensor(K)}
        elif self.args.dep_src == 'sgbm':
            output = {'rgb': rgb, 'seg': seg, 'dep': depth_sgbm, 'confidence': confidence_sgbm,  'gt': gt, 'K': torch.Tensor(K)}
        elif self.args.dep_src == 'slam+sgbm' or self.args.dep_src == 'sgbm+slam':
            output = {'rgb': rgb, 'seg': seg, 'dep0': depth_features, 'dep1': depth_sgbm, 'confidence0': confidence_features,
             'confidence1': confidence_sgbm, 'gt': gt, 'K': torch.Tensor(K)}
        else:
            raise NotImplementedError

        return output

    def _load_data(self, idx):


        path_gt = os.path.join(self.args.dir_data,
                               self.sample_list[idx]['gt'])

        gt = read_depth(path_gt)
        #seg = _calc_norm_masks(gt)
        gt = Image.fromarray(gt.astype('float32'), mode='F')

        path_rgb = os.path.join(self.args.dir_data,
                                self.sample_list[idx]['rgb'])

        path_seg = os.path.join(self.args.dir_data,
                                self.sample_list[idx]['seg'])

        if 'slam' in self.dep_src: 
            path_depth_features = os.path.join(self.args.dir_data,
                                    self.sample_list[idx]['depth_features'])
            
            depth_features, confidence_features = read_sparse_depth(path_depth_features, (self.height, self.width))

            depth_features = Image.fromarray(depth_features.astype('float32'), mode='F')
            confidence_features = Image.fromarray(confidence_features.astype('float32'), mode='F')

        if 'sgbm' in self.dep_src:
            path_depth_sgbm = os.path.join(self.args.dir_data,
                                    self.sample_list[idx]['depth_sgbm'])
            path_confidence_sgbm = os.path.join(self.args.dir_data,
                                    self.sample_list[idx]['confidence_sgbm'])

            depth_sgbm = read_depth(path_depth_sgbm)
            confidence_sgbm = read_depth(path_confidence_sgbm)

            confidence_sgbm[depth_sgbm > 10] = 100
            confidence_sgbm[depth_sgbm < 0.4] = 100
            depth_sgbm[depth_sgbm > 10] = 0
            depth_sgbm[depth_sgbm < 0.4] = 0

            if self.constrain_sgbm:
                depth_sgbm[abs(depth_sgbm - gt) > 1] = 0

            if self.input_conf == 'binary':
                confidence_sgbm = np.zeros_like(confidence_sgbm)
                confidence_sgbm[depth_sgbm > 0] = 1

            depth_sgbm = Image.fromarray(depth_sgbm.astype('float32'), mode='F')
            confidence_sgbm = Image.fromarray(confidence_sgbm.astype('float32'), mode='F')

        rgb = Image.open(path_rgb)
        seg = np.load(path_seg.replace("seg0", "new_seg0"))
        seg = Image.fromarray(seg)

        w1, h1 = rgb.size
        if w1 == 640 and h1 == 480:
            f, cx, cy = 0.25, 320, 240
            K = [f, f, cx, cy]
        elif w1 == 320 and h1 == 240:
            f, cx, cy = 0.125, 160, 120
            K = [f, f, cx, cy]
        else:
            raise NotImplementedError 

        if 'slam' in self.dep_src and 'sgbm' in self.dep_src:

            w1, h1 = rgb.size
            w2, h2 = depth_features.size
            w3, h3 = gt.size
            w4, h4 = depth_sgbm.size

            assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3 and w1 == w4 and h1 == h4

            return rgb, seg, depth_features, confidence_features, depth_sgbm, confidence_sgbm, gt, K

        elif 'slam' == self.dep_src:

            w1, h1 = rgb.size
            w2, h2 = gt.size
            w3, h3 = depth_features.size

            assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3

            return rgb, seg, depth_features, confidence_features, None, None, gt, K

        elif 'sgbm' == self.dep_src:

            w1, h1 = rgb.size
            w2, h2 = gt.size
            w3, h3 = depth_sgbm.size

            assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3

            return rgb, seg, None, None, depth_sgbm, confidence_sgbm, gt, K

    def get_sparse_depth(self, dep, num_sample):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_sample]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        return dep_sp
