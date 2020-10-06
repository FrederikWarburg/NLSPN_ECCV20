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

    # check if file is empty
    if os.stat(file_name).st_size == 0:
        return image_depth

    file_depth = np.loadtxt(file_name, delimiter=',')
    file_depth = file_depth.reshape(-1,4) # [features, params]
    
    for (x,y,z,sigma) in file_depth:
        image_depth[int(y), int(x)] = z

    # Consider empty depth
    #assert (np.max(image_depth) == 0) or (np.max(image_depth) > 255), \
    #    "np.max(depth_png)={}, path={}".format(np.max(image_depth), file_name)

    #image_depth = image_depth.astype(np.float32) / 256.0
    return image_depth


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
        rgb, depth_features, depth_sgbm, gt, K = self._load_data(idx)

        if self.augment and self.mode == 'train':
            # Top crop if needed
            if self.args.top_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.args.top_crop, 0,
                              height - self.args.top_crop, width)
                depth_features = TF.crop(depth_features, self.args.top_crop, 0,
                                height - self.args.top_crop, width)
                depth_sgbm = TF.crop(depth_sgbm, self.args.top_crop, 0,
                                height - self.args.top_crop, width)
                gt = TF.crop(gt, self.args.top_crop, 0,
                             height - self.args.top_crop, width)
                K[3] = K[3] - self.args.top_crop

            width, height = rgb.size

            _scale = np.random.uniform(1.0, 1.5)
            scale = np.int(height * _scale)
            degree = np.random.uniform(-5.0, 5.0)
            flip = np.random.uniform(0.0, 1.0)

            # Horizontal flip
            if flip > 0.5:
                rgb = TF.hflip(rgb)
                depth_features = TF.hflip(depth_features)
                depth_sgbm = TF.hflip(depth_sgbm)
                gt = TF.hflip(gt)
                K[2] = width - K[2]

            # Rotation
            rgb = TF.rotate(rgb, angle=degree, resample=Image.BICUBIC)
            depth_features = TF.rotate(depth_features, angle=degree, resample=Image.NEAREST)
            depth_sgbm = TF.rotate(depth_sgbm, angle=degree, resample=Image.NEAREST)
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
            depth_features = TF.resize(depth_features, scale, Image.NEAREST)
            depth_sgbm = TF.resize(depth_sgbm, scale, Image.NEAREST)
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
            depth_features = TF.crop(depth_features, h_start, w_start, self.height, self.width)
            depth_sgbm = TF.crop(depth_sgbm, h_start, w_start, self.height, self.width)
            gt = TF.crop(gt, h_start, w_start, self.height, self.width)

            K[2] = K[2] - w_start
            K[3] = K[3] - h_start

            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225), inplace=True)

            depth_features = TF.to_tensor(np.array(depth_features))
            depth_sgbm = TF.to_tensor(np.array(depth_sgbm))
            depth_features = depth_features / _scale
            depth_sgbm = depth_sgbm / _scale

            gt = TF.to_tensor(np.array(gt))
            gt = gt / _scale

        elif self.mode in ['train', 'val']:
            # Top crop if needed
            if self.args.top_crop > 0:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.args.top_crop, 0,
                              height - self.args.top_crop, width)
                depth_features = TF.crop(depth_features, self.args.top_crop, 0,
                                height - self.args.top_crop, width)
                depth_sgbm = TF.crop(depth_sgbm, self.args.top_crop, 0,
                                height - self.args.top_crop, width)
                gt = TF.crop(gt, self.args.top_crop, 0,
                             height - self.args.top_crop, width)
                K[3] = K[3] - self.args.top_crop

            # Crop
            width, height = rgb.size

            assert self.height <= height and self.width <= width, \
                "patch size is larger than the input size"

            h_start = random.randint(0, height - self.height)
            w_start = random.randint(0, width - self.width)

            rgb = TF.crop(rgb, h_start, w_start, self.height, self.width)
            depth_features = TF.crop(depth_features, h_start, w_start, self.height, self.width)
            depth_sgbm = TF.crop(depth_sgbm, h_start, w_start, self.height, self.width)
            gt = TF.crop(gt, h_start, w_start, self.height, self.width)

            K[2] = K[2] - w_start
            K[3] = K[3] - h_start

            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225), inplace=True)

            depth_features = TF.to_tensor(np.array(depth_features))
            depth_sgbm = TF.to_tensor(np.array(depth_sgbm))

            gt = TF.to_tensor(np.array(gt))

        else: # test
            if self.args.top_crop > 0 and self.args.test_crop:
                width, height = rgb.size
                rgb = TF.crop(rgb, self.args.top_crop, 0,
                              height - self.args.top_crop, width)
                depth_features = TF.crop(depth_features, self.args.top_crop, 0,
                                height - self.args.top_crop, width)
                depth_sgbm = TF.crop(depth_sgbm, self.args.top_crop, 0,
                                height - self.args.top_crop, width)
                gt = TF.crop(gt, self.args.top_crop, 0,
                             height - self.args.top_crop, width)
                K[3] = K[3] - self.args.top_crop

            rgb = TF.to_tensor(rgb)
            rgb = TF.normalize(rgb, (0.485, 0.456, 0.406),
                               (0.229, 0.224, 0.225), inplace=True)

            depth_features = TF.to_tensor(np.array(depth_features))
            depth_sgbm = TF.to_tensor(np.array(depth_sgbm))

            gt = TF.to_tensor(np.array(gt))

        if self.args.num_sample > 0:
            depth_features = self.get_sparse_depth(depth_features, self.args.num_sample)
            depth_sgbm = self.get_sparse_depth(depth_sgbm, self.args.num_sample)

        if self.args.dep_src == 'slam':
            output = {'rgb': rgb, 'dep': depth_features, 'gt': gt, 'K': torch.Tensor(K)}
        elif self.args.dep_src == 'sgbm':
            output = {'rgb': rgb, 'dep': depth_sgbm, 'gt': gt, 'K': torch.Tensor(K)}
        elif self.args.dep_src == 'slam+sgbm' or self.args.dep_src == 'sgbm+slam':
            output = {'rgb': rgb, 'dep': (depth_features, depth_sgbm), 'gt': gt, 'K': torch.Tensor(K)}

        return output

    def _load_data(self, idx):
        path_rgb = os.path.join(self.args.dir_data,
                                self.sample_list[idx]['rgb'])
        path_depth_features = os.path.join(self.args.dir_data,
                                  self.sample_list[idx]['depth_features'])
        path_depth_sgbm = os.path.join(self.args.dir_data,
                                  self.sample_list[idx]['depth_sgbm'])
        path_gt = os.path.join(self.args.dir_data,
                               self.sample_list[idx]['gt'])
        #path_calib = os.path.join(self.args.dir_data,
        #                          self.sample_list[idx]['K'])

        depth_features = read_sparse_depth(path_depth_features, (self.height, self.width))
        depth_sgbm = read_depth(path_depth_sgbm)

        gt = read_depth(path_gt)

        rgb = Image.open(path_rgb)
        depth_features = Image.fromarray(depth_features.astype('float32'), mode='F')
        depth_sgbm = Image.fromarray(depth_sgbm.astype('float32'), mode='F')
        gt = Image.fromarray(gt.astype('float32'), mode='F')

        #calib = read_calib_file()# path_calib)
        f, cx, cy = 0.25, 320, 240
        K = [f, f, cx, cy]

        w1, h1 = rgb.size
        w2, h2 = depth_features.size
        w3, h3 = gt.size
        w4, h4 = depth_sgbm.size

        assert w1 == w2 and w1 == w3 and h1 == h2 and h1 == h3 and w1 == w4 and h1 == h4

        return rgb, depth_features, depth_sgbm, gt, K

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
