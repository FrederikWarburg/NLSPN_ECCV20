import os
import random

import argparse
import numpy as np
import shutil
from PIL import Image
import cv2

def resize_depth(im, scale = 4, GT = True):

    im = np.asarray(im).astype(np.float32)
    
    h,w = im.shape
    hb, wb = int(np.ceil(h / scale)), int(np.ceil(w / scale))

    newim = np.zeros((hb, wb))
    for i in range(hb):
        for j in range(wb):
            area = im[(i*scale):((i+1)*scale),(j*scale):((j+1)*scale)]
            area = area[area > 0]
            #print(area, np.mean(area))
            if GT:
                if len(area):
                    newim[i,j] = np.mean(area)
            else:
                if len(area) > 0 and i % 2 == 0 and j % 2 == 0:
                    newim[i,j] = np.mean(area) #area[0] #np.mean(area)
    
    newim = newim.astype(np.uint16)
    
    return newim


def resize_folder(basepath, newpath, scale):

    shutil.copy(basepath + '/calib_cam_to_cam.txt', newpath + '/calib_cam_to_cam.txt')
    shutil.copy(basepath + '/calib_imu_to_velo.txt', newpath + '/calib_imu_to_velo.txt')
    shutil.copy(basepath + '/calib_velo_to_cam.txt', newpath + '/calib_velo_to_cam.txt')
    shutil.copytree(basepath + '/oxts', newpath + '/oxts')

    for p in ['image_02', 'image_03']:
        if not os.path.exists(newpath  + '/' + p + '/data/'): os.makedirs(newpath  + '/' + p + '/data/')
        for i in os.listdir(basepath + '/' +  p +'/data'):
            im = Image.open(basepath  + '/' + p + '/data/' + i)
            im.thumbnail((93, 306), Image.ANTIALIAS)
            im.save(newpath  + '/' + p + '/data/' + i)

    for p in ['image_02', 'image_03']:
        if not os.path.exists(newpath + '/proj_depth/groundtruth/'+ p): 
            os.makedirs(newpath + '/proj_depth/groundtruth/'+p)
        for i in os.listdir(basepath + '/proj_depth/groundtruth/' + p):
            im = Image.open(basepath + '/proj_depth/groundtruth/'+p+'/' + i)
            im = resize_depth(im, 4, True)
            cv2.imwrite(newpath + '/proj_depth/groundtruth/'+p+'/' + i, im)

    for p in ['image_02', 'image_03']:
        if not os.path.exists(newpath + '/proj_depth/velodyne_raw/' + p): 
            os.makedirs(newpath + '/proj_depth/velodyne_raw/' + p)
        for i in os.listdir(basepath + '/proj_depth/velodyne_raw/' + p):
            im = Image.open(basepath + '/proj_depth/velodyne_raw/' + p + '/' + i)
            im = resize_depth(im, 4, False)
            cv2.imwrite(newpath + '/proj_depth/velodyne_raw/' + p + '/' + i, im)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Reduce size of KITTI dataset")

    parser.add_argument('--path_root', type=str, required=True,
                        help="Path to KITTI dataset")
    parser.add_argument('--path_out', type=str, required=False,
                        help="Path to reduce KITTI dataset")
    parser.add_argument('--scale', type=float, required=False,
                        default=4, help='The scale for reducing the size')
    args = parser.parse_args()

    """
    if os.path.exists(args.path_out):
        shutil.rmtree(args.path_out)

    for split in ['train','val']:
        for folder in os.listdir(os.path.join(args.path_root,split)):
            print("==> ", folder)
            src = os.path.join(args.path_root, split, folder)
            dst = os.path.join(args.path_out, split, folder)

            if not os.path.exists(dst): 
                os.makedirs(dst)
            
            resize_folder(src, dst, args.scale)
    """

    os.makedirs(os.path.join(args.path_out, 'depth_selection', 'val_selection_cropped', 'image'))
    os.makedirs(os.path.join(args.path_out, 'depth_selection', 'val_selection_cropped', 'groundtruth_depth'))
    os.makedirs(os.path.join(args.path_out, 'depth_selection', 'val_selection_cropped', 'velodyne_raw'))

    src = os.path.join(args.path_root, 'depth_selection', 'val_selection_cropped', 'intrinsic')
    dst = os.path.join(args.path_out, 'depth_selection', 'val_selection_cropped', 'intrinsic')
    shutil.copytree(src, dst)

    for i in os.listdir(os.path.join(args.path_root, 'depth_selection', 'val_selection_cropped', 'image')):
        im = Image.open(os.path.join(args.path_root, 'depth_selection', 'val_selection_cropped', 'image', i))
        im.thumbnail((93, 306), Image.ANTIALIAS)
        im.save(os.path.join(args.path_out, 'depth_selection', 'val_selection_cropped', 'image', i))

    for i in os.listdir(os.path.join(args.path_root, 'depth_selection', 'val_selection_cropped', 'groundtruth_depth')):
        im = Image.open(os.path.join(args.path_root, 'depth_selection', 'val_selection_cropped', 'groundtruth_depth', i))
        im = resize_depth(im, 4, True)
        cv2.imwrite(os.path.join(args.path_out, 'depth_selection', 'val_selection_cropped', 'groundtruth_depth', i), im)

    for i in os.listdir(os.path.join(args.path_root, 'depth_selection', 'val_selection_cropped', 'velodyne_raw')):
        im = Image.open(os.path.join(args.path_root, 'depth_selection', 'val_selection_cropped', 'velodyne_raw', i))
        im = resize_depth(im, 4, False)
        cv2.imwrite(os.path.join(args.path_out, 'depth_selection', 'val_selection_cropped', 'velodyne_raw', i), im)

    os.makedirs(os.path.join(args.path_out, 'depth_selection', 'test_depth_completion_anonymous', 'image'))
    os.makedirs(os.path.join(args.path_out, 'depth_selection', 'test_depth_completion_anonymous', 'groundtruth_depth'))
    os.makedirs(os.path.join(args.path_out, 'depth_selection', 'test_depth_completion_anonymous', 'velodyne_raw'))

    src = os.path.join(args.path_root, 'depth_selection', 'test_depth_completion_anonymous', 'intrinsic')
    dst = os.path.join(args.path_out, 'depth_selection', 'test_depth_completion_anonymous', 'intrinsic')
    shutil.copytree(src, dst)

    for i in os.listdir(os.path.join(args.path_root, 'depth_selection', 'test_depth_completion_anonymous', 'image')):
        im = Image.open(os.path.join(args.path_root, 'depth_selection', 'test_depth_completion_anonymous', 'image', i))
        im.thumbnail((93, 306), Image.ANTIALIAS)
        im.save(os.path.join(args.path_out, 'depth_selection', 'test_depth_completion_anonymous', 'image', i))

    for i in os.listdir(os.path.join(args.path_root, 'depth_selection', 'test_depth_completion_anonymous', 'velodyne_raw')):
        im = Image.open(os.path.join(args.path_root, 'depth_selection', 'test_depth_completion_anonymous', 'velodyne_raw', i))
        im = resize_depth(im, 4, False)
        cv2.imwrite(os.path.join(args.path_out, 'depth_selection', 'test_depth_completion_anonymous', 'velodyne_raw', i), im)
