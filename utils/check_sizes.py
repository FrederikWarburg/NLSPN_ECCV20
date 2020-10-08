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

def check_sizes(basepath, newpath, scale):

    for p in ['image_02', 'image_03']:
        for i in os.listdir(basepath + '/' +  p +'/data'):
            if not os.path.exists(basepath + '/proj_depth/groundtruth/'+p+'/' + i): continue
            if not os.path.exists(basepath + '/proj_depth/velodyne_raw/' + p + '/' + i): continue
            
            im = Image.open(basepath  + '/' + p + '/data/' + i)
            wa, ha = im.size
            h,w = im.size
            hb, wb = int(np.ceil(h / scale)), int(np.ceil(w / scale))
            im.thumbnail((hb, wb), Image.ANTIALIAS)
            im.save(newpath  + '/' + p + '/data/' + i)
            
            w1, h1 = im.size
            im = Image.open(basepath + '/proj_depth/groundtruth/'+p+'/' + i)
            wb, hb = im.size
            im = resize_depth(im, 4, True)
            cv2.imwrite(newpath + '/proj_depth/groundtruth/'+p+'/' + i, im)

            h2, w2 = im.shape

            im = Image.open(basepath + '/proj_depth/velodyne_raw/' + p + '/' + i)
            wc, hc = im.size
            im = resize_depth(im, 4, False)
            cv2.imwrite(newpath + '/proj_depth/velodyne_raw/'+p+'/' + i, im)
  
            h3, w3 = im.shape

            if h1 != h2 or h1 != h3 or w1 != w2 or w1 != w3:
                print(w1,w2,w3,h1,h2,h3)
                print(wa,wb,wc,ha,hb,hc)
                print(basepath  + '/' + p + '/data/' + i)
                print(basepath + '/proj_depth/groundtruth/'+p+'/' + i)
                print(basepath + '/proj_depth/velodyne_raw/' + p + '/' + i)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Reduce size of KITTI dataset")

    parser.add_argument('--path_root', type=str, required=True,
                        help="Path to KITTI dataset")
    parser.add_argument('--path_out', type=str, required=False,
                        help="Path to reduce KITTI dataset")
    parser.add_argument('--scale', type=float, required=False,
                        default=4, help='The scale for reducing the size')
    args = parser.parse_args()


    for split in ['train','val']:
        for folder in os.listdir(os.path.join(args.path_root,split)):
            print("==> ", folder)
            src = os.path.join(args.path_root, split, folder)
            dst = os.path.join(args.path_out, split, folder)

            check_sizes(src, dst, args.scale)
