#!/usr/bin/env python3
__copyright__ = """

    SLAMcore Confidential
    ---------------------

    SLAMcore Limited
    All Rights Reserved.
    (C) Copyright 2020

    NOTICE:

    All information contained herein is, and remains the property of SLAMcore
    Limited and its suppliers, if any. The intellectual and technical concepts
    contained herein are proprietary to SLAMcore Limited and its suppliers and
    may be covered by patents in process, and are protected by trade secret or
    copyright law. Dissemination of this information or reproduction of this
    material is strictly forbidden unless prior written permission is obtained
    from SLAMcore Limited.
"""

__license__ = "SLAMcore Confidential"

"""
Current script converts the TARTANAIR dataset to the EuRoC dataset (format used by
our SLAMcore modules)
"""
import argparse
import os
from os.path import join
import shutil
from collections import OrderedDict
from typing import Dict, List, Tuple

from loguru import logger
from scipy.io import loadmat
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

DEFAULT_NUM_CAMS = 2
BASELINE = 0.25 # m
FOCAL_LENGTH = 320 #

def create_euroc_filestruct(
    toplevel_dir: str, num_cams: int = DEFAULT_NUM_CAMS, *args, **kargs
) -> Tuple[List[str], List[str], List[str], str, str, List[str]]:
    """Create the file hierarchy for a dataset in the EuRoC format.

    :param toplevel_dir: Path to the directory to be created. If the given directory exists it will be deleted.
    :param num_cams: Number of camera directories to be initialised
    :returns: Tuple containing Paths to directories: camera, imu and ground truth
    """

    if os.path.isdir(toplevel_dir):
        logger.warning("Removing previously existing directory...")
        shutil.rmtree(toplevel_dir)
        os.makedirs(toplevel_dir)

    cam_dirs, depth_dirs, poses = [], [], []
    for i in range(num_cams):
        # image directories
        fname = join(toplevel_dir, "cam{}".format(i), "data")
        os.makedirs(fname)
        cam_dirs.append(fname)

        # depth
        fname = join(toplevel_dir, "ground_truth", "depth{}".format(i), "data")
        os.makedirs(fname)
        depth_dirs.append(fname)

        # pose
        fname = join(toplevel_dir, "ground_truth", "pose{}".format(i))
        os.makedirs(fname)
        poses.append(fname)

    # depth from sparse feature directories
    depth_sparse = join(toplevel_dir, "depth_sparse0", "data")
    os.makedirs(depth_sparse)

    # depth from SGBM directories
    depth_SGBM = join(toplevel_dir, "depth_SGBM0", "data")
    os.makedirs(depth_SGBM)

    return cam_dirs, depth_dirs, depth_sparse, depth_SGBM, poses


def matchFeatures(kp1, des1, kp2, des2):
    """ Match features based on descriptors and contraint matches such they are muturial best matches.

    :param kp1: Keypoints from frame 1
    :param des1: Descriptors from frame 1
    :param kp2: Keypoints from frame 2
    :param des2: Descriptors from frame 2
    :returns: An array with the best matches
    """

    # create BFMatcher object with  frame consistency (marriage problem)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # we know that the stereo images does not have any rotation, so we know that the y1 == y2. 
    # We can enforce this as an constraint
    idx = np.asarray([i for i, match in enumerate(matches) if abs(kp1[match.queryIdx].pt[1] - kp2[match.trainIdx].pt[1]) < 5])

    if len(idx) > 0:
        matches = np.asarray(matches)[idx]
    else:
        matches = []

    return matches

def computeFeatureDepth(matches, kp1, kp2):
    """ compute depth of features. Assuming only change in x-direction.
    Following this blogpost: 
    https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-iii-95460d3eddf0
    
    :param matches: index with best matches in frame 1 and frame 2
    :param kp1: Keypoints from frame 1
    :param kp2: Keypoints from frame 2
    :returns: The estimate points in 2.5D wrt frame 1 and 3D. 
    """
    
    points3D, points2_5D = [], []
    for match in matches: 
        x1 = kp1[match.queryIdx].pt[0]
        x2 = kp2[match.trainIdx].pt[0]

        y1 = kp1[match.queryIdx].pt[1]
        
        if (x2 -x1) <= 1: continue

        Z = (FOCAL_LENGTH * BASELINE) / (x2 -x1)
        X = x1 / FOCAL_LENGTH * Z
        Y = y1 / FOCAL_LENGTH * Z

        if Z > 0:
            points3D.append([X, Y, Z])
            points2_5D.append([x1, y1, Z])
        else:
            continue

    return np.asarray(points2_5D), np.asarray(points3D)

def computeMeasurementUncertainty(points3D, m_measurementError):
    """ compute measurement uncertainty from points in 3D
    Following this paper: 
    http://rpg.ifi.uzh.ch/docs/ICRA14_Pizzoli.pdf
    
    :param points3D: An array with 3D points
    :param m_measurementError: Measurement error in frame 2 in pixels
    :returns: estimated measurement uncertainty in 3D space.
    """
    
    sigmas = []
    for point1 in points3D:
        m_T_C1C2 = np.asarray([BASELINE, 0, 0])

        # Compute angle (alpha) between ray in the first camera and translation between cameras
        T_C1C2Norm = np.linalg.norm(m_T_C1C2)
        point1Norm = point1 / np.linalg.norm(point1)
        tNormized = m_T_C1C2 / T_C1C2Norm
        alpha = np.arccos(np.matmul(point1Norm, tNormized))
        
        # Compute angle (beta) between the vector difference between ray and translation in camera 1 frame
        pointA = point1 - m_T_C1C2
        pointANorm = pointA / np.linalg.norm(pointA)
        beta = np.arccos(-1.0 * np.matmul(pointANorm, tNormized))
        betaPlus = beta + 2.0 * np.arctan(m_measurementError / (2.0 * FOCAL_LENGTH))
        
        # Compute remaining angle (gamma) in triangle
        gamma = np.pi - alpha - betaPlus
        
        # Compute length of ray measurement error
        point1PlusNorm = T_C1C2Norm * ( np.sin(betaPlus) / np.sin(gamma) )
        
        # Compute uncertainty
        m_sigma = abs(point1PlusNorm - np.linalg.norm(point1))
    
        sigmas.append(m_sigma)

    return np.asarray(sigmas)

def computeDepthAndUncertaintyFromFeatures(imR, imL, detector, measurementError):
    """ The pipeline to compute depth with associated uncertainty from feature correspondences

    :param imR: right image
    :param imL: left image
    :param detector: feature detector (e.g AKAZE)
    :param measurementError: Measurement error in frame 2 in pixels
    :returns: An array with the detected features in 2.5D + their uncertainty in 3D space, thus each row has x1,y1,Z,sigma
    """

    # find the keypoints and descriptors
    kpR, desR = detector.detectAndCompute(imR,None)
    kpL, desL = detector.detectAndCompute(imL,None)

    # find matches
    if len(kpR) == 0 or len(kpL) == 0:
        return np.asarray([[]])

    matches = matchFeatures(kpR, desR, kpL, desL)

    if len(matches) == 0:
        return np.asarray([[]])
    
    # calculate depth
    points2_5D, points3D = computeFeatureDepth(matches, kpR, kpL)

    # calculate uncertainty for depth
    sigmas = computeMeasurementUncertainty(points3D, measurementError)

    # concatenate in one matrix
    features_and_uncertainties = np.concatenate([points2_5D, sigmas.reshape(-1,1)], axis =1) 
    
    return features_and_uncertainties

def computeDepthSGBM(imR, imL):
    """ The pipeline to compute depth with associated uncertainty from SGBM

    :param imR: right image
    :param imL: left image
    :returns: Two 1D images, one with the estimated depth and one with the estimated uncertainty.
    """

    imgL = cv2.cvtColor(imL,cv2.COLOR_RGB2GRAY)
    imgR = cv2.cvtColor(imR,cv2.COLOR_RGB2GRAY)

    win_size = 5
    min_disp = -1
    max_disp = 16 * 7 -1 #min_disp * 9
    num_disp = max_disp - min_disp # Needs to be divisible by 16
    #Create Block matching object. 
    stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
        numDisparities = num_disp,
        blockSize = 5,
        uniquenessRatio = 2,
        speckleWindowSize = 5,
        speckleRange = 1,
        disp12MaxDiff = 1,
        P1 = 4*3*win_size**2,
        P2 =32*3*win_size**2
    ) 

    #Compute disparity map
    disparity_map = stereo.compute(imgL, imgR)
    disparity_map = disparity_map.astype(float) / 16.0
    #Show disparity map before generating 3D cloud to verify that point cloud will be usable. 

    depth = np.zeros_like(disparity_map)
    depth[disparity_map > 0.0] = (BASELINE * FOCAL_LENGTH) / (disparity_map[disparity_map > 0.0])

    # https://drive.google.com/file/d/1nbYoZdKY7KzwUTm-0kFMUXqg2gY7rtLH/view
    # Eq 10 used to get uncertainties
    # Where we haven't observed any depth information we set the uncertainty to 0
    uncertainty = 0.0 * np.ones_like(depth)
    uncertainty[depth > 0.0] = depth[depth > 0.0] **2 / ( FOCAL_LENGTH * BASELINE )

    return depth, uncertainty

def main():
    """Main."""

    parser = argparse.ArgumentParser()
    parser.description = "Convert the TARTANAIR dataset -> EuRoC-like format"
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help=("Path to the top-level input directory for conversion"),
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output",
        required=False,
        help=("Path to the top-level input directory for conversion"),
    )
    parser.add_argument(
        "-e",
        "--error",
        default=1,
        required=False,
        help=("Measurement error in pixels (default is 1px)"),
    )

    # parse cmdline args
    args = parser.parse_args()

    # Initiate AKAZE detector
    akaze = cv2.AKAZE_create()

    for split in ['train', 'val', 'test']:
        for env in os.listdir(os.path.join(args.input, split)):
            for seqpath in os.listdir(join(args.input, split, env, 'Easy')):
                if not os.path.isdir(join(args.input, split, env, 'Easy', seqpath)): continue
                if seqpath == '.ipynb_checkpoints': continue

                dirs = create_euroc_filestruct(join(args.output, split, env, 'Easy', seqpath))

                for im in tqdm(sorted(os.listdir(join(args.input, split, env, 'Easy', seqpath, 'image_right'))), desc="Converting {}".format(seqpath)):
                    im = im.replace('_right.png','')

                    # load stereo data (RGB, depth, pose)
                    imR = np.asarray(Image.open(join(args.input, split, env, 'Easy',seqpath, 'image_right', im + '_right.png')))
                    imL = np.asarray(Image.open(join(args.input, split, env, 'Easy',seqpath, 'image_left', im + '_left.png')))

                    # estimate feature based sparse depth
                    est_features_and_uncertainties = computeDepthAndUncertaintyFromFeatures(imR, imL, akaze, args.error) 

                    # save estimated features and uncertainties
                    fname = join(dirs[2], im + '.csv')
                    np.savetxt(fname, est_features_and_uncertainties, delimiter = ',')

                    # estimate SGBM based semi-dense depth
                    est_depth_SGBM, est_uncertainties_SGBM = computeDepthSGBM(imR, imL)

                    # save estimated depth map and uncertainties based on SGBM
                    np.save(join(dirs[3], im + '_depth.npy'), est_depth_SGBM)
                    np.save(join(dirs[3], im + '_uncertainty.npy'), est_uncertainties_SGBM)

                    # copy images
                    for i, cam in enumerate(['right', 'left']):
                        src = join(args.input, split, env, 'Easy', seqpath, 'image_' + cam, im + '_' + cam + '.png')
                        dst = join(dirs[0][i], im + '.png')
                        shutil.copy(src, dst)

                    # copy depth images
                    for i, cam in enumerate(['right', 'left']):
                        src = join(args.input, split, env, 'Easy', seqpath, 'depth_' + cam, im + '_' + cam + '_depth.npy')
                        dst = join(dirs[1][i], im + '.npy')
                        shutil.copy(src, dst)


                    # copy segmentations images
                    for i, cam in enumerate(['right', 'left']):
                        src = join(args.input, split, env, 'Easy', seqpath, 'seg_' + cam, im + '_' + cam + '_seg.npy')
                        dst = join(dirs[1][i], im + '.npy')
                        shutil.copy(src, dst)

                # copy pose
                for i, cam in enumerate(['right', 'left']):
                    poses = np.loadtxt(join(args.input, split, env, 'Easy', seqpath, 'pose_' + cam + '.txt'))
                    np.savetxt(join(dirs[4][i], 'data.csv'), poses, delimiter = ',')

if __name__ == "__main__":
    main()
