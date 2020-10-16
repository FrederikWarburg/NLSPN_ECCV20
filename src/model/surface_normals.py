import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from PIL import Image

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from math import *


def sph2cart(rthetaphi):
    #takes list rthetaphi (single coord)
    r       = rthetaphi[0]
    theta   = rthetaphi[1]* pi/180 # to radian
    phi     = rthetaphi[2]* pi/180
    x = r * sin( theta ) * cos( phi )
    y = r * sin( theta ) * sin( phi )
    z = r * cos( theta )
    return [x,y,z]

def cart2sph(xyz):
    #takes list xyz (single coord)
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    r       =  np.linalg.norm(xyz, axis=-1)
    theta   =  acos(z/r)*180/ pi #to degrees
    phi     =  atan2(y,x)*180/ pi
    return [r,theta,phi]


def cart2cmap(xyz, cmap):
    """
        converts xyz coordinates to spherical coordinates and then displays in a specified colormap 
    """
    
    rthetaphi = cart2sph(xyz)
    phi = rthetaphi[1]
    theta = rthetaphi[2] + 180.0
    rgb = cmap[int(phi), int(theta)]

    return rgb
    

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

def calc_normals(depth):
    zy, zx = np.gradient(depth)  
    # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
    # to reduce noise

    normal = np.dstack((-zx, -zy, np.ones_like(depth)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    
    return normal

if __name__ == "__main__":
    
    basepath = '../TartanAirEuroc/'
    seq = 'P003'
    im_idx = '000450'

    depth0 = np.load(join(basepath + seq + "/depth_right/" + im_idx + "_right_depth.npy"))

    normal = calc_normals(depth0)

    normal_spherical = np.zeros_like(normal)
    cmap = create_custom_colormap()
    for x in range(normal.shape[0]):
        for y in range(normal.shape[1]):
            normal_spherical[x,y,:] = cart2cmap(normal[x,y,:],cmap)