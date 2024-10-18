"""A toy implementation of a template matching routine."""

import einops
import mmdf
import numpy as np
import torch
from eulerangles import matrix2euler, euler2matrix

from libtilt.interpolation import insert_into_image_2d
from scipy.spatial.transform import Rotation as R 

import map_modification
import so3_grid
import projection
import test_io
import simulate
import correlation

import sys


def main():
    #Take input
    simulated_image = sys.argv[1]
    simulated_map = sys.argv[2]
    #Get the rotation matrix
    rotvec= [0, 0, 0]
    r = R.from_rotvec(rotvec)
    euler_angles = r.as_euler('zyz')
    rotation_matrix = torch.from_numpy(euler2matrix(euler_angles, axes='zyz', intrinsic=True,right_handed_rotation=True))
    
    #Read in image and map
    
    #apply filters to the map if necessary
    
    #Extract Fourier slice
    
    
    #Apply CTF and any filters or envelope function
    
    #FFT and apply filters to micrograph
    
    #Cross correlate
    
    #Get the max pixel
    
    #Cross correlate to random noise of same mean and std
    
    #Calculate and print out SNR
    






if __name__ == "__main__":
    main()
