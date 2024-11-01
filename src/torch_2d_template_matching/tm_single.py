"""A toy implementation of a template matching routine."""

import einops
import mmdf
import numpy as np
import torch
import torch.nn.functional as F
import mrcfile
from eulerangles import matrix2euler, euler2matrix

from libtilt.interpolation import insert_into_image_2d
from libtilt.filters.whitening import get_whitening_2d
from scipy.spatial.transform import Rotation as R 

import map_modification
import so3_grid
import projection
import test_io
import simulate
import correlation

import sys



def main():
    pixel_size = 1.0
    #Take input can easily make this multiple
    simulated_image = sys.argv[1]
    simulated_map = sys.argv[2]
    
    #Get the rotation matrix
    rotvecs = np.array([[0, 0, 0],[1, 1, 1]])
    rot_len = rotvecs.shape[0]
    r = R.from_rotvec(rotvecs)
    euler_angles = r.as_euler('zyz', degrees=True)
    print(euler_angles.shape)
    rotation_matrices = torch.from_numpy(euler2matrix(euler_angles, axes='zyz', intrinsic=True,right_handed_rotation=False)).float()
    
    #Read in image and map
    mrc_map = test_io.load_mrc(simulated_map)
    mrc_image = test_io.load_mrc(simulated_image)
    #crop edge 100 pixels
    mrc_image = mrc_image[100:-100,100:-100]
    
    #should apply whitening here
    whitening_filter = get_whitening_2d(mrc_image)
    image_dft = torch.fft.rfftn(mrc_image, dim=(-2, -1))
    #image_dft *= whitening_filter
    mrc_image = torch.real(torch.fft.irfftn(image_dft, dim=(-2, -1)))
    #subtract means
    mrc_image -= torch.mean(mrc_image) 
    #normlize std 1
    mrc_image /= torch.std(mrc_image)
    #save normal image
    with mrcfile.new('test_normal_sum.mrc', overwrite=True) as mrc:
        mrc.set_data(mrc_image.detach().numpy())    
    
    image_size = mrc_image.shape
    volume_full_size = (min(image_size),image_size[0],image_size[1])
    #pad map to image size with edge mean
    # Average edge pixels (take this out to util)
    mean_edge = projection.average_edge_pixels_3d(mrc_map)
    #pad volume with this average
    padded_volume = projection.pad_to_shape_3d(
            volume=mrc_map,
            volume_shape=mrc_map.shape,
            shape=volume_full_size,
            pad_val=mean_edge,
    )
    #sets mean zero of this by zero central pixel
    #pad
    pad_length = padded_volume.shape[-1] //2
    #padded_volume = F.pad(padded_volume, pad=[pad_length] * 6, mode='constant', value=0)
    dft_volume = torch.fft.fftshift(padded_volume, dim=(-3, -2, -1))
    dft_volume = torch.fft.rfftn(dft_volume, dim=(-3, -2, -1))
    dft_volume[0,0,0] = 0 + 0j
    padded_volume = torch.fft.irfftn(dft_volume, dim=(-3, -2, -1))
    padded_volume = torch.fft.ifftshift(padded_volume, dim=(-3, -2, -1))
    #padded_volume = torch.real(padded_volume[..., pad_length:-pad_length, pad_length:-pad_length, pad_length:-pad_length])
    
    
    #I think keeping the volume small and padding it a bit for ctf and then again for whitening
    # is the more efficient thing to do here
    
    
    #save normal volume
    with mrcfile.new('test_normal_volume.mrc', overwrite=True) as mrc:
        mrc.set_data(padded_volume.detach().numpy())    
    
    #Extract Fourier slice
    #defoci = torch.tensor([-1.0, 1.0])
    defoci = torch.arange(-1.6, 0.4, 0.2)
    defoc_len = defoci.shape[0]
    
    
    projections = projection.project_reference(padded_volume, rotation_matrices, defoci, mrc_image.shape, pixel_size, whitening_filter)
    
    
    
    #Can apply to the perfect optics from simulator and exit wavefunction and check it's the same
    
    
    

    
    #Apply CTF and any filters or envelope function (in projection now)
    #mean_proj = einops.reduce(projections, 'defoc angles h w -> defoc angles', 'mean')
    #mean_proj = einops.rearrange(mean_proj, 'defoc angles -> defoc angles 1 1')
    #normal_proj = projections - mean_proj
    
    #Save these for testing - looks good
    save_projections = torch.fft.fftshift(projections, dim=(-2, -1))
    #save_projections = projections
    with mrcfile.new('test_proj.mrc', overwrite=True) as mrc:
        mrc.set_data(save_projections.detach().numpy())
    

    
    #Cross correlate
    xcorr = correlation.cross_correlate(mrc_image, projections)
    print(projections.shape)
    print(xcorr.shape)
    std_xcorr = einops.reduce(xcorr, 'b h w -> b', reduction=projection.std_reduction)
    std_xcorr = einops.rearrange(std_xcorr, 'b -> b 1 1')
    SNR_all = xcorr / std_xcorr
    #Get the max pixels - I currently lose the orientation and defocus that produced each pixel
    max_index = torch.argmax(SNR_all, axis=0)
    rot_index = max_index % rot_len
    rot_map = torch.from_numpy(euler_angles[rot_index,:]).float()
    rot_map = einops.rearrange(rot_map, 'h w b -> b h w')
    print(rot_map.shape)
    defoc_index = max_index // rot_len
    defoc_map = defoci[defoc_index]
    print(defoc_map.shape)
    print(max_index.shape)
    max_pixels = einops.reduce(SNR_all, 'b h w -> h w', 'max')
    print(max_pixels.shape)
    
    
    #Calculate and print out SNR
   
    
    #write outputs to file
    #save all xcorr as test
    with mrcfile.new('test_xcorr_all.mrc', overwrite=True) as mrc:
        mrc.set_data(SNR_all.detach().numpy())    
    #max xcorr map
    with mrcfile.new('test_xcorr.mrc', overwrite=True) as mrc:
        mrc.set_data(max_pixels.detach().numpy())
    #max defocus map
    with mrcfile.new('test_defoc.mrc', overwrite=True) as mrc:
        mrc.set_data(defoc_map.detach().numpy())    
    #Max angle map (3 frames mrc)
    with mrcfile.new('test_angles.mrc', overwrite=True) as mrc:
        mrc.set_data(rot_map.detach().numpy())   

    #do some thresholding
    





if __name__ == "__main__":
    main()