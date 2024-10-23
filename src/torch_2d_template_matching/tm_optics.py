import einops
import torch
import mrcfile

import projection
import test_io
import simulate
import correlation

import libtilt.image_handler.modify_image as mdfy
import libtilt.filters.whitening as wht
import libtilt.filters.bandpass as bp

import sys

def main():
    pixel_size = 1.0
    #Take input can easily make this multiple
    simulated_image = sys.argv[1]
    simulated_projection = sys.argv[2]
    do_whiten = True
    bp_low = 1/99
    bp_high = 1/10
    
    
    #load the sim image
    mrc_image = test_io.load_mrc(simulated_image)
    #keep only 1 image if multiple given
    mrc_image = einops.reduce(mrc_image, '... h w -> h w', 'max')
    print(f"image shape: {mrc_image.shape}")
    #crop edge 100 pixels
    mrc_image = mrc_image[100:-100,100:-100]
    print(f"image shape: {mrc_image.shape}")
    #Get the whitening filter
    whitening_filter = wht.get_whitening_2d(mrc_image)
    if do_whiten:
        #Apply this filter to the image
        mrc_image = wht.whiten_image_2d(mrc_image, whitening_filter)
    if bp_low > 0:
        mrc_image = bp.bandpass_2d(mrc_image, bp_low,  bp_high, 0)
    #modify the image to mean zero and std 1
    mrc_image = mdfy.mean_zero(mrc_image)
    mrc_image = mdfy.std_one(mrc_image)
    
    #load the perfect image (already has ctf applied)
    projection_image = test_io.load_mrc(simulated_projection)
    #keep only 1 image if multiple given
    projection_image = einops.reduce(projection_image, '... h w -> h w', 'max')
    print(f"projection shape: {projection_image.shape}")
    #crop edge 100 pixels
    projection_image = projection_image[100:-100,100:-100]
    print(f"projection shape: {projection_image.shape}")
    #whiten image
    if do_whiten:
        projection_image = wht.whiten_image_2d(projection_image, whitening_filter)
    if bp_low > 0:
        projection_image = bp.bandpass_2d(projection_image, bp_low,  bp_high, 0)
    #modify image to mean 0 and std 1
    projection_image = mdfy.mean_zero(projection_image)
    projection_image = mdfy.std_one(projection_image)
    #shift to corner
    projection_image = torch.fft.fftshift(projection_image, dim=(-2, -1))
    
    #xcorr
    xcorr = correlation.cross_correlate_single(mrc_image, projection_image)
    std_xcorr = torch.std(xcorr)
    SNR_all = xcorr / std_xcorr
    
    #write this
    with mrcfile.new('test_xcorr_sim.mrc', overwrite=True) as mrc:
        mrc.set_data(SNR_all.detach().numpy())
    
    
if __name__ == "__main__":
    main()
