import einops
import torch
import mrcfile

import torch_2d_template_matching.test_io
import torch_2d_template_matching.correlation

import libtilt.image_handler.modify_image as mdfy
import libtilt.filters.whitening as wht
import libtilt.filters.bandpass as bp
import libtilt.fft_utils as fftutil

import sys

def calc_SNR(
    simulated_image: str,
    simulated_projection: str,
    pixel_size: float=1.0,
    do_whiten: bool=True,
    do_phase_randomize: bool=False,
    bp_low: float=-1/99,
    bp_high: float=1/2
):   
    #load the sim image
    mrc_image = torch_2d_template_matching.test_io.load_mrc(simulated_image)
    #keep only 1 image if multiple given
    mrc_image = einops.reduce(mrc_image, '... h w -> h w', 'max')
    #print(f"image shape: {mrc_image.shape}")
    #crop edge 100 pixels
    mrc_image = mrc_image[100:-100,100:-100]

    noise_image = torch.ones_like(mrc_image) * torch.mean(mrc_image)
    poisson_noise = torch.poisson(noise_image)
    #print(f"image shape: {mrc_image.shape}")
    #Get the whitening filter
    whitening_filter = wht.get_whitening_2d(mrc_image)
    if do_whiten:
        #Apply this filter to the image
        mrc_image = wht.whiten_image_2d(mrc_image, whitening_filter)
        poisson_noise = wht.whiten_image_2d(poisson_noise, whitening_filter)
    if bp_low > 0:
        mrc_image = bp.bandpass_2d(mrc_image, bp_low,  bp_high, 0)
        poisson_noise = bp.bandpass_2d(poisson_noise, bp_low,  bp_high, 0)
    #modify the image to mean zero and std 1
    mrc_image = mdfy.mean_zero(mrc_image)
    mrc_image = mdfy.std_one(mrc_image)
    poisson_noise = mdfy.mean_zero(poisson_noise)
    poisson_noise = mdfy.std_one(poisson_noise)
    
    #load the perfect image (already has ctf applied)
    projection_image = torch_2d_template_matching.test_io.load_mrc(simulated_projection)
    #keep only 1 image if multiple given
    projection_image = einops.reduce(projection_image, '... h w -> h w', 'max')
    #print(f"projection shape: {projection_image.shape}")
    #crop edge 100 pixels
    projection_image = projection_image[100:-100,100:-100]
    projection_image_random = torch.zeros_like(projection_image)
    #print(f"projection shape: {projection_image.shape}")
    #whiten image
    if do_whiten:
        projection_image = wht.whiten_image_2d(projection_image, whitening_filter)
    elif do_phase_randomize:
        projection_fft = torch.fft.rfftn(projection_image, dim=(-2, -1))
        cuton = 0
        projection_fft = fftutil.phase_randomize_2d(projection_fft, projection_image.shape, True, cuton)
        projection_image_random = torch.fft.irfftn(projection_fft, dim=(-2, -1))
    if bp_low > 0:
        projection_image = bp.bandpass_2d(projection_image, bp_low,  bp_high, 0)
    #modify image to mean 0 and std 1
    projection_image = mdfy.mean_zero(projection_image)
    projection_image = mdfy.std_one(projection_image)
    #shift to corner
    projection_image = torch.fft.fftshift(projection_image, dim=(-2, -1))

    if do_phase_randomize:
        projection_image_random = mdfy.mean_zero(projection_image_random)
        projection_image_random = mdfy.std_one(projection_image_random)
        projection_image_random = torch.fft.fftshift(projection_image_random, dim=(-2, -1))
    
    #xcorr
    xcorr = torch_2d_template_matching.correlation.cross_correlate_single(mrc_image, projection_image)
    xcorr_poisson =  torch_2d_template_matching.correlation.cross_correlate_single(poisson_noise, projection_image)
    xcorr_random = torch.zeros_like(xcorr)
    if do_phase_randomize:
        xcorr_random = torch_2d_template_matching.correlation.cross_correlate_single(mrc_image, projection_image_random)
    xcorr -= xcorr_random
    xcorr += torch.mean(xcorr_random)
    std_xcorr = torch.std(xcorr)
    SNR_all = xcorr / std_xcorr
    SNR_poisson = xcorr / torch.std(xcorr_poisson)
    max_SNR = torch.max(SNR_all)
    max_SNR_poisson = torch.max(SNR_poisson)
    max_signal = torch.max(xcorr).item()
    return [max_SNR, max_signal, max_SNR_poisson]
