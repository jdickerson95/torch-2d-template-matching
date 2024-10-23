# this needs to do is take all of the angles and ake projection images
#Then take the given defoci and apply CTF
# can maybe also apply whitening filter here
import einops
import torch
import torch.nn.functional as F
from libtilt.projection.project_fourier import project_fourier
from libtilt.ctf.relativistic_wavelength import calculate_relativistic_electron_wavelength
from libtilt.ctf.ctf_2d import calculate_ctf

def average_edge_pixels_3d(volume):
    # Create a mask for edge pixels
    edge_mask = torch.zeros_like(volume, dtype=torch.bool)
    edge_mask[0, :, :] = True
    edge_mask[-1, :, :] = True
    edge_mask[:, 0, :] = True
    edge_mask[:, -1, :] = True
    edge_mask[:, :, 0] = True
    edge_mask[:, :, -1] = True

    # Extract edge pixels and calculate the average
    edge_pixels = volume[edge_mask]
    average = torch.mean(edge_pixels)

    return average

def std_reduction(
	image: torch.Tensor,
	reduced_axes: torch.Tensor
):
    return torch.std(image, dim=reduced_axes, unbiased=False)
    
def edge_mean_reduction_2d(
	image: torch.Tensor,
	reduced_axes: torch.Tensor
):
    # Create a mask for edge pixels
    top_edge = image[:, :, 0, :]
    bottom_edge = image[:, :, -1, :]
    left_edge = image[:, :, :, 0]
    right_edge = image[:, :, :, -1]
    
    edge_pixels = torch.cat([top_edge, bottom_edge, left_edge, right_edge],dim=2)
    average = torch.mean(edge_pixels, dim=2)

    return average

def calculate_box_padding(
        pixel_size: float,  # A
        defoci: torch.Tensor,  # in um, negative underfocus
        beam_energy: float,  # eV
):
    """
    Padding a box so no information is lost when the CTF is applied
    """
    # I'm going to ignore the Cs, then the highest resolution always moves furthest
    # and I have an overestimation so box will be big enough
    defocus = torch.max(torch.abs(defoci)) * 1E4  # use max defocus, A
    nyquist_freq = pixel_size * 2
    wavelength = calculate_relativistic_electron_wavelength(beam_energy) * 1E10  # this returns in m -> A
    theta = nyquist_freq * wavelength
    pad_distance = int(torch.abs(defocus * theta) / pixel_size)  # distance on all sides in pixels
    pad_distance += pad_distance%2  # make it even
    return pad_distance


def pad_to_shape_2d(
        image: torch.Tensor,
        image_shape: tuple[int,int],
        shape: tuple[int,int],
        pad_val: float,
):
    y_pad = shape[0] - image_shape[0]
    x_pad = shape[1] - image_shape[1]
    p2d = (y_pad//2, y_pad//2 + y_pad%2, x_pad//2, x_pad//2 + x_pad%2)
    padded_image = F.pad(image, p2d, "constant", pad_val)
    return padded_image
    
def pad_to_shape_3d(
        volume: torch.Tensor,
        volume_shape: tuple[int,int,int],
        shape: tuple[int,int,int],
        pad_val: float,
):
    z_pad = shape[0] - volume_shape[0]
    y_pad = shape[1] - volume_shape[1]
    x_pad = shape[2] - volume_shape[2]
    p3d = (z_pad//2, z_pad//2 + z_pad%2, y_pad//2, y_pad//2 + y_pad%2, x_pad//2, x_pad//2 + x_pad%2)
    padded_volume = F.pad(volume, p3d, "constant", pad_val)
    return padded_volume


def project_reference(
        volume_map: torch.Tensor,
        rotation_matrices: torch.Tensor,
        defoci: torch.Tensor,
        full_size: tuple[int, int],
        pixel_size: float,
        whitening_filter: torch.Tensor,
        beam_energy: float = 300000,
        astigmatism: float = 0,
        astigmatism_angle: float = 0,
        spherical_aberration: float = 2.7,
        amplitude_contrast: float = 0.07,
        phase_shift: float = 0,
        rfft: bool = True,
        fftshift: bool = False,
):
    # it may be more efficient to stay in Fourier space
    # until no more modifications are made, but I will keep FFT/iFFT
    # for now with a possibility of changing later

    # project the map
    projection_images = project_fourier(
        volume=volume_map,
        rotation_matrices=rotation_matrices,
        rotation_matrix_zyx=False,   #Need to triple check that this is correct
        pad=True
    )
    
    
    # calculate the CTF. size of CTF will depend on how much need to pad image
    # I think pad to make sure Nyquist is still in the box
    # Not padding to full image size yet as I think it's more efficient to have smaller
    # boxes and pad later
    '''
    pad_distance = calculate_box_padding(
        pixel_size=pixel_size,  # A
        defoci=defoci,  # in um, negative underfocus
        beam_energy=beam_energy,  # eV
    )
    '''
    # pad the image on all sides with zeros
    # might be easier to sum to zero arrray of correct size
    #p2d = (pad_distance, pad_distance, pad_distance, pad_distance)
    #padded_projections = F.pad(projection_images, p2d, "constant", 0)
    # calculate a ctf of this size
    # I want negative defocus to be underfocus so flip the sign
    image_shape = projection_images.shape[-2:]
    ctf = calculate_ctf(
            defocus=defoci,
            astigmatism=astigmatism,
            astigmatism_angle=astigmatism_angle,
            voltage=beam_energy,
            spherical_aberration=spherical_aberration,
            amplitude_contrast=amplitude_contrast,
            b_factor=0,
            phase_shift=phase_shift,
            pixel_size=pixel_size,
            image_shape=image_shape,
            rfft=rfft,
            fftshift=fftshift,
    )
    # apply the ctf
    dft_projections = torch.fft.rfftn(projection_images, dim=(-2, -1))
    ctf = einops.rearrange(ctf, 'b h w -> b 1 h w')
    ctf_dft_projection = dft_projections * ctf
    
    
    #need to apply whitening filter, then zero central pixel
    #ctf_dft_projection *= whitening_filter
    
    
    #mean zero
    ctf_dft_projection[:, :, 0, 0] = 0 + 0j
    
    defocused_projections = torch.real(torch.fft.irfftn(ctf_dft_projection, dim=(-2, -1)))
    # pad the reference image to full size
    # again might be easier to sum to zero array of correct size
    
    #proj mean zero
    mean_proj = einops.reduce(defocused_projections, 'defoc angles h w -> defoc angles', reduction=edge_mean_reduction_2d)
    mean_proj = einops.rearrange(mean_proj, 'defoc angles -> defoc angles 1 1')
    defocused_projections = defocused_projections - mean_proj
    #flip contrast
    defocused_projections *= -1
    #proj std 1
    std_proj = einops.reduce(defocused_projections, 'defoc angles h w -> defoc angles', reduction=std_reduction)
    std_proj = einops.rearrange(std_proj, 'defoc angles -> defoc angles 1 1')
    defocused_projections = defocused_projections / std_proj
    '''
    mean_proj = einops.reduce(defocused_projections, 'defoc angles h w -> defoc angles', reduction=edge_mean_reduction_2d)
    mean_proj = einops.rearrange(mean_proj, 'defoc angles -> defoc angles 1 1')
    defocused_projections = defocused_projections - mean_proj
    
    reference_images = pad_to_shape_2d(
        image= defocused_projections,
        image_shape=defocused_projections.shape[-2:],
        shape=full_size,
        pad_val=0,
    )
    '''
    # fft shift to the edge

    reference_images = torch.fft.fftshift(defocused_projections, dim=(-2, -1))
    
    
    
    # return these projection images
    return reference_images
    
def modify_perf_image(
    projection_image: torch.Tensor,
):
    print('test')
    





