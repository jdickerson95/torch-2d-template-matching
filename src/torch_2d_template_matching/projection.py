# this needs to do is take all of the angles and ake projection images
#Then take the given defoci and apply CTF
# can maybe also apply whitening filter here
import einops
import torch
import torch.nn.functional as F
from libtilt.projection import project_fourier
from libtilt.ctf.relativistic_wavelength import calculate_relativistic_electron_wavelength
from libtilt.ctf.ctf_2d import calculate_ctf


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

def pad_to_shape(
        image: torch.Tensor,
        image_shape: tuple[int,int],
        shape: tuple[int,int],
):
    y_pad = shape[0] - image_shape[0]
    x_pad = shape[1] - image_shape[1]
    p2d = (y_pad//2, y_pad//2 + y_pad%2, x_pad//2, x_pad//2 + x_pad%2)
    padded_image = F.pad(image, p2d, "constant", 0)
    return padded_image



def project_reference(
        volume_map: torch.Tensor,
        rotation_matrices: torch.Tensor,
        defoci: torch.Tensor,
        full_size: tuple[int,int],
        pixel_size = float,
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
    )
    # calculate the CTF. size of CTF will depend on how much need to pad image
    # I think pad to make sure Nyquist is still in the box
    # Not padding to full image size yet as I think it's more efficient to have smaller
    # boxes and pad later

    pad_distance = calculate_box_padding(
        pixel_size=pixel_size,  # A
        defoci=defoci,  # in um, negative underfocus
        beam_energy=beam_energy,  # eV
    )

    # pad the image on all sides with zeros
    # might be easier to sum to zero arrray of correct size
    p2d = (pad_distance, pad_distance, pad_distance, pad_distance)
    padded_projections = F.pad(projection_images, p2d, "constant", 0)
    # calculate a ctf of this size
    # I want negative defocus to be underfocus so flip the sign
    image_shape = padded_projections.shape()
    ctf = calculate_ctf(
            defocus=(defoci*-1),
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
    dft_projections = torch.fft.rfftn(padded_projections, dim=(-2, -1))
    ctf = einops.rearrange(ctf, 'b h w -> b 1 h w')
    ctf_dft_projection = dft_projections * ctf
    defocused_projections = torch.real(torch.fft.irfftn(ctf_dft_projection, dim=(-2, -1)))
    # pad the reference image to full size
    # again might be easier to sum to zero array of correct size
    reference_images = pad_to_shape(
        image= defocused_projections,
        image_shape=defocused_projections.shape[-2:],
        shape=full_size,
    )
    # fft shift to the edge
    reference_images = torch.fft.fftshift(reference_images, dim=(-2, -1))
    # return these projection images
    return reference_images





