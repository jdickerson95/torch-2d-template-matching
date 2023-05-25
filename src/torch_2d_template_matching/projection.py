# this needs to do is take all of the angles and ake projection images
#Then take the given defoci and apply CTF
# can maybe also apply whitening filter here
import torch
from libtilt.projection import project_fourier
from libtilt.ctf.relativistic_wavelength import calculate_relativistic_electron_wavelength


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
    pad_distance = torch.abs(defocus * theta) / pixel_size  # distance on all sides in pixels
    return pad_distance

def project_reference(
        volume_map: torch.Tensor,
        rotation_matrices: torch.Tensor,
        defoci: torch.Tensor,
        pixel_size = float,
        beam_energy: float = 300000,
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
    # calculate a ctf of this size and apply the ctf
    # pad the reference image to full size
    # fft shift to the edge
    # return this projection image





