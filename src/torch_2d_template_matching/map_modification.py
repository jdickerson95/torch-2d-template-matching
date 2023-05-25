from libtilt.filters.filters import b_envelope
from libtilt.ctf.relativistic_wavelength import calculate_relativistic_electron_wavelength
import torch
from libtilt.grids import fftfreq_grid



def apply_filter_3d(
        volume_map: torch.Tensor,
        filter: torch.Tensor,
):
    dft = torch.fft.rfftn(volume_map, dim=(-3, -2, -1))
    mod_dft = dft * filter
    return torch.real(torch.fft.irfftn(mod_dft, dim=(-3, -2, -1)))


def apply_b_map(
        volume_map: torch.Tensor,
        B: float,
        pixel_size: float,
):
    #Calculate the envelope
    b_env = b_envelope(
        B=B,
        image_shape=volume_map.shape[-3:],
        pixel_size=pixel_size,
        rfft=True,
        fftshifted=False,

    )
    #Apply
    return apply_filter_3d(
        map=volume_map,
        filter=b_env,
    )

def calculate_Cc_envelope(
        image_shape: torch.Tensor,
        pixel_size: float,
        Cc: float,
        beam_energy: float,
        delta_en: float,
        rfft: bool,
        fftshift: bool,
):
    frequency_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
    )
    frequency_grid_px = frequency_grid / pixel_size  # this grid is in A
    delta_defocus = (Cc * 1E7) * ((delta_en/beam_energy)**2)**0.5  # Cc mm -> A
    wavelength = calculate_relativistic_electron_wavelength(beam_energy)*1E10  # this returns in m -> A
    cc_tensor = torch.exp(-0.5 * (torch.pi*delta_defocus*wavelength)**2 * frequency_grid_px**4)
    return cc_tensor


def apply_Cc_envelope(
        volume_map: torch.Tensor,
        pixel_size: float,  # A
        Cc: float = 2.7,  # mm
        beam_energy: float = 300000,  # eV
        delta_en: float = 0.7,  # eV
        rfft: bool = True,
        fftshift: bool = False,
) :
    cc_env = calculate_Cc_envelope(
        image_shape=volume_map.shape[-3:],
        pixel_size=pixel_size,
        Cc=Cc,
        beam_energy=beam_energy,
        delta_en=delta_en,
        rfft=rfft,
        fftshift=fftshift,
    )
    return apply_filter_3d(
        map=map,
        filter=cc_env,
    )


def calculate_Cs_envelope(
        image_shape: torch.Tensor,
        pixel_size: float,
        Cs: float,
        beam_energy: float,
        semi_angle: float,
        defocus: float,
        rfft: bool,
        fftshift: bool,
):
    frequency_grid = fftfreq_grid(
        image_shape=image_shape,
        rfft=rfft,
        fftshift=fftshift,
        norm=True,
    )
    frequency_grid_px = frequency_grid / pixel_size  # this grid is in A
    wavelength = calculate_relativistic_electron_wavelength(beam_energy)*1E10  # this returns in m -> A
    semi_angle *= 1E3  # mrad -> rad
    defocus *= 1E4  # um -> A
    cc_tensor = torch.exp(-(torch.pi*semi_angle/wavelength)**2 *
                          (Cs*wavelength**3*frequency_grid_px**3 + wavelength*defocus*frequency_grid_px)**2)
    return cc_tensor


def apply_Cs_envelope(
        volume_map: torch.Tensor,
        pixel_size: float,  # A
        defocus: float, # in um, negative underfocus
        Cs: float = 2.7,  # mm
        beam_energy: float = 300000,  # eV
        semi_angle: float = 0.01,  # mrad
        rfft: bool = True,
        fftshift: bool = False,
) :
    cs_env = calculate_Cs_envelope(
        image_shape=volume_map.shape[-3:],
        pixel_size=pixel_size,
        Cs=Cs,
        beam_energy=beam_energy,
        semi_angle=semi_angle,
        defocus=defocus,
        rfft=rfft,
        fftshift=fftshift,
    )
    return apply_filter_3d(
        map=map,
        filter=cs_env,
    )



def phase_randomise_map():
    return 'test'




# still mtf, low/high pass filter

