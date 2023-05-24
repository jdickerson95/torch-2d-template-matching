from libtilt.rotational_averaging.rotational_average_dft import rotational_average_dft_2d
import torch


# whitening filter,
# note that the whitening filter must include the CTF, so is applied to the template after projection,
# also applied to the image
def apply_noise_filter(
        image:torch.Tensor,
):
    dft = torch.fft.rfftn(image, dim=(-2, -1))
    rotational_average_2d, frequency_bins_2d = rotational_average_dft_2d(
        dft=dft,
        image_shape=image.shape[-2:],
        rfft=False,
        fftshifted=False,
        return_2d_average=True #return 2D
    )
    whitening_filter = 1/rotational_average_2d**0.5
    mod_dft = dft * whitening_filter
    return torch.real(torch.fft.irfftn(mod_dft, dim=(-2, -1)))

#low/high pass filter, phase randomisation


