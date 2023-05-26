import torch


def cross_correlate(
        exp_images: torch.Tensor,
        projection_images: torch.Tensor,
):
    # at the moment, there is nothing about what to store
    image_dft = torch.fft.rfftn(exp_images, dim=(-2, -1))
    reference_dft = torch.fft.rfftn(projection_images, dim=(-2, -1))
    product = image_dft * reference_dft
    result = torch.real(torch.fft.irfftn(product, dim=(-2, -1)))
    return result
