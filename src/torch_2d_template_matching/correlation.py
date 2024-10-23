import torch
import einops


def cross_correlate(
        exp_images: torch.Tensor,
        projection_images: torch.Tensor,
):
    # at the moment, there is nothing about what to store
    print("dfts of image")
    image_dft = torch.fft.rfftn(exp_images, dim=(-2, -1))
    
    #Should apply a filter here
    
    
    print("dfts of reference")
    reference_dft = torch.fft.rfftn(projection_images, dim=(-2, -1))   
    
    # collapse references to same dimension
    reference_dft = einops.rearrange(reference_dft, 'defoc angles h w -> (defoc angles) h w')
    
    #zero central pixel here for good measure
    reference_dft[:,0,0] = 0 + 0j
    
    
    print("convolution")
    product = image_dft * reference_dft
    print("Back to real")

    result = torch.real(torch.fft.irfftn(product, dim=(-2, -1)))
    return result
    
def cross_correlate_single(
        exp_images: torch.Tensor,
        projection_images: torch.Tensor,
):
    # at the moment, there is nothing about what to store
    print("dfts of image")
    image_dft = torch.fft.rfftn(exp_images, dim=(-2, -1))

    print("dfts of reference")
    reference_dft = torch.fft.rfftn(projection_images, dim=(-2, -1))   
    
    print("convolution")
    product = image_dft * reference_dft
    print("Back to real")

    result = torch.real(torch.fft.irfftn(product, dim=(-2, -1)))
    return result
