"""A toy implementation of a template matching routine."""

import einops
import mmdf
import numpy as np
import torch
import napari
from eulerangles import matrix2euler

from libtilt.interpolation import insert_into_image_2d

import map_modification
import so3_grid
import projection
import test_io
import simulate
import correlation


def main():
    # get inputs, still need to do this
    B = 50
    # euler angle range so user can restrict range, agian will be a user input
    phi = (-180, 180)
    theta = (-90, 90)
    psi = (-180, 180)
    # load mrc
    sim_image_shape = (2048, 2048)
    pixel_size = 1.0
    mrc_filepath = "/Users/josh/git/torch-2d-template-matching/data/7qn5.mrc"
    mrc_map = test_io.load_mrc_map(mrc_filepath)
    # apply any filters to the model

    mrc_map = map_modification.apply_b_map(mrc_map, B, pixel_size)
    # project model
    res = 0
    h3_grid = so3_grid.get_h3_grid_at_resolution(res)
    rotation_matrices = torch.zeros((len(h3_grid), 3, 3))
    for i, h in enumerate(h3_grid):
        rotation_matrices[i] = so3_grid.h3_to_rotation_matrix(h)
    # convert to euler
    euler_angles = matrix2euler(rotation_matrices,
                                axes='zyz',
                                intrinsic=True,
                                right_handed_rotation=True)
    # only search in euler angle range
    indices = np.argwhere(
        (phi[0] <= euler_angles[:, 0]) & (euler_angles[:, 0] < phi[1]) & (theta[0] <= euler_angles[:, 1])
        & (euler_angles[:, 1] < theta[1]) & (psi[0] <= euler_angles[:, 2]) & (euler_angles[:, 2] < psi[1]))
    indices = np.reshape(indices, (-1,))
    rotation_matrices = rotation_matrices[indices]
    defoci = torch.arange(-1.2, -0.8, 0.2)
    projections = projection.project_reference(mrc_map, rotation_matrices, defoci, sim_image_shape, pixel_size)
    # load images or make simulated image
    sim_images = simulate.main(
        phi=phi,
        theta=theta,
        psi=psi,
        sim_image_shape=sim_image_shape,
        sim_pixel_spacing=pixel_size
    )
    # do correlation
    print(f"projections:{projections.shape}")
    print(f"sim_images:{sim_images.shape}")
    print("starting xcorr")
    xcorr = correlation.cross_correlate(sim_images, projections)
    print('test')
    
    viewer = napari.Viewer()
    viewer.add_image(
        sim_images.numpy(),
        name='sim_images',
        contrast_limits=(0, torch.max(sim_images))
    )
    viewer.add_image(
        xcorr.numpy(),
        name='template matching result',
        contrast_limits=(0, torch.max(xcorr)),
        colormap='inferno',
        blending='additive',
        opacity=0.3,
    )
    napari.run()
    

if __name__ == "__main__":
    main()

    '''
    INPUT_MODEL_FILE = 'data/4v6x-ribo.cif'
    N_PARTICLES = 30
    SIMULATION_PIXEL_SPACING = 1
    SIMULATION_IMAGE_SHAPE = (4096, 4096)
    ADD_NOISE = False

    # load molecular model, center and rescale
    print(f"loading model from {INPUT_MODEL_FILE}")
    df = mmdf.read(INPUT_MODEL_FILE)
    atom_zyx = torch.tensor(df[['z', 'y', 'x']].to_numpy()).float()  # (n_atoms, 3)
    atom_zyx -= torch.mean(atom_zyx, dim=-1, keepdim=True)
    atom_zyx /= SIMULATION_PIXEL_SPACING

    # randomly place in volume
    print("placing particles in 3D volume")
    volume_shape = (SIMULATION_IMAGE_SHAPE[0] // 3, *SIMULATION_IMAGE_SHAPE)
    pz, py, px = [
        np.random.uniform(low=0, high=dim_length, size=N_PARTICLES)
        for dim_length in volume_shape
    ]
    particle_positions = einops.rearrange([pz, py, px], 'zyx b -> b 1 zyx')
    per_particle_atom_positions = atom_zyx + particle_positions

    # simulate image
    print("simulating image")
    atom_yx = per_particle_atom_positions[..., 1:]
    atom_yx = einops.rearrange(atom_yx, 'particles atoms yx -> (particles atoms) yx')
    n_atoms = atom_yx.shape[0]
    values = torch.ones(n_atoms)
    image = torch.zeros(SIMULATION_IMAGE_SHAPE)
    weights = torch.zeros_like(image)
    image, weights = insert_into_image_2d(
        data=values,
        coordinates=atom_yx,
        image=image,
        weights=weights
    )

    if ADD_NOISE is True:
        image = image + np.random.normal(loc=0, scale=50, size=SIMULATION_IMAGE_SHAPE)
    else:
        image = image

    # simulate a reference image for template matching
    print("simulating reference")
    reference_zyx = atom_zyx + np.array([0, *SIMULATION_IMAGE_SHAPE]) // 2
    reference_yx = reference_zyx[..., 1:]
    n_atoms = reference_yx.shape[0]

    values = torch.ones(n_atoms)
    reference = torch.zeros(SIMULATION_IMAGE_SHAPE)
    weights = torch.zeros_like(image)
    reference, weights = insert_into_image_2d(
        data=values,
        coordinates=reference_yx,
        image=reference,
        weights=weights,
    )
    reference = torch.fft.fftshift(reference, dim=(-2, -1))

    print("convolution theorem-ing it up")
    image_dft = torch.fft.rfftn(image, dim=(-2, -1))
    reference_dft = torch.fft.rfftn(reference, dim=(-2, -1))
    product = image_dft * reference_dft
    result = torch.real(torch.fft.irfftn(product, dim=(-2, -1)))

    # visualise results
    viewer = napari.Viewer()
    viewer.add_image(
        image.numpy(),
        name='image',
        contrast_limits=(0, torch.max(image))
    )
    viewer.add_image(
        reference.numpy(),
        name='reference',
        visible=False,
        contrast_limits=(0, torch.max(reference))
    )
    viewer.add_image(
        result.numpy(),
        name='template matching result',
        contrast_limits=(0, torch.max(result)),
        colormap='inferno',
        blending='additive',
        opacity=0.3,
    )
    napari.run()
    '''
