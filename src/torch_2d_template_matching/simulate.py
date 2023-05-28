# Simulating images to make testing easier
# Use a pdb/cif model to generate simulated images


import einops
import numpy as np
import torch
import mmdf
from libtilt.interpolation import insert_into_image_2d
from libtilt.ctf.ctf_2d import calculate_ctf
from so3_grid import get_h3_grid_at_resolution
from so3_grid import h3_to_rotation_matrix
from eulerangles import matrix2euler
import napari


def load_model(
        file_path: str,
        pixel_size: float,
) -> torch.Tensor:
    df = mmdf.read(file_path)
    atom_zyx = torch.tensor(df[['z', 'y', 'x']].to_numpy()).float()  # (n_atoms, 3)
    atom_zyx -= torch.mean(atom_zyx, dim=-1, keepdim=True)  # center
    atom_zyx /= pixel_size
    return atom_zyx


def place_in_volume(
        num_particles: int,
        atom_zyx: torch.Tensor,
        image_shape: tuple[int, int],
        phi: float,
        theta: float,
        psi: float
):
    # here I get H3 grid at a particular resolution
    h3_res = 0  # 0 to restrict space for testing
    h3_grid = get_h3_grid_at_resolution(0)
    rotation_matrices_all = torch.zeros((len(h3_grid), 3, 3))
    for i, h in enumerate(h3_grid):
        rotation_matrices_all[i] = h3_to_rotation_matrix(h)
    # convert to euler
    euler_angles = matrix2euler(rotation_matrices_all,
                                axes='zyz',
                                intrinsic=True,
                                right_handed_rotation=True)
    # only search in euler angle range
    indices = np.argwhere(
        (phi[0] <= euler_angles[:, 0]) & (euler_angles[:, 0] < phi[1]) & (theta[0] <= euler_angles[:, 1])
        & (euler_angles[:, 1] < theta[1]) & (psi[0] <= euler_angles[:, 2]) & (euler_angles[:, 2] < psi[1]))
    indices = np.reshape(indices, (-1,))
    rotation_matrices_all = rotation_matrices_all[indices]

    # Take n random matrices
    weights = torch.ones(rotation_matrices_all.shape[0])  # Use even weights
    # I would ideally set these to 0
    random_indices = torch.multinomial(weights, num_particles, replacement=True)  # replace so same orientation can
    # convert to rotation matrices
    rotation_matrices = torch.zeros((num_particles, 3, 3))
    for i, idx in enumerate(random_indices):

        rotation_matrices[i] = h3_to_rotation_matrix(h3_grid[idx])

    # rotate the atoms to these
    rotated_atom_zyx = torch.matmul(atom_zyx, rotation_matrices)
    '''
    rotation_matrices = einops.rearrange(rotation_matrices, '... i j -> ... 1 i j')
    atom_zyx = einops.rearrange(atom_zyx, 'n coords -> n coords 1')
    rotated_atom_zyx = rotation_matrices @ atom_zyx
    rotated_atom_zyx = einops.rearrange(rotated_atom_zyx, '... n coords 1 -> ... n coords')
    '''
    print(rotated_atom_zyx.shape)
    # Place these in a 3D volume
    print("placing particles in 3D volume")
    # I'm placing in z, but I am going to apply the same ctf to them all
    # so they will all have the same defocus
    volume_shape = (image_shape[0] // 3, *image_shape)
    # I think things off the edge was causing an issue
    # volume_shape = (image_shape[0] // 3, image_shape[0]-image_shape[0]/5, image_shape[1]-image_shape[1]/5)
    pz, py, px = [
        np.random.uniform(low=0, high=dim_length, size=num_particles)
        for dim_length in volume_shape
    ]
    particle_positions = einops.rearrange([pz, py, px], 'zyx b -> b 1 zyx')
    print(particle_positions.shape)
    particle_atom_positions = rotated_atom_zyx + particle_positions
    print(particle_atom_positions.shape)
    return particle_atom_positions


def simulate_image(
        per_particle_atom_positions: torch.Tensor,
        image_shape: tuple[int, int]
):
    print("simulating image without ctf applied")
    atom_yx = per_particle_atom_positions[..., 1:]
    atom_yx = einops.rearrange(atom_yx, 'particles atoms yx -> (particles atoms) yx')
    n_atoms = atom_yx.shape[0]
    values = torch.ones(n_atoms)
    image = torch.zeros(image_shape)
    weights = torch.zeros_like(image)
    image, weights = insert_into_image_2d(
        data=values,
        coordinates=atom_yx,
        image=image,
        weights=weights
    )
    return image


def apply_ctf(
        sim_image: torch.Tensor,
        defocus: float,
        image_shape: tuple[int, int],
        pixel_size: float
):
    ctf = calculate_ctf(
        defocus=(defocus * -1),
        astigmatism=0,
        astigmatism_angle=0,
        voltage=300000,
        spherical_aberration=2.7,
        amplitude_contrast=0.07,
        b_factor=0,
        phase_shift=0,
        pixel_size=pixel_size,
        image_shape=image_shape,
        rfft=True,
        fftshift=False,
    )

    dft_image = torch.fft.rfftn(sim_image, dim=(-2, -1))
    sim_fourier = dft_image * ctf
    sim_image_ctf = torch.real(torch.fft.irfftn(sim_fourier, dim=(-2, -1)))
    return sim_image_ctf


def main(
        phi: tuple[float, float],
        theta: tuple[float, float],
        psi: tuple[float, float],
        sim_image_shape: tuple[int, int],
        sim_pixel_spacing: float
):
    n_particles = 30
    # sim_pixel_spacing = 1
    # sim_image_shape = (2048, 2048)
    defocus = -1.0  # microns
    file_path = "/Users/josh/git/torch-2d-template-matching/data/7qn5.pdb"
    # file_path = "/Users/josh/git/torch-2d-template-matching/data/4v6x-ribo.cif"
    atom_zyx = load_model(file_path, 1.0)
    print(atom_zyx.shape)
    all_particle_atom_positions = place_in_volume(
        num_particles=n_particles,
        atom_zyx=atom_zyx,
        image_shape=sim_image_shape,
        phi=phi,
        theta=theta,
        psi=psi
    )
    sim_image_no_ctf = simulate_image(
        per_particle_atom_positions=all_particle_atom_positions,
        image_shape=sim_image_shape
    )
    sim_image_final = apply_ctf(
        sim_image=sim_image_no_ctf,
        defocus=defocus,
        image_shape=sim_image_shape,
        pixel_size=sim_pixel_spacing
    )

    '''
    # view this in napari to make sure it's okay
    viewer = napari.Viewer()
    viewer.add_image(
        sim_image_final.numpy(),
        name='image',
        contrast_limits=(0, torch.max(sim_image_final))
    )
    napari.run()
    '''
    return sim_image_final


'''
if __name__ == "__main__":
    main()
'''
