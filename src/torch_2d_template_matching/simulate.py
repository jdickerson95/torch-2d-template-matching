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
        image_shape: tuple[int, int]
):
    # here I get H3 grid at a particular resolution
    h3_res = 0  # 0 to restrict space for testing
    h3_grid = get_h3_grid_at_resolution(0)
    # Take n random points on the grid
    weights = torch.ones((len(h3_grid)))  # Use even weights
    random_indices = torch.multinomial(weights, num_particles, replacement=True)  # replace so same orientation can
    # convert to rotation matrices
    rotation_matrices = torch.zeros((num_particles, 3, 3))
    for i, idx in enumerate(random_indices):
        rotation_matrices[i] = h3_to_rotation_matrix(h3_grid[idx])
    # rotate the atoms to these
    print(type(rotation_matrices))
    print(type(atom_zyx))
    rotated_atom_zyx = torch.matmul(atom_zyx, rotation_matrices)
    print(rotated_atom_zyx.shape)
    # Place these in a 3D volume
    print("placing particles in 3D volume")
    # I'm placing in z, but I am going to apply the same ctf to them all
    # so they will all have the same defocus
    volume_shape = (image_shape[0] // 3, *image_shape)
    pz, py, px = [
        np.random.uniform(low=0, high=dim_length, size=num_particles)
        for dim_length in volume_shape
    ]
    particle_positions = einops.rearrange([pz, py, px], 'zyx b -> b 1 zyx')
    particle_atom_positions = rotated_atom_zyx + particle_positions
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


def main():
    n_particles = 5
    sim_pixel_spacing = 1
    sim_image_shape = (4096, 4096)
    defocus = -1.0  # microns
    file_path = "/Users/josh/git/torch-2d-template-matching/data/4v6x-ribo.cif"
    atom_zyx = load_model(file_path, 1.0)
    print(atom_zyx.shape)
    all_particle_atom_positions = place_in_volume(n_particles, atom_zyx, sim_image_shape)
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
    # view this in napari to make sure it's okay
    viewer = napari.Viewer()
    viewer.add_image(
        sim_image_final.numpy(),
        name='image',
        contrast_limits=(0, torch.max(sim_image_final))
    )
    napari.run()


if __name__ == "__main__":
    main()
