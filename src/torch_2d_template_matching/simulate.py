# Simualating images to make testing easier
# Use a pdb/cif model to generate simulated images


import einops
import numpy as np
import torch
from so3_grid import get_h3_grid_at_resolution
from so3_grid import h3_to_rotation_matrix


def place_in_volume(
        num_particles: int,
        atom_zyx: torch.Tensor,
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
    print(rotation_matrices)

    # Place these in a 3D volume

#if __name__ == "__main__":
#    place_in_volume(5)


