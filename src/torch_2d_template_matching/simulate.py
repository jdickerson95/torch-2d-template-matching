# Simulating images to make testing easier
# Use a pdb/cif model to generate simulated images

import time
import einops
import numpy as np
import torch
import mmdf
from libtilt.interpolation import insert_into_image_2d
from libtilt.ctf.ctf_2d import calculate_ctf
from libtilt.ctf.relativistic_wavelength import calculate_relativistic_electron_wavelength
from libtilt.image_handler.doseweight_movie import dose_weight_3d_volume

from torch_2d_template_matching.so3_grid import get_h3_grid_at_resolution
from torch_2d_template_matching.so3_grid import h3_to_rotation_matrix
from eulerangles import matrix2euler
#import napari
from scipy.spatial.transform import Rotation as R
import json
from pathlib import Path
import pandas as pd
import mrcfile

import multiprocessing as mp
from functools import partial

#calcualte scattering potential
SCATTERING_PARAMS_PATH = Path(__file__).parent / "elastic_scattering_factors.json"

with open(SCATTERING_PARAMS_PATH, "r") as f:
    data = json.load(f)

SCATTERING_PARAMETERS_A = {k: v for k, v in data["parameters_a"].items() if v != []}
SCATTERING_PARAMETERS_B = {k: v for k, v in data["parameters_b"].items() if v != []}

BOND_SCALING_FACTOR = 1.043


def load_model(
        file_path: str,
        pixel_size: float,
) -> torch.Tensor:
    df = mmdf.read(file_path)
    print(list(df.columns))
    pd.set_option('display.max_columns', None)
    print(df.head(10))
    atom_zyx = torch.tensor(df[['z', 'y', 'x']].to_numpy()).float()  # (n_atoms, 3)
    atom_zyx -= torch.mean(atom_zyx, dim=0, keepdim=True)  # center
    #atom_zyx /= pixel_size
    atom_id = df['element'].str.upper().tolist() 
    atom_b_factor = torch.tensor(df['b_isotropic'].to_numpy()).float()
    return atom_zyx, atom_id, atom_b_factor #atom_zyx is in units of angstroms


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
    # rotated_atom_zyx = torch.matmul(atom_zyx, rotation_matrices)

    rotation_matrices = einops.rearrange(rotation_matrices, '... i j -> ... 1 i j')
    print(atom_zyx[0])
    atom_xyz = torch.flip(atom_zyx, [1])
    print(atom_xyz[0])
    atom_xyz = einops.rearrange(atom_xyz, 'n coords -> n coords 1')
    rotated_atom_xyz = rotation_matrices @ atom_xyz
    rotated_atom_xyz = einops.rearrange(rotated_atom_xyz, '... n coords 1 -> ... n coords')
    rotated_atom_zyx = torch.flip(rotated_atom_xyz, [1])

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
    print(f"atom pos: {per_particle_atom_positions.shape}")
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

def get_scattering_potential_of_voxel(
        zyx_coords1: torch.Tensor,  # Shape: (N, 3)
        zyx_coords2: torch.Tensor,  # Shape: (N, 3)
        bPlusB: torch.Tensor,
        atom_id: str,
        lead_term: float,
        scattering_params_a: dict  # Add parameter dictionary
):
    # Compare signs element-wise for batched coordinates
    t1 = (zyx_coords1[:, 2] * zyx_coords2[:, 2]) >= 0  # Shape: (N,)
    t2 = (zyx_coords1[:, 1] * zyx_coords2[:, 1]) >= 0  # Shape: (N,)
    t3 = (zyx_coords1[:, 0] * zyx_coords2[:, 0]) >= 0  # Shape: (N,)

    temp_potential = torch.zeros(len(zyx_coords1))

    for i, bb in enumerate(bPlusB):
        # Handle z dimension
        x_term = torch.where(
            t1,
            torch.special.erf(bb * zyx_coords2[:, 2]) - torch.special.erf(bb * zyx_coords1[:, 2]),
            torch.abs(torch.special.erf(bb * zyx_coords2[:, 2])) + torch.abs(torch.special.erf(bb * zyx_coords1[:, 2]))
        )
        
        # Handle y dimension
        y_term = torch.where(
            t2,
            torch.special.erf(bb * zyx_coords2[:, 1]) - torch.special.erf(bb * zyx_coords1[:, 1]),
            torch.abs(torch.special.erf(bb * zyx_coords2[:, 1])) + torch.abs(torch.special.erf(bb * zyx_coords1[:, 1]))
        )
        
        # Handle x dimension
        z_term = torch.where(
            t3,
            torch.special.erf(bb * zyx_coords2[:, 0]) - torch.special.erf(bb * zyx_coords1[:, 0]),
            torch.abs(torch.special.erf(bb * zyx_coords2[:, 0])) + torch.abs(torch.special.erf(bb * zyx_coords1[:, 0]))
        )

        t0 = z_term * y_term * x_term
        temp_potential += scattering_params_a[atom_id][i] * torch.abs(t0)

    return lead_term * temp_potential

def get_scattering_potential_of_voxel_old(
        zyx_coords1: torch.Tensor,  # Shape: (3,)
        zyx_coords2: torch.Tensor,  # Shape: (3,)
        bPlusB: torch.Tensor,
        atom_id: str,
        lead_term: float
):
    # Compare signs for individual coordinates
    t1 = (zyx_coords1[2] * zyx_coords2[2]) >= 0
    t2 = (zyx_coords1[1] * zyx_coords2[1]) >= 0
    t3 = (zyx_coords1[0] * zyx_coords2[0]) >= 0

    temp_potential = 0.0
    for i, bb in enumerate(bPlusB):
        # Handle z dimension
        if t1:
            z_term = torch.special.erf(bb * zyx_coords2[2]) - torch.special.erf(bb * zyx_coords1[2])
        else:
            z_term = torch.abs(torch.special.erf(bb * zyx_coords2[2])) + torch.abs(torch.special.erf(bb * zyx_coords1[2]))
        
        # Handle y dimension
        if t2:
            y_term = torch.special.erf(bb * zyx_coords2[1]) - torch.special.erf(bb * zyx_coords1[1])
        else:
            y_term = torch.abs(torch.special.erf(bb * zyx_coords2[1])) + torch.abs(torch.special.erf(bb * zyx_coords1[1]))
        
        # Handle x dimension
        if t3:
            x_term = torch.special.erf(bb * zyx_coords2[0]) - torch.special.erf(bb * zyx_coords1[0])
        else:
            x_term = torch.abs(torch.special.erf(bb * zyx_coords2[0])) + torch.abs(torch.special.erf(bb * zyx_coords1[0]))

        t0 = z_term * y_term * x_term
        temp_potential += SCATTERING_PARAMETERS_A[atom_id][i] * torch.abs(t0)

    return lead_term * temp_potential

def GetPixelRange(atom_id: str, bPlusB: torch.Tensor, upsampled_pixel_size: float, lead_term: float, cutoff_percent: float) -> int:
    pix_idx = 0
    zyx_coords1 = torch.tensor([(pix_idx*upsampled_pixel_size)-(upsampled_pixel_size/2),(pix_idx*upsampled_pixel_size)-(upsampled_pixel_size/2),(pix_idx*upsampled_pixel_size)-(upsampled_pixel_size/2)])
    zyx_coords2 = torch.tensor([(pix_idx*upsampled_pixel_size)+(upsampled_pixel_size/2),(pix_idx*upsampled_pixel_size)+(upsampled_pixel_size/2),(pix_idx*upsampled_pixel_size)+(upsampled_pixel_size/2)])
    max_vox_potential = get_scattering_potential_of_voxel(zyx_coords1, zyx_coords2, bPlusB, atom_id, lead_term)
    this_vox_potential = max_vox_potential
    cutoff_value = max_vox_potential * cutoff_percent
    while this_vox_potential > cutoff_value:
        pix_idx += 1
        zyx_coords1 = torch.tensor([(pix_idx*upsampled_pixel_size)-(upsampled_pixel_size/2),(pix_idx*upsampled_pixel_size)-(upsampled_pixel_size/2),(pix_idx*upsampled_pixel_size)-(upsampled_pixel_size/2)])   
        zyx_coords2 = torch.tensor([(pix_idx*upsampled_pixel_size)+(upsampled_pixel_size/2),(pix_idx*upsampled_pixel_size)+(upsampled_pixel_size/2),(pix_idx*upsampled_pixel_size)+(upsampled_pixel_size/2)])
        this_vox_potential = get_scattering_potential_of_voxel(zyx_coords1, zyx_coords2, bPlusB, atom_id, lead_term)
    return pix_idx

    # My Thought for the sampling was was % drop in scattering factor with distance
def get_size_neighborhood_percent(atoms_id, max_b_factor, upsampled_pixel_size, lead_term):
    unique_atom_types = atoms_id.unique()
    #Get the bplusB param for each atom type
    bPlusB_max = torch.zeros((len(unique_atom_types), 5))
    cutoff_percent = 0.1 #percentage of max voxel value to use for cutoff
    
    max_px_index = 0
    for i, atom_id in enumerate(unique_atom_types):
        for j in range(5):
            bPlusB_max[i,j] = 2 * torch.pi / (max_b_factor + SCATTERING_PARAMETERS_B[atom_id][j])
        pix_idx = GetPixelRange(atom_id, bPlusB_max[i], upsampled_pixel_size, lead_term, cutoff_percent)
        if i == 0:
            max_px_index = pix_idx
        else:
            max_px_index = max(max_px_index, pix_idx)
        
    #print(f"max_px_index: {max_px_index}")
    return max_px_index

def get_size_neighborhood_cistem(mean_b_factor, upsampled_pixel_size):
    return 1 + torch.round((0.4 * (0.6 * mean_b_factor)**0.5 + 0.2) / upsampled_pixel_size) # This is the size of the neighborhood in pixels

def process_atom_batch(batch_args):
    try:
        # Unpack the tuple correctly
        (atom_indices_batch, bPlusB_batch, atoms_id_filtered_batch, 
         voxel_offsets_flat, upsampled_shape, upsampled_pixel_size, lead_term,
         scattering_params_a) = batch_args 
    
        # Move tensors to CPU and ensure they're contiguous
        atom_indices_batch = atom_indices_batch.cpu().contiguous()
        voxel_offsets_flat = voxel_offsets_flat.cpu().contiguous()
        
        # Initialize local volume grid for this batch
        local_volume = torch.zeros(upsampled_shape, device='cpu')
        
        # Add debug print to verify data
        print(f"Processing batch of size {len(atom_indices_batch)}")
        
        # Process each atom in the batch
        for i in range(len(atom_indices_batch)):
            atom_pos = atom_indices_batch[i]
            atom_id = atoms_id_filtered_batch[i]  # This should now work with list indexing
            
            # Calculate voxel positions relative to atom center
            voxel_positions = atom_pos.view(1, 3) + voxel_offsets_flat
            
            #print(voxel_positions.shape)
            # Check bounds for each dimension separately
            valid_z = (voxel_positions[:, 0] >= 0) & (voxel_positions[:, 0] < upsampled_shape[0])
            valid_y = (voxel_positions[:, 1] >= 0) & (voxel_positions[:, 1] < upsampled_shape[1])
            valid_x = (voxel_positions[:, 2] >= 0) & (voxel_positions[:, 2] < upsampled_shape[2])
            valid_mask = valid_z & valid_y & valid_x
            
            if valid_mask.any():
            # Calculate coordinates relative to atom center for potential calculation
                relative_coords = ((voxel_positions[valid_mask] - atom_pos) * upsampled_pixel_size)
                coords1 = relative_coords - (upsampled_pixel_size/2)
                coords2 = relative_coords + (upsampled_pixel_size/2)
            
            # Calculate potentials for valid positions
                potentials = get_scattering_potential_of_voxel(
                    coords1,
                    coords2,
                    bPlusB_batch[i],
                    atom_id,
                    lead_term,
                    scattering_params_a  # Pass the parameters
            )
                
                # Get valid voxel positions
                valid_positions = voxel_positions[valid_mask].long()
                
                # Update local volume
                local_volume[valid_positions[:, 0], 
                             valid_positions[:, 1], 
                             valid_positions[:, 2]] += potentials
    except Exception as e:
        print(f"Error in process_atom_batch: {str(e)}")
        raise e
            
    return local_volume

def process_atoms_parallel(atom_indices, bPlusB, atoms_id_filtered, voxel_offsets_flat, upsampled_shape, upsampled_pixel_size, lead_term, n_cores):
    # Ensure all inputs are on CPU and contiguous
    atom_indices = atom_indices.cpu().contiguous()
    voxel_offsets_flat = voxel_offsets_flat.cpu().contiguous()

    # Convert pandas Series to list if necessary
    if hasattr(atoms_id_filtered, 'tolist'):
        atoms_id_filtered = atoms_id_filtered.tolist()
    
    num_atoms = len(atom_indices)
    batch_size = max(1, num_atoms // (n_cores * 4))  # Divide work into smaller batches
    
    print(f"Processing {num_atoms} atoms in batches of {batch_size}")
    
    # Prepare batches
    batches = []
    for start_idx in range(0, num_atoms, batch_size):
        end_idx = min(start_idx + batch_size, num_atoms)
        batch_args = (
            atom_indices[start_idx:end_idx],
            bPlusB[start_idx:end_idx],
            atoms_id_filtered[start_idx:end_idx],
            voxel_offsets_flat,
            upsampled_shape,
            upsampled_pixel_size,
            lead_term,
            SCATTERING_PARAMETERS_A  
        )
        batches.append(batch_args)
    
    # Process batches in parallel
    with mp.Pool(n_cores) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(process_atom_batch, batches)):
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"Processed {(i + 1) * batch_size} atoms of {num_atoms}")
    
    # Combine results
    final_volume = torch.zeros(upsampled_shape, device='cpu')
    for result in results:
        final_volume += result
    
    return final_volume


def simulate_3d_volume(
        pdb_filename: str,
        sim_volume_shape: tuple[int, int, int],
        sim_pixel_spacing: float,
        n_cpu_cores: int
):
    start_time = time.time()

    atoms_zyx, atoms_id, atoms_b_factor = load_model(pdb_filename, sim_pixel_spacing)

    b_scaling = 1
    added_B = 10.0
    atoms_b_factor_scaled = 0.25 * (atoms_b_factor * b_scaling + added_B) # I'm not sure what the 0.25 is
    mean_b_factor = torch.mean(atoms_b_factor_scaled)
    print(f"mean_b_factor: {mean_b_factor}")
    max_b_factor = torch.max(atoms_b_factor_scaled)

    beam_energy = 300000 # eV
    wavelength = calculate_relativistic_electron_wavelength(beam_energy) # meters
    wavelength_A = wavelength * 1e10

    dose_weighting = True
    num_frames = 30
    flux = 1
    dose_B = -1
    apply_dqe = False

    #setup a grid with half the pixel size and double desired shape
    upsampling = 1
    upsampled_pixel_size = sim_pixel_spacing / upsampling
    upsampled_shape = tuple(np.array(sim_volume_shape) * upsampling)
    volume_grid = torch.zeros(upsampled_shape)

    #I'm making sure the index is on the edge of a voxel
    origin_idx = (int(upsampled_shape[0] // 2), int(upsampled_shape[1] // 2), int(upsampled_shape[2] // 2))

    lead_term = BOND_SCALING_FACTOR * wavelength_A / 8.0 / upsampled_pixel_size / upsampled_pixel_size
    #Not sure where the 8 comes from here

    # CisTEM does it like this though... I will use this for now
    size_neighborhood = get_size_neighborhood_cistem(mean_b_factor, upsampled_pixel_size)
    neighborhood_range = torch.arange(-size_neighborhood, size_neighborhood+1) #pixels to do either side
    print(f"size_neighborhood: {size_neighborhood}")

    # I will also try supersampling with the more traditional approach and compare... 

    # Filter out hydrogen atoms
    non_h_mask = [id != "H" for id in atoms_id]
    atoms_zyx_filtered = atoms_zyx[non_h_mask]
    atoms_id_filtered = [id for i, id in enumerate(atoms_id) if non_h_mask[i]]
    atoms_b_factor_scaled_filtered = atoms_b_factor_scaled[non_h_mask]

    # Calculate B-factors for all atoms at once
    b_params = torch.stack([torch.tensor(SCATTERING_PARAMETERS_B[atom_id]) for atom_id in atoms_id_filtered])  # Shape: (n_atoms, 5)
    bPlusB = 2 * torch.pi / torch.sqrt(atoms_b_factor_scaled_filtered.unsqueeze(1) + b_params)  # Shape: (n_atoms, 5)

    # Calculate atom indices in volume
    # Convert from centered Angstroms to voxel coordinates
    atom_indices = torch.zeros_like(atoms_zyx_filtered)
    atom_indices[:, 0] = (atoms_zyx_filtered[:, 0] / upsampled_pixel_size) + origin_idx[0]  # z
    atom_indices[:, 1] = (atoms_zyx_filtered[:, 1] / upsampled_pixel_size) + origin_idx[1]  # y
    atom_indices[:, 2] = (atoms_zyx_filtered[:, 2] / upsampled_pixel_size) + origin_idx[2]  # x
    atom_indices = torch.round(atom_indices).int() # this is again the edge of the voxel

    # Create coordinate grids for the neighborhood
    sz, sy, sx = torch.meshgrid(neighborhood_range, neighborhood_range, neighborhood_range, indexing='ij')
    
    # Calculate relative coordinates for the voxel corners
    # Stack and flatten differently to ensure we maintain 3D structure
    voxel_offsets = torch.stack([sz, sy, sx])  # (3, n, n, n)
    # Flatten while preserving the relative positions
    voxel_offsets_flat = voxel_offsets.reshape(3, -1).T  # (n^3, 3)

    volume_grid = process_atoms_parallel(
    atom_indices=atom_indices,
    bPlusB=bPlusB,
    atoms_id_filtered=atoms_id_filtered,
    voxel_offsets_flat=voxel_offsets_flat,
    upsampled_shape=upsampled_shape,
    upsampled_pixel_size=upsampled_pixel_size,
    lead_term=lead_term,
    n_cores=n_cpu_cores  # Use the existing n_cpu_cores variable
)
    #Undo the upsanmpling here
    # Bin the volume back to original size using 3D average pooling
    volume_grid_binned = torch.nn.functional.avg_pool3d(
        volume_grid.unsqueeze(0),  # Add batch dimension
        kernel_size=upsampling,
        stride=upsampling
    ).squeeze(0)  # Remove batch dimension

    # I would then do any moodifications to volume grid here
    # apply a dose filter if specified 
    if dose_weighting:
        volume_grid_binned = dose_weight_3d_volume(
            volume=volume_grid_binned,
            num_frames=num_frames,
            pixel_size=sim_pixel_spacing,
            flux=flux,
            Bfac=dose_B
    )

    # Apply a DQE if specified 

    # and finally save it as an mrc file
    mrcfile.write("/Users/josh/git/2dtm_tests/simulator/simulated_volume_dw_us1.mrc", volume_grid_binned.numpy(), overwrite=True)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total simulation time: {minutes} minutes {seconds} seconds")


def main(
        phi: tuple[float, float],
        theta: tuple[float, float],
        psi: tuple[float, float],
        sim_image_shape: tuple[int, int],
        sim_pixel_spacing: float
):
    n_particles = 20
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
