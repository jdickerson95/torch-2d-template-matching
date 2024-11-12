# Simulating images to make testing easier
# Use a pdb/cif model to generate simulated images

import time
import einops
import numpy as np
import torch
import math
import mmdf
from libtilt.interpolation import insert_into_image_2d
from libtilt.ctf.ctf_2d import calculate_ctf
from libtilt.ctf.relativistic_wavelength import calculate_relativistic_electron_wavelength
from libtilt.image_handler.doseweight_movie import dose_weight_3d_volume
from libtilt.image_handler.doseweight_movie import cumulative_dose_filter_3d
from libtilt.grids import fftfreq_grid
#from torch_fourier_rescale.fourier_rescale_3d import fourier_rescale_3d
#from torch_fourier_rescale.fourier_rescale_3d import fourier_rescale_3d_force_size

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

# Assuming a perfect counting - so no read noise, and no coincednce loss. Then we have a flat NPS and the DQE is just DQE(0)*MTF^2
# The parameters below are for a 5 gaussian fit to 300 KeV , 2.5 EPS from Ruskin et al. with DQE(0) = 0.791

DQE_PARAMETERS_A = [-0.01516, -0.5662, -0.09731, -0.01551, 21.47]

DQE_PARAMETERS_B = [0.02671, -0.02504, 0.162, 0.2831, -2.28]

DQE_PARAMETERS_C = [0.01774, 0.1441, 0.1082, 0.07916, 1.372]

cisTEM_offset = 0.5

def fourier_rescale_3d_force_size(
    volume: torch.Tensor,
    target_size: float,
) -> torch.Tensor:
    """
    Crop a 3D Fourier-transformed volume to an even target size while maintaining symmetry using rfft.
    
    Parameters:
    - volume (torch.Tensor): The input volume in real space of shape (D, H, W).
    - target_size (int): The desired even size to crop in Fourier space for each dimension.
    
    Returns:
    - cropped_volume (torch.Tensor): The cropped volume back in real space with dimensions (target_size, target_size, target_size).
    """
    # Ensure the target size is even
    assert target_size % 2 == 0, "Target size must be even."
    
    # Get the original size of the volume
    original_size = volume.shape[0]  # Assumes volume is cubic for simplicity
    assert volume.shape[0] == volume.shape[1] == volume.shape[2], "Volume must be cubic."
    
    # Step 1: Perform real-to-complex Fourier Transform (rfftn) and shift the zero frequency to the center
    volume = torch.fft.fftshift(volume, dim=(-3, -2, -1))
    volume_fft = torch.fft.rfftn(volume, dim=(-3, -2, -1))
    volume_fft_shifted = torch.fft.fftshift(volume_fft, dim=(-3, -2, -1))  # Shift along first two dimensions only

    # Calculate the dimensions of the rfftn output
    rfft_size_z, rfft_size_y, rfft_size_x = volume_fft_shifted.shape

    # Calculate cropping indices for each dimension
    center_z = rfft_size_z // 2
    center_y = rfft_size_y // 2

    # Define the cropping ranges
    crop_start_z = center_z - target_size // 2
    crop_end_z = center_z + target_size // 2
    crop_start_y = center_y - target_size // 2
    crop_end_y = center_y + target_size // 2
    crop_start_x = 0  # Start from the beginning in the last dimension (low frequencies)
    crop_end_x = target_size // 2 + 1  # Crop from the high-frequency end only along the last dimension

    # Step 2: Crop the Fourier-transformed volume
    cropped_fft = volume_fft_shifted[crop_start_z:crop_end_z, crop_start_y:crop_end_y, -crop_end_x:]
    
    
    # Step 3: Inverse shift and apply the inverse rFFT to return to real space
    cropped_fft_shifted_back = torch.fft.ifftshift(cropped_fft, dim=(-3, -2))
    cropped_volume = torch.fft.irfftn(cropped_fft_shifted_back, s=(target_size, target_size, target_size), dim=(-3, -2, -1))
    cropped_volume = torch.fft.ifftshift(cropped_volume, dim=(-3, -2, -1))

    return cropped_volume

#Apply DQE function, this should really go in libtilt filters
#Can specify if this is a FFT or image being passed in
def apply_dqe_volume(volume, dqe_parameters_a, dqe_parameters_b, dqe_parameters_c, pixel_size: float):
    
    dft_volume = torch.fft.rfftn(volume, dim=(-3, -2, -1))
    fft_freq_px = (
        fftfreq_grid(
            image_shape=volume.shape,
            rfft=True,
            fftshift=False,
            norm=True,
        )
        / pixel_size
    )

    weight = 0
    for i in range(5):
        weight += (dqe_parameters_a[i] * torch.exp(-1.0 * torch.pow((fft_freq_px - dqe_parameters_b[i])/dqe_parameters_c[i], 2)))

    dft_volume *= weight
    volume = torch.fft.irfftn(dft_volume, dim=(-3, -2, -1))
    return volume

def load_model(
        file_path: str,
        pixel_size: float,
) -> torch.Tensor:
    df = mmdf.read(file_path)
    #print(list(df.columns))
    pd.set_option('display.max_columns', None)
    #print(df.head(10))
    atom_zyx = torch.tensor(df[['z', 'y', 'x']].to_numpy()).float()  # (n_atoms, 3)
    atom_zyx -= torch.mean(atom_zyx, dim=0, keepdim=True)  # center
    #atom_zyx /= pixel_size
    atom_id = df['element'].str.upper().tolist() 
    atom_b_factor = torch.tensor(df['b_isotropic'].to_numpy()).float()
    return atom_zyx, atom_id, atom_b_factor #atom_zyx is in units of angstroms

def select_gpus(gpu_ids: list[int] = None, num_gpus: int = 1) -> list[torch.device]:
    """
    Select multiple GPU devices based on IDs or available memory.
    
    Args:
        gpu_ids: List of specific GPU IDs to use. If None, selects GPUs with most available memory.
        num_gpus: Number of GPUs to use if gpu_ids is None.
    
    Returns:
        list[torch.device]: Selected GPU devices or [CPU] if no GPU available
    """
    if not torch.cuda.is_available():
        print("No GPU available, using CPU")
        return [torch.device("cpu")]
        
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("No GPU available, using CPU")
        return [torch.device("cpu")]
    
    # If specific GPUs requested, validate and return them
    if gpu_ids is not None:
        valid_devices = []
        for gpu_id in gpu_ids:
            if gpu_id >= n_gpus:
                print(f"Requested GPU {gpu_id} not available. Max GPU ID is {n_gpus-1}")
                continue
            valid_devices.append(torch.device(f"cuda:{gpu_id}"))
        
        if not valid_devices:
            print("No valid GPUs specified. Using CPU")
            return [torch.device("cpu")]
        return valid_devices
    
    # Find GPUs with most available memory
    gpu_memory_available = []
    print("\nAvailable GPUs:")
    for i in range(n_gpus):
        torch.cuda.set_device(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory
        allocated_memory = torch.cuda.memory_allocated(i)
        available = total_memory - allocated_memory
        
        print(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")
        print(f"  Total memory: {total_memory/1024**3:.1f} GB")
        print(f"  Available memory: {available/1024**3:.1f} GB")
        
        gpu_memory_available.append((i, available))
    
    # Sort by available memory and select the top num_gpus
    gpu_memory_available.sort(key=lambda x: x[1], reverse=True)
    selected_gpus = [torch.device(f"cuda:{idx}") for idx, _ in gpu_memory_available[:num_gpus]]
    
    print("\nSelected GPUs:", [str(device) for device in selected_gpus])
    return selected_gpus
    
def calculate_batch_size(
        total_atoms: int,
        neighborhood_size: int,
        device: torch.device,
        safety_factor: float = 0.8,  # Use only 80% of available memory by default
        min_batch_size: int = 100
) -> int:
    """
    Calculate optimal batch size based on available GPU memory and data size.
    
    Args:
        total_atoms: Total number of atoms to process
        neighborhood_size: Size of neighborhood around each atom
        device: PyTorch device (GPU/CPU)
        safety_factor: Fraction of available memory to use (0.0 to 1.0)
    
    Returns:
        Optimal batch size
    """
    if device.type == 'cpu':
        return min(1000, total_atoms)
        
    # Get available GPU memory in bytes
    gpu_memory = torch.cuda.get_device_properties(device).total_memory
    #gpu_memory_available = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    
    # Calculate memory requirements per atom
    voxels_per_atom = (2 * neighborhood_size + 1) ** 3
    bytes_per_float = 4  # 32-bit float
    
    # Memory needed for:
    # 1. Voxel positions (float32): batch_size * voxels_per_atom * 3 coordinates
    # 2. Valid mask (bool): batch_size * voxels_per_atom
    # 3. Relative coordinates (float32): batch_size * voxels_per_atom * 3
    # 4. Potentials (float32): batch_size * voxels_per_atom
    # Plus some overhead for temporary variables
    memory_per_atom = (
        voxels_per_atom * (3 * bytes_per_float)  # Voxel positions
        + voxels_per_atom * 1  # Valid mask (bool)
        + voxels_per_atom * (3 * bytes_per_float)  # Relative coordinates
        + voxels_per_atom * bytes_per_float # Potentials
        + 1024                              # Additional overhead
    )
    
    # Calculate batch size
    optimal_batch_size = int((gpu_memory * safety_factor) / memory_per_atom)
    
    # Ensure batch size is at least 1 but not larger than total atoms
    optimal_batch_size = max(min_batch_size, min(optimal_batch_size, total_atoms))
    
    print(f"Total GPU memory: {gpu_memory / 1024**3:.2f} GB")
    #print(f"Available GPU memory: {gpu_memory_available / 1024**3:.2f} GB")
    print(f"Estimated memory per atom: {memory_per_atom / 1024**2:.2f} MB")
    print(f"Optimal batch size: {optimal_batch_size}")
    
    return optimal_batch_size


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
        scattering_params_a: dict,  # Add parameter dictionary
        device: torch.device = None
):
    """
    Calculate scattering potential for a voxel.
    """
    # If device not specified, use the device of input tensors
    if device is None:
        device = zyx_coords1.device

    # Get scattering parameters for this atom type and move to correct device
    # Convert parameters to tensor and move to device
    if isinstance(scattering_params_a[atom_id], torch.Tensor):
        a_params = scattering_params_a[atom_id].clone().detach().to(device)
    else:
        a_params = torch.as_tensor(scattering_params_a[atom_id], device=device)
    
    # Compare signs element-wise for batched coordinates
    t1 = (zyx_coords1[:, 2] * zyx_coords2[:, 2]) >= 0  # Shape: (N,)
    t2 = (zyx_coords1[:, 1] * zyx_coords2[:, 1]) >= 0  # Shape: (N,)
    t3 = (zyx_coords1[:, 0] * zyx_coords2[:, 0]) >= 0  # Shape: (N,)

    temp_potential = torch.zeros(len(zyx_coords1), device=device)

    for i, bb in enumerate(bPlusB):
        a = a_params[i]
        # Handle x dimension
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
        
        # Handle z dimension
        z_term = torch.where(
            t3,
            torch.special.erf(bb * zyx_coords2[:, 0]) - torch.special.erf(bb * zyx_coords1[:, 0]),
            torch.abs(torch.special.erf(bb * zyx_coords2[:, 0])) + torch.abs(torch.special.erf(bb * zyx_coords1[:, 0]))
        )

        t0 = z_term * y_term * x_term
        temp_potential += a * torch.abs(t0)

    return lead_term * temp_potential

def get_scattering_potential_of_voxel_batched(
        zyx_coords1: torch.Tensor,  # Shape: [N, 3]
        zyx_coords2: torch.Tensor,  # Shape: [N, 3]
        bPlusB: torch.Tensor,      # Shape: [1, 5]
        atom_id: str,
        lead_term: float,
        scattering_params_a: dict,
        device: torch.device
):
    """
    Vectorized version of scattering potential calculation
    """
    # Get scattering parameters
    a_params = scattering_params_a[atom_id]
    
    # Calculate signs for all coordinates at once
    t1 = (zyx_coords1[:, 2] * zyx_coords2[:, 2]) >= 0
    t2 = (zyx_coords1[:, 1] * zyx_coords2[:, 1]) >= 0
    t3 = (zyx_coords1[:, 0] * zyx_coords2[:, 0]) >= 0

    # Initialize potentials
    temp_potential = torch.zeros(len(zyx_coords1), device=device)
    
    # Calculate all error functions at once
    for i, a in enumerate(a_params):
        bb = bPlusB[0, i]  # Single value for this parameter
        
        # Calculate terms
        x_term = torch.where(
            t1,
            torch.special.erf(bb * zyx_coords2[:, 2]) - torch.special.erf(bb * zyx_coords1[:, 2]),
            torch.abs(torch.special.erf(bb * zyx_coords2[:, 2])) + torch.abs(torch.special.erf(bb * zyx_coords1[:, 2]))
        )
        
        y_term = torch.where(
            t2,
            torch.special.erf(bb * zyx_coords2[:, 1]) - torch.special.erf(bb * zyx_coords1[:, 1]),
            torch.abs(torch.special.erf(bb * zyx_coords2[:, 1])) + torch.abs(torch.special.erf(bb * zyx_coords1[:, 1]))
        )
        
        z_term = torch.where(
            t3,
            torch.special.erf(bb * zyx_coords2[:, 0]) - torch.special.erf(bb * zyx_coords1[:, 0]),
            torch.abs(torch.special.erf(bb * zyx_coords2[:, 0])) + torch.abs(torch.special.erf(bb * zyx_coords1[:, 0]))
        )

        t0 = z_term * y_term * x_term
        temp_potential = temp_potential + (a * torch.abs(t0))

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
        (atom_indices_batch, atom_dds_batch, bPlusB_batch, atoms_id_filtered_batch, 
         voxel_offsets_flat, upsampled_shape, upsampled_pixel_size, lead_term,
         scattering_params_a) = batch_args 
    
        # Move tensors to CPU and ensure they're contiguous
        atom_indices_batch = atom_indices_batch.cpu().contiguous()
        atom_dds_batch = atom_dds_batch.cpu().contiguous()
        voxel_offsets_flat = voxel_offsets_flat.cpu().contiguous()
        
        # Initialize local volume grid for this batch
        local_volume = torch.zeros(upsampled_shape, device='cpu')
        
        # Add debug print to verify data
        print(f"Processing batch of size {len(atom_indices_batch)}")
        
        #offset_test = upsampled_pixel_size/2
        # Process each atom in the batch
        for i in range(len(atom_indices_batch)):
            atom_pos = atom_indices_batch[i]
            atom_dds = atom_dds_batch[i]
            atom_id = atoms_id_filtered_batch[i]  # This should now work with list indexing
            
            # Calculate voxel positions relative to atom center
            voxel_positions = atom_pos.view(1, 3) + voxel_offsets_flat #indX/Y/Z equivalent
            
            #print(voxel_positions.shape)
            # Check bounds for each dimension separately
            valid_z = (voxel_positions[:, 0] >= 0) & (voxel_positions[:, 0] < upsampled_shape[0])
            valid_y = (voxel_positions[:, 1] >= 0) & (voxel_positions[:, 1] < upsampled_shape[1])
            valid_x = (voxel_positions[:, 2] >= 0) & (voxel_positions[:, 2] < upsampled_shape[2])
            valid_mask = valid_z & valid_y & valid_x
            
            if valid_mask.any():
            # Calculate coordinates relative to atom center for potential calculation
                relative_coords = ((voxel_positions[valid_mask] - atom_pos - atom_dds - cisTEM_offset) * upsampled_pixel_size)
                coords1 = relative_coords
                coords2 = relative_coords + upsampled_pixel_size
            
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

def process_atoms_parallel(atom_indices, atom_dds, bPlusB, atoms_id_filtered, voxel_offsets_flat, upsampled_shape, upsampled_pixel_size, lead_term, n_cores):
    # Ensure all inputs are on CPU and contiguous
    atom_indices = atom_indices.cpu().contiguous()
    atom_dds = atom_dds.cpu().contiguous()
    voxel_offsets_flat = voxel_offsets_flat.cpu().contiguous()

    # Convert pandas Series to list if necessary
    if hasattr(atoms_id_filtered, 'tolist'):
        atoms_id_filtered = atoms_id_filtered.tolist()
    
    num_atoms = len(atom_indices)
    batch_size = max(1, num_atoms // (n_cores))  # Divide work into smaller batches
    
    print(f"Processing {num_atoms} atoms in batches of {batch_size}")
    
    # Prepare batches
    batches = []
    for start_idx in range(0, num_atoms, batch_size):
        end_idx = min(start_idx + batch_size, num_atoms)
        batch_args = (
            atom_indices[start_idx:end_idx],
            atom_dds[start_idx:end_idx],
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

def process_atoms_gpu(
        atom_indices: torch.Tensor,
        bPlusB: torch.Tensor,
        atoms_id_filtered: list,
        voxel_offsets_flat: torch.Tensor,
        upsampled_shape: tuple,
        upsampled_pixel_size: float,
        lead_term: float,
        device: torch.device
):
    
    # Pre-compute scattering parameters for all atom types in batch
    unique_atom_types = list(set(atoms_id_filtered))
    scattering_params = {atom_type: torch.tensor(SCATTERING_PARAMETERS_A[atom_type], device=device) 
                        for atom_type in unique_atom_types}
    
    # Clear CUDA cache before calculating batch size
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    batch_size = calculate_batch_size(
        total_atoms=len(atom_indices),
        neighborhood_size=voxel_offsets_flat.shape[0] // 3,  # Size of one dimension of neighborhood
        device=device
    )
    # Initialize volume grid on GPU
    volume_grid = torch.zeros(upsampled_shape, device=device)



    try:
        for start_idx in range(0, len(atom_indices), batch_size):
            end_idx = min(start_idx + batch_size, len(atom_indices))
            
            # Get batch data
            atom_pos_batch = atom_indices[start_idx:end_idx]
            atom_id_batch = atoms_id_filtered[start_idx:end_idx]
            bPlusB_batch = bPlusB[start_idx:end_idx]
            
            # Calculate voxel positions for all atoms in batch
            voxel_positions = atom_pos_batch.unsqueeze(1) + voxel_offsets_flat.to(device)
            
            # Check bounds for each dimension
            valid_z = (voxel_positions[..., 0] >= 0) & (voxel_positions[..., 0] < upsampled_shape[0])
            valid_y = (voxel_positions[..., 1] >= 0) & (voxel_positions[..., 1] < upsampled_shape[1])
            valid_x = (voxel_positions[..., 2] >= 0) & (voxel_positions[..., 2] < upsampled_shape[2])
            valid_mask = valid_z & valid_y & valid_x

            # Process all valid positions for the batch at once
            valid_positions_all = []
            potentials_all = []
            
            for i, (atom_pos, atom_id, valid) in enumerate(zip(atom_pos_batch, atom_id_batch, valid_mask)):
                if valid.any():
                    # Calculate coordinates relative to atom center
                    relative_coords = ((voxel_positions[i][valid] - atom_pos) * upsampled_pixel_size)
                    coords1 = relative_coords - (upsampled_pixel_size/2)
                    coords2 = relative_coords + (upsampled_pixel_size/2)
                    
                    # Calculate potentials
                    potentials = get_scattering_potential_of_voxel(
                            coords1,
                            coords2,
                            bPlusB_batch[i],
                            atom_id,
                            lead_term,
                            scattering_params,  # Pass pre-computed params
                            device=device
                    )
                
                    # Update volume grid
                    valid_positions = voxel_positions[i][valid].long()
                    valid_positions_all.append(valid_positions)
                    potentials_all.append(potentials)

            
            # Batch update volume grid
            if valid_positions_all:
                valid_positions_cat = torch.cat(valid_positions_all, dim=0)
                potentials_cat = torch.cat(potentials_all, dim=0)
                volume_grid.index_put_((valid_positions_cat[:, 0], 
                                      valid_positions_cat[:, 1], 
                                      valid_positions_cat[:, 2]),
                                     potentials_cat,
                                     accumulate=True)

        
            # progress update
            if (start_idx + batch_size) % (batch_size * 5) == 0:
                print(f"Processed {start_idx + batch_size} atoms of {len(atom_indices)}")
    except Exception as e:
        print(f"Error during GPU processing: {str(e)}")
        raise e
    
    return volume_grid

def process_atoms_gpu2(
        atom_indices: torch.Tensor,
        atom_dds: torch.Tensor,
        bPlusB: torch.Tensor,
        atoms_id_filtered: list,
        voxel_offsets_flat: torch.Tensor,
        upsampled_shape: tuple,
        upsampled_pixel_size: float,
        lead_term: float,
        device: torch.device
):
    # Ensure consistent dtype
    atom_indices = atom_indices.to(torch.float32)
    atom_dds = atom_dds.to(torch.float32)
    bPlusB = bPlusB.to(torch.float32)
    # Pre-compute scattering parameters
    unique_atom_types = list(set(atoms_id_filtered))
    scattering_params = {
        atom_type: torch.as_tensor(SCATTERING_PARAMETERS_A[atom_type], device=device)
        for atom_type in unique_atom_types
    }

    # Initialize volume grid
    volume_grid = torch.zeros(upsampled_shape, dtype=torch.float32, device=device)

    # Calculate optimal batch size based on GPU memory
    total_voxels = voxel_offsets_flat.shape[0]
    batch_size = calculate_batch_size(
        total_atoms=len(atom_indices),
        neighborhood_size=voxel_offsets_flat.shape[0] // 3,  # Size of one dimension of neighborhood
        device=device
    ) # Adjust this based on available GPU memory
    
    print(f"Processing with batch size: {batch_size}")

    for start_idx in range(0, len(atom_indices), batch_size):
        end_idx = min(start_idx + batch_size, len(atom_indices))
        current_batch_size = end_idx - start_idx
        
        # Get batch data
        batch_indices = atom_indices[start_idx:end_idx].to(device)
        batch_atom_dds = atom_dds[start_idx:end_idx].to(device)
        batch_bPlusB = bPlusB[start_idx:end_idx].to(device)
        batch_atoms_id = atoms_id_filtered[start_idx:end_idx]

        # Process each atom type in parallel
        for atom_type in unique_atom_types:
            # Get indices for this atom type in the batch
            type_mask = torch.tensor([aid == atom_type for aid in batch_atoms_id], device=device)
            if not type_mask.any():
                continue

            type_indices = torch.where(type_mask)[0]
            n_atoms_of_type = len(type_indices)
            
            if n_atoms_of_type == 0:
                continue

            # Calculate positions for all atoms of this type at once
            atom_positions = batch_indices[type_indices]  # Shape: [n_atoms, 3]
            atom_pos_dds = batch_atom_dds[type_indices]
            
            # Reshape for broadcasting
            atom_positions = atom_positions.view(n_atoms_of_type, 1, 3)  # Shape: [n_atoms, 1, 3]
            atom_pos_dds = atom_pos_dds.view(n_atoms_of_type, 1, 3)  # Shape: [n_atoms, 1, 3]
            offsets = voxel_offsets_flat.to(device).view(1, -1, 3)  # Shape: [1, n_voxels, 3]
            
            # Calculate all voxel positions at once
            voxel_positions = atom_positions + offsets  # Shape: [n_atoms, n_voxels, 3]
            
            # Check bounds
            valid_z = (voxel_positions[..., 0] >= 0) & (voxel_positions[..., 0] < upsampled_shape[0])
            valid_y = (voxel_positions[..., 1] >= 0) & (voxel_positions[..., 1] < upsampled_shape[1])
            valid_x = (voxel_positions[..., 2] >= 0) & (voxel_positions[..., 2] < upsampled_shape[2])
            valid_mask = valid_z & valid_y & valid_x  # Shape: [n_atoms, n_voxels]

            # Calculate relative coordinates for all valid positions
            offset_tensor = torch.full_like(voxel_positions, cisTEM_offset)
            relative_coords = voxel_positions - atom_positions - atom_pos_dds - offset_tensor
            relative_coords = relative_coords * upsampled_pixel_size
            
            # Prepare coordinates for potential calculation
            valid_coords = relative_coords[valid_mask]
            coords1 = valid_coords 
            coords2 = valid_coords + upsampled_pixel_size
            coords1 = coords1.to(torch.float32)
            coords2 = coords2.to(torch.float32)
            
            # Get bPlusB for valid atoms
            valid_bPlusB = batch_bPlusB[type_indices][valid_mask.any(dim=1)]
            
            if len(valid_bPlusB) > 0:
                # Calculate potentials for all valid positions
                potentials = get_scattering_potential_of_voxel_batched(
                    coords1,
                    coords2,
                    valid_bPlusB,
                    atom_type,
                    lead_term,
                    scattering_params,
                    device
                )

                # Get valid positions for updating volume grid
                valid_positions = voxel_positions[valid_mask].long()
                
                # Update volume grid
                volume_grid.index_put_(
                    (valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2]),
                    potentials,
                    accumulate=True
                )

        if (start_idx + batch_size) % (batch_size * 5) == 0:
            print(f"Processed {start_idx + batch_size} atoms of {len(atom_indices)}")

    return volume_grid

def process_atoms_serial(
        atom_indices: torch.Tensor,
        bPlusB: torch.Tensor,
        atoms_id_filtered: list,
        voxel_offsets_flat: torch.Tensor,
        upsampled_shape: tuple,
        upsampled_pixel_size: float,
        lead_term: float
):
    """
    Process atoms serially (for CPU processing within a worker process)
    """
    # Initialize local volume grid
    local_volume = torch.zeros(upsampled_shape, device='cpu')
    
    # Process each atom
    for i in range(len(atom_indices)):
        atom_pos = atom_indices[i]
        atom_id = atoms_id_filtered[i]
        
        # Calculate voxel positions relative to atom center
        voxel_positions = atom_pos.view(1, 3) + voxel_offsets_flat
        
        # Check bounds for each dimension separately
        valid_z = (voxel_positions[:, 0] >= 0) & (voxel_positions[:, 0] < upsampled_shape[0])
        valid_y = (voxel_positions[:, 1] >= 0) & (voxel_positions[:, 1] < upsampled_shape[1])
        valid_x = (voxel_positions[:, 2] >= 0) & (voxel_positions[:, 2] < upsampled_shape[2])
        valid_mask = valid_z & valid_y & valid_x
        
        if valid_mask.any():
            relative_coords = ((voxel_positions[valid_mask] - atom_pos) * upsampled_pixel_size)
            coords1 = relative_coords - (upsampled_pixel_size/2)
            coords2 = relative_coords + (upsampled_pixel_size/2)
            
            potentials = get_scattering_potential_of_voxel(
                coords1,
                coords2,
                bPlusB[i],
                atom_id,
                lead_term,
                SCATTERING_PARAMETERS_A
            )
            
            valid_positions = voxel_positions[valid_mask].long()
            local_volume[valid_positions[:, 0], 
                        valid_positions[:, 1], 
                        valid_positions[:, 2]] += potentials
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} atoms")
            
    return local_volume

def process_device_atoms(args):
    """
    Process atoms for a single device in parallel.
    """
    (device_atom_indices, device_atom_dds, device_bPlusB, device_atoms_id, voxel_offsets_flat,
     upsampled_shape, upsampled_pixel_size, lead_term, device, n_cpu_cores) = args
    
    print(f"\nProcessing atoms on {device}")
    
    if device.type == "cuda":
        volume_grid = process_atoms_gpu2(
            atom_indices=device_atom_indices.to(device),
            atom_dds=device_atom_dds.to(device),
            bPlusB=device_bPlusB.to(device),
            atoms_id_filtered=device_atoms_id,
            voxel_offsets_flat=voxel_offsets_flat.to(device),
            upsampled_shape=upsampled_shape,
            upsampled_pixel_size=upsampled_pixel_size,
            lead_term=lead_term,
            device=device
        )
    else:
        volume_grid = process_atoms_parallel(
            atom_indices=device_atom_indices,
            atom_dds=device_atom_dds,
            bPlusB=device_bPlusB,
            atoms_id_filtered=device_atoms_id,
            voxel_offsets_flat=voxel_offsets_flat,
            upsampled_shape=upsampled_shape,
            upsampled_pixel_size=upsampled_pixel_size,
            lead_term=lead_term,
            n_cores=n_cpu_cores
        )
    
    return volume_grid

def get_upsampling(wanted_pixel_size: float, wanted_output_size: int, MAX_3D_SIZE: int = 1536):
    found_the_best_binning = False
    upsampling = 1.0
    # Check to make sure the sampling is sufficient, if not, oversample and bin at the end.
    if wanted_pixel_size > 1.5:
        if wanted_output_size * 4 < MAX_3D_SIZE:
            print("\nOversampling your 3d by a factor of 4 for calculation.")
            upsampling= 4.0
            found_the_best_binning = True

    if wanted_pixel_size > 0.75 and not found_the_best_binning:
        if wanted_output_size * 2 < MAX_3D_SIZE:
            print("\nOversampling your 3d by a factor of 2 for calculation.")
            upsampling = 2.0

    return int(upsampling)


def simulate_3d_volume(
        pdb_filename: str,
        sim_volume_shape: tuple[int, int, int],
        sim_pixel_spacing: float,
        n_cpu_cores: int,
        gpu_ids: list[int] = None, # [-999] is cpu, None is auto most available memory, [0, 1, 2], etc is specific gpu
        num_gpus: int = 1
):
    start_time = time.time()

    # Select devices
    if gpu_ids == [-999]:  # Special case for CPU-only
        devices = [torch.device("cpu")]
    else:
        devices = select_gpus(gpu_ids, num_gpus)
    
    print(f"Using devices: {[str(device) for device in devices]}")

    atoms_zyx, atoms_id, atoms_b_factor = load_model(pdb_filename, sim_pixel_spacing)

    # constants and stuff
    b_scaling = 0.5
    added_B = 0.0
    atoms_b_factor_scaled = 0.25 * (atoms_b_factor * b_scaling + added_B) # I'm not sure what the 0.25 is
    #atoms_b_factor_scaled = (atoms_b_factor * b_scaling + added_B) # I'm not sure what the 0.25 is
    mean_b_factor = torch.mean(atoms_b_factor_scaled)
    # max_b_factor = torch.max(atoms_b_factor_scaled)

    beam_energy = 300000 # eV
    wavelength = calculate_relativistic_electron_wavelength(beam_energy) # meters
    wavelength_A = wavelength * 1e10

    dose_weighting = True
    num_frames = 50
    flux = 1
    dose_B = -1
    apply_dqe = True

    #setup a grid with half the pixel size and double desired shape
    upsampling = get_upsampling(sim_pixel_spacing, sim_volume_shape[0])
    upsampled_pixel_size = sim_pixel_spacing / upsampling
    upsampled_shape = tuple(np.array(sim_volume_shape) * upsampling)

    #I'm making sure the index is on the edge of a voxel
    #origin_idx = (int(upsampled_shape[0] // 2), int(upsampled_shape[1] // 2), int(upsampled_shape[2] // 2))
    #cisTEM doesn't do this, it has origin as a float
    origin_idx = (upsampled_shape[0] / 2, upsampled_shape[1] / 2, upsampled_shape[2] / 2)


    #lead_term = BOND_SCALING_FACTOR * wavelength_A / 8.0 / upsampled_pixel_size / upsampled_pixel_size
    lead_term = BOND_SCALING_FACTOR * wavelength_A / 8.0 / (sim_pixel_spacing**2)
    #Not sure where the 8 comes from here

        # CisTEM does it like this though... I will use this for now
    size_neighborhood = get_size_neighborhood_cistem(mean_b_factor, upsampled_pixel_size)
    neighborhood_range = torch.arange(-size_neighborhood, size_neighborhood+1) 

    # Filter out hydrogen atoms
    non_h_mask = [id != "H" for id in atoms_id]
    atoms_zyx_filtered = atoms_zyx[non_h_mask]
    atoms_id_filtered = [id for i, id in enumerate(atoms_id) if non_h_mask[i]]
    atoms_b_factor_scaled_filtered = atoms_b_factor_scaled[non_h_mask]

    # Calculate atom indices in volume
    # Convert from centered Angstroms to voxel coordinates

    atom_indices = torch.zeros_like(atoms_zyx_filtered)
    atom_dds = torch.zeros_like(atoms_zyx_filtered)
    '''
    atom_indices[:, 0] = (atoms_zyx_filtered[:, 0] / upsampled_pixel_size) + origin_idx[0]  # z
    atom_indices[:, 1] = (atoms_zyx_filtered[:, 1] / upsampled_pixel_size) + origin_idx[1]  # y
    atom_indices[:, 2] = (atoms_zyx_filtered[:, 2] / upsampled_pixel_size) + origin_idx[2]  # x
    atom_indices = torch.round(atom_indices).int() # this is again the edge of the voxel
    '''
    this_coords = (atoms_zyx_filtered / upsampled_pixel_size) + torch.tensor(origin_idx).unsqueeze(0) + cisTEM_offset
    atom_indices = torch.floor(this_coords)
    atom_dds = this_coords - atom_indices - cisTEM_offset

    # Create coordinate grids for the neighborhood
    sz, sy, sx = torch.meshgrid(neighborhood_range, neighborhood_range, neighborhood_range, indexing='ij')
    # Calculate relative coordinates for the voxel corners
    # Stack and flatten differently to ensure we maintain 3D structure
    voxel_offsets = torch.stack([sz, sy, sx])  # (3, n, n, n)
    # Flatten while preserving the relative positions
    voxel_offsets_flat = voxel_offsets.reshape(3, -1).T  # (n^3, 3)

    # Split atoms across available devices
    num_devices = len(devices)
    atoms_per_device = len(atoms_zyx) // num_devices
    device_outputs = []

    before_volume = time.time()
    elapsed_time = before_volume - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Before main simulation time: {minutes} minutes {seconds} seconds")
    # Handle CPU and GPU cases separately
    if devices[0].type == "cpu":
        b_params = torch.stack([torch.tensor(SCATTERING_PARAMETERS_B[atom_id]) 
                              for atom_id in atoms_id_filtered])
        bPlusB = 2 * torch.pi / torch.sqrt(atoms_b_factor_scaled_filtered.unsqueeze(1) + b_params)
        # If CPU only, use the original parallel processing directly
        volume_grid = process_atoms_parallel(
            atom_indices=atom_indices,
            atom_dds=atom_dds,
            bPlusB=bPlusB,
            atoms_id_filtered=atoms_id_filtered,
            voxel_offsets_flat=voxel_offsets_flat,
            upsampled_shape=upsampled_shape,
            upsampled_pixel_size=upsampled_pixel_size,
            lead_term=lead_term,
            n_cores=n_cpu_cores
        )
        device_outputs = [volume_grid]
    else:
        device_args = []
        for i, device in enumerate(devices):
            # Calculate start and end indices for this device
            start_idx = i * atoms_per_device
            end_idx = start_idx + atoms_per_device if i < num_devices - 1 else len(atom_indices)
            
            # Get device-specific data
            device_atom_indices = atom_indices[start_idx:end_idx]
            device_atom_dds = atom_dds[start_idx:end_idx]
            device_atoms_id = atoms_id_filtered[start_idx:end_idx]
            device_b_factor = atoms_b_factor_scaled_filtered[start_idx:end_idx]
            
            # Calculate B-factors for this device's atoms
            b_params = torch.stack([torch.tensor(SCATTERING_PARAMETERS_B[atom_id]) 
                              for atom_id in device_atoms_id])
            device_bPlusB = 2 * torch.pi / torch.sqrt(device_b_factor.unsqueeze(1) + b_params)

            args = (device_atom_indices, device_atom_dds, device_bPlusB, device_atoms_id, voxel_offsets_flat,
               upsampled_shape, upsampled_pixel_size, lead_term, device, n_cpu_cores)
            device_args.append(args)
        
        # Process on all devices in parallel
        with mp.get_context('spawn').Pool(processes=len(devices)) as pool:
            device_outputs = pool.map(process_device_atoms, device_args)

    # Combine results from all devices
    main_device = devices[0]
    final_volume = torch.zeros(upsampled_shape, device=main_device)
    for volume in device_outputs:
        final_volume += volume.to(main_device) 

    #Undo the upsanmpling here
    '''
    # Bin the volume back to original size using 3D average pooling
    volume_grid_binned = torch.nn.functional.avg_pool3d(
        final_volume.unsqueeze(0),  # Add batch dimension
        kernel_size=upsampling,
        stride=upsampling
    ).squeeze(0)  # Remove batch dimension
    '''
    #Dose weight
    final_volume = final_volume.cpu()
    modify_signal = 1  # weird parameter here
    if dose_weighting:
        dose_filter = cumulative_dose_filter_3d(
            volume=final_volume,
            num_frames=num_frames,
            start_exposure=0,
            pixel_size=upsampled_pixel_size,
            flux=flux,
            Bfac=dose_B   
        )
        print(f"Dose filter min: {dose_filter.min()}, max: {dose_filter.max()}")
        final_volume_FFT = torch.fft.rfftn(final_volume.contiguous(), dim=(-3, -2, -1))
        
        if modify_signal == 1:
            # Add small epsilon to prevent division by zero
            denominator = 1 + dose_filter
            epsilon = 1e-10
            denominator = torch.clamp(denominator, min=epsilon)
            modification = (1-(1-dose_filter)/denominator)
            
            # Check for invalid values
            if torch.any(torch.isnan(modification)):
                print("Warning: NaN values in modification factor")
                modification = torch.nan_to_num(modification, nan=1.0)
                
            final_volume_FFT *= modification
        elif modify_signal == 2:
            final_volume_FFT *= dose_filter**0.5
        else:
            final_volume_FFT *= dose_filter
        
        final_volume = torch.fft.irfftn(final_volume_FFT, dim=(-3, -2, -1))
        # Check for NaN values in final result
        if torch.any(torch.isnan(final_volume)):
            print("Warning: NaN values in final volume")

        #volume_grid_binned = dose_weight_3d_volume(
        #    volume=volume_grid_binned,
        #    num_frames=num_frames,
        #    pixel_size=sim_pixel_spacing,
        #    flux=flux,
        #    Bfac=dose_B
        #)
    
    if apply_dqe:
        final_volume = apply_dqe_volume(
            volume=final_volume,
            dqe_parameters_a=DQE_PARAMETERS_A,
            dqe_parameters_b=DQE_PARAMETERS_B,
            dqe_parameters_c=DQE_PARAMETERS_C,
            pixel_size=upsampled_pixel_size
        )
    
    #fourier crop
    '''
    volume_grid_binned, _ = fourier_rescale_3d(
        image=final_volume,
        source_spacing=upsampled_pixel_size,
        target_spacing=sim_pixel_spacing
    )
    '''
    if upsampling > 1:
        final_volume = fourier_rescale_3d_force_size(
            volume=final_volume,
            target_size=sim_volume_shape[0],
        ) 
    
    # I would then do any moodifications to volume grid here

    # Move to CPU only for final save
    # apply a dose filter if specified 
    #volume_grid_binned = volume_grid_binned.cpu()

    before_dw = time.time()
    elapsed_time = before_dw - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Before dw simulation time: {minutes} minutes {seconds} seconds")


        
    after_dw = time.time()
    elapsed_time = after_dw - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"After dw simulation time: {minutes} minutes {seconds} seconds")
    # Apply a DQE if specified 

    mrcfile.write("simulated_volume_dw_us2_cpu.mrc", 
                  final_volume.numpy(), overwrite=True)

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
