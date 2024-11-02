import h3
import torch
import einops
from scipy.spatial.transform import Rotation as R
from eulerangles import invert_rotation_matrices


def get_h3_grid_at_resolution(resolution: int) -> list[str]:
    """Get h3 cells (their h3 index) at a given resolution.

    Each cell appears once
    - resolution 0:       122 cells, every   ~20 degrees
    - resolution 1:       842 cells, every  ~7.5 degrees
    - resolution 2:     5,882 cells, every    ~3 degrees
    - resolution 3:    41,162 cells, every    ~1 degrees
    - resolution 4:   288,122 cells, every  ~0.4 degrees
    - resolution 5: 2,016,842 cells, every ~0.17 degrees
    c.f. https://h3geo.org/docs/core-library/restable

    """
    res0 = h3.get_res0_indexes()
    if resolution == 0:
        h = list(res0)
    else:
        h = [h3.h3_to_children(idx, resolution) for idx in res0]
        h = [item for sublist in h for item in sublist]  # flatten
    return h


def geo_to_xyz(latlon: torch.Tensor):
    latlon = torch.as_tensor(latlon, dtype=torch.float32)

    # Convert latitude and longitude from degrees to radians
    lat_radians = torch.deg2rad(latlon[..., 0])
    lon_radians = torch.deg2rad(latlon[..., 1])

    # Convert geographic coordinates to cartesian
    x = torch.cos(lat_radians) * torch.cos(lon_radians)
    y = torch.cos(lat_radians) * torch.sin(lon_radians)
    z = torch.sin(lat_radians)
    return einops.rearrange([x, y, z], 'xyz ... -> ... xyz')


def xyz_to_geo(xyz: torch.Tensor):
    # Convert cartesian coordinates to geographic
    lat_radians = torch.arcsin(xyz[..., 2])
    lon_radians = torch.atan2(xyz[..., 1], xyz[..., 0])

    # Convert latitude and longitude from radians to degrees
    latlon = einops.rearrange([lat_radians, lon_radians], 'latlon ... -> ... latlon')
    return torch.rad2deg(latlon)


def euler_to_rotation_matrix(euler_angles: torch.Tensor):
    rotation = R.from_euler(seq='ZYZ', angles=euler_angles, degrees=True)
    return rotation.inv().as_matrix()


def euler_to_h3(euler_angles: torch.Tensor) -> str:
    rotation_matrix = euler_to_rotation_matrix(euler_angles)
    xyz = rotation_matrix[:, 2]  # rotated z-vector
    latlon = xyz_to_geo(xyz)
    return geo_to_xyz(latlon)


def h3_to_rotation_matrix(h: str) -> torch.Tensor:
    geo = h3.h3_to_geo(h)
    z = geo_to_xyz(geo)  # rotated z vector, on unit sphere

    # generate x and y in the plane
    random_vector = torch.rand((3, ))
    random_vector /= torch.linalg.norm(random_vector)
    while torch.dot(z, random_vector) == 1:
        random_vector = torch.rand((3, ))
        random_vector /= torch.linalg.norm(random_vector)
    y = torch.cross(z, random_vector)
    x = torch.cross(y, z)

    # construct rotation matrix
    rotation_matrix = torch.zeros((3, 3))
    rotation_matrix[:, 0] = x
    rotation_matrix[:, 1] = y
    rotation_matrix[:, 2] = z

    rotation_matrix = invert_rotation_matrices(rotation_matrix)
    return rotation_matrix


def h3_to_xyz(h: str):
    return geo_to_xyz(torch.tensor(h3.h3_to_geo(h)))

def get_uniform_euler_angles(in_plane_step: float = 1.5, out_of_plane_step: float = 2.5) -> torch.Tensor:
    """Generate uniform euler angles (ZYZ convention) using Hopf fibration.

    Uses right-handed, intrinsic (active) rotations in ZYZ convention:
    1. First rotation by alpha around Z axis   [-180° to +180°]
    2. Second rotation by beta around Y' axis  [0° to 180°]
    3. Third rotation by gamma around Z'' axis [-180° to +180°]
    
    Note: The prime (') notation indicates the rotated coordinate system.
    
    Args:
        in_plane_step: Angular step for in-plane rotation (alpha, gamma) in degrees
        out_of_plane_step: Angular step for out-of-plane rotation (beta) in degrees
    
    Returns:
        torch.Tensor of shape (N, 3) containing euler angles in degrees [alpha, beta, gamma]
        where:
            alpha ∈ [-180°, +180°]  - First rotation around Z
            beta  ∈ [0°, 180°]      - Second rotation around Y'
            gamma ∈ [-180°, +180°]  - Third rotation around Z''
    """
    # Convert steps to radians
    in_plane_step = torch.deg2rad(torch.tensor(in_plane_step))
    out_of_plane_step = torch.deg2rad(torch.tensor(out_of_plane_step))
    
    # Calculate number of samples for each angle
    n_beta = int(torch.ceil(torch.tensor(180) / torch.rad2deg(out_of_plane_step)))
    
    # Generate beta values (out-of-plane angle)
    beta = torch.linspace(0, torch.pi, n_beta)
    
    # Initialize list to store all angles
    euler_angles = []
    
    # For each beta, calculate appropriate number of alpha and gamma samples
    for b in beta:
        # Number of samples for alpha at this beta
        n_alpha = max(1, int(torch.ceil(2 * torch.pi * torch.sin(b) / in_plane_step)))
        n_gamma = int(torch.ceil(2 * torch.pi / in_plane_step))
        
        # Generate alpha and gamma values
        alpha = torch.linspace(-torch.pi, torch.pi, n_alpha)
        gamma = torch.linspace(-torch.pi, torch.pi, n_gamma)
        
        # Create all combinations using einops
        alpha = einops.repeat(alpha, 'a -> a g', g=n_gamma)
        gamma = einops.repeat(gamma, 'g -> a g', a=n_alpha)
        beta_expanded = torch.full_like(alpha, b)
        
        # Stack angles
        angles = einops.rearrange([alpha, beta_expanded, gamma], 
                                'xyz h w -> (h w) xyz')
        euler_angles.append(angles)
    
    # Combine all angles and convert to degrees
    euler_angles = torch.cat(euler_angles, dim=0)
    euler_angles = torch.rad2deg(euler_angles)
    
    return euler_angles


# import napari
#
# viewer = napari.Viewer()
# for resolution in range(0, 3):
#     geo = [torch.tensor(h3.h3_to_geo(cell)) for cell in
#            get_h3_grid_at_resolution(resolution)]
#     geo = torch.stack(geo)
#     xyz = geo_to_xyz(geo)
#     viewer.add_points(xyz, name=f'res{resolution}')
# napari.run()

# # res0 appears to be separated by ~20 degrees
# for cell in cells:
#     xyz_i = geo_to_xyz(torch.tensor(h3.h3_to_geo(cell)))
#     xyz_i /= torch.linalg.norm(xyz_i)
#     disk = h3.k_ring(cell, k=1)
#     for h in disk:
#         xyz_j = geo_to_xyz(torch.tensor(h3.h3_to_geo(h)))
#         xyz_j /= torch.linalg.norm(xyz_j)
#         angle = torch.rad2deg(torch.acos(torch.dot(xyz_i, xyz_j)))
#         print(angle)

# # res 1 7.5 degrees
# my_resolution = 1
# for index_0 in h3.get_res0_indexes():
#     for child_index in h3.h3_to_children(index_0, my_resolution):
#         xyz_i = geo_to_xyz(torch.tensor(h3.h3_to_geo(child_index)))
#         xyz_i /= torch.linalg.norm(xyz_i)
#         disk = h3.k_ring(child_index, k=1)
#         for h in disk:
#             xyz_j = geo_to_xyz(torch.tensor(h3.h3_to_geo(h)))
#             xyz_j /= torch.linalg.norm(xyz_j)
#             angle = torch.rad2deg(torch.acos(torch.dot(xyz_i, xyz_j)))
#         print(angle)
#
# # res2 ~every 3 degrees
# my_resolution = 2
# for index_0 in h3.get_res0_indexes():
#     for child_index in h3.h3_to_children(index_0, my_resolution):
#         xyz_i = geo_to_xyz(torch.tensor(h3.h3_to_geo(child_index)))
#         xyz_i /= torch.linalg.norm(xyz_i)
#         disk = h3.k_ring(child_index, k=1)
#         for h in disk:
#             xyz_j = geo_to_xyz(torch.tensor(h3.h3_to_geo(h)))
#             xyz_j /= torch.linalg.norm(xyz_j)
#             angle = torch.rad2deg(torch.acos(torch.dot(xyz_i, xyz_j)))
#         print(angle)
#
# # res3 ~every 1 degrees
# my_resolution = 3
# for index_0 in h3.get_res0_indexes():
#     print(len(h3.h3_to_children(index_0, my_resolution)))
#     for child_index in h3.h3_to_children(index_0, my_resolution):
#         xyz_i = geo_to_xyz(torch.tensor(h3.h3_to_geo(child_index)))
#         xyz_i /= torch.linalg.norm(xyz_i)
#         disk = h3.k_ring(child_index, k=1)
#         for h in disk:
#             xyz_j = geo_to_xyz(torch.tensor(h3.h3_to_geo(h)))
#             xyz_j /= torch.linalg.norm(xyz_j)
#             angle = torch.rad2deg(torch.acos(torch.dot(xyz_i, xyz_j)))
#         print(angle)
