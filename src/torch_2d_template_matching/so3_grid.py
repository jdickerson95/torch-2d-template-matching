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
