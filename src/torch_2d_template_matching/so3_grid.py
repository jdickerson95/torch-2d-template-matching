import collections

import h3
import torch
import einops

cells = h3.get_res0_indexes()
geo = [torch.tensor(h3.h3_to_geo(cell)) for cell in cells]
geo = torch.stack(geo, dim=0)


def geo_to_xyz(latlon: torch.Tensor):
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


def get_h3_at_resolution(resolution: int) -> list[str]:
    res0 = h3.get_res0_indexes()
    if resolution == 0:
        h = list(res0)
    else:
        h = [h3.h3_to_children(idx, resolution) for idx in res0]
        h = [item for sublist in h for item in sublist]  # flatten
    return h


def h3_to_xyz(h: str):
    return geo_to_xyz(torch.tensor(h3.h3_to_geo(h)))

import napari
viewer = napari.Viewer()
for resolution in range(0, 3):
    geo = [torch.tensor(h3.h3_to_geo(cell)) for cell in get_h3_at_resolution(resolution)]
    geo = torch.stack(geo)
    xyz = geo_to_xyz(geo)
    viewer.add_points(xyz, name=f'res{resolution}')
napari.run()




#
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
