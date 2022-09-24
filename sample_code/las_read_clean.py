# import open3d as o3d
# from pyntcloud import PyntCloud

import numpy as np
from scipy import stats
import pylas
import os

cwd = os.getcwd()

print("Load a ply point cloud, print it, and render it")
# load by pylas
pcd = pylas.read(f'{cwd}\\Assignment01-Data-everybuilding\\B1.las')
a = pcd.points
print(pcd.x[0], pcd.y[0], pcd.z[0])

print(pcd.header.point_format_id)
# create save file
new_file = pylas.create(point_format_id=pcd.header.point_format_id, file_version=pcd.header.version)


# change height
# pcd_withheight = pcd.points
# find mode for this dataset
pcd_z_mode = stats.mode(pcd.z)[0][0]
b = stats.mode(pcd.z)
print('z value max={}, min={}, mode={}'.format(max(pcd.z), min(pcd.z), pcd_z_mode))

# choose points with z value more than the mode
new_file.points = pcd.points[pcd.z > (pcd_z_mode+2)]
print(type(pcd.points))
print(len(pcd.points[pcd.z > pcd_z_mode]))
print('z', stats.mode(pcd.z))


print(pcd.points)
print(new_file.x)
# save las file
new_file.write(f'{cwd}\\Assignment01-Data\\B1_02.las')
print(new_file.points)
