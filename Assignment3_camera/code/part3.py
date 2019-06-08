import numpy as np
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import cv2

import plotly
import plotly.graph_objs as go

# read the camera matrices for the library and the matching points
library_camera_matrix_one = np.loadtxt(fname = "../data/library/library1_camera.txt")
library_camera_matrix_two = np.loadtxt(fname = "../data/library/library2_camera.txt")
library_matches = np.loadtxt(fname='../data/library/library_matches.txt')

# read the camera matrices for the house and the matching points
house_camera_matrix_one = np.loadtxt(fname = "../data/house/house1_camera.txt")
house_camera_matrix_two = np.loadtxt(fname = "../data/house/house2_camera.txt")
house_matches = np.loadtxt(fname='../data/house/house_matches.txt')

# calculate the camera centers for the library with singular value decomposition
_, _, library_camera_one_v = np.linalg.svd(library_camera_matrix_one)
_, _, library_camera_two_v = np.linalg.svd(library_camera_matrix_two)

# calculate the camera centers for the house with singular value decomposition
_, _, house_camera_one_v = np.linalg.svd(house_camera_matrix_one)
_, _, house_camera_two_v = np.linalg.svd(house_camera_matrix_two)

#matching points for the library
matching_points = []

# calculate the camera centers in 3D
cam_1_center = library_camera_one_v[-1] # the 3D coordinate of the camera center is the last row in V
cam_1_center = cam_1_center/cam_1_center[-1] # divide by the last entry to get vector [x, y, z, 1]
cam_2_center = library_camera_two_v[-1] # the 3D coordinate of the camera center is the last row in V
cam_2_center = cam_2_center/cam_2_center[-1] # divide by the last entry to get vector [x, y, z, 1]

# loop over all matching points
for matches in library_matches:
	# calculate matrix A from the formula AX=0 according to the pages 8, 9, 10 in the slides from the assignment
    A = [] # 4x4 matrix
    A.append(matches[0] * library_camera_matrix_one[2] - library_camera_matrix_one[0])
    A.append(matches[1] * library_camera_matrix_one[2] - library_camera_matrix_one[1])
    A.append(matches[2] * library_camera_matrix_two[2] - library_camera_matrix_two[0])
    A.append(matches[3] * library_camera_matrix_two[2] - library_camera_matrix_two[1])
    A = np.array(A)
    
    # do singular value decomposition of A to get homogeneous coordinates for the matching points
    U,D,V = np.linalg.svd(A)

    # the 3D coordinate of the matching points is the last row in V
    # cv2.triangulatePoints(library_camera_matrix_one, library_camera_matrix_two, (matches[0],matches[1]), (matches[2],matches[3]))
    homogeneous_coordinates = (-1) * V[-1] # multiply by -1 because the build in function of opencv gives inverted coordinates (?)
    homogeneous_coordinates = homogeneous_coordinates/homogeneous_coordinates[-1] # divide by the last entry to get vector [x, y, z, 1]

    matching_points.append(homogeneous_coordinates)
    
matching_points = np.array(matching_points, dtype='float64')

# print all homogeneous coordinates in a diagram together with the camera centers
mp = go.Scatter3d(x=matching_points[:, 0], y=matching_points[:, 1], z=matching_points[:, 2], mode="markers", marker = dict(color = 'blue', opacity = 0.8), name='Matching Points')
c1 = go.Scatter3d(x=[cam_1_center[0]], y=[cam_1_center[1]], z=[cam_1_center[2]], mode="markers", marker = dict(color = 'red', opacity = 0.8), name='Camera 1')
c2 = go.Scatter3d(x=[cam_2_center[0]], y=[cam_2_center[1]], z=[cam_2_center[2]], mode="markers", marker = dict(color = 'orange', opacity = 0.8), name='Camera 2')
layout = go.Layout(title='3D-Projection Library')
plotly.offline.plot(go.Figure(data=[mp, c1, c2], layout=layout), filename="3D_projection_library.html", auto_open=False)

# matching points for the house
matching_points = []

# calculate the camera centers in 3D
cam_1_center = house_camera_one_v[-1] # the 3D coordinate of the camera center is the last row in V
cam_1_center = cam_1_center/cam_1_center[-1] # divide by the last entry to get vector [x, y, z, 1]
cam_2_center = house_camera_two_v[-1] # the 3D coordinate of the camera center is the last row in V
cam_2_center = cam_2_center/cam_2_center[-1] # divide by the last entry to get vector [x, y, z, 1]

# loop over all matching points
for matches in house_matches:
	# calculate matrix A from the formula AX=0 according to the pages 8, 9, 10 in the slides from the assignment
    A = [] # 4x4 matrix
    A.append(matches[0] * house_camera_matrix_one[2] - house_camera_matrix_one[0])
    A.append(matches[1] * house_camera_matrix_one[2] - house_camera_matrix_one[1])
    A.append(matches[2] * house_camera_matrix_two[2] - house_camera_matrix_two[0])
    A.append(matches[3] * house_camera_matrix_two[2] - house_camera_matrix_two[1])
    A = np.array(A)
    
    # do singular value decomposition of A to get homogeneous coordinates for the matching points
    U,D,V = np.linalg.svd(A)

    # the 3D coordinate of the matching points is the last row in V
    # cv2.triangulatePoints(library_camera_matrix_one, library_camera_matrix_two, (matches[0],matches[1]), (matches[2],matches[3]))
    homogeneous_coordinates = (-1) * V[-1] # multiply by -1 because the build in function of opencv gives inverted coordinates (?)
    homogeneous_coordinates = homogeneous_coordinates/homogeneous_coordinates[-1] # divide by the last entry to get vector [x, y, z, 1]

    matching_points.append(homogeneous_coordinates)
    
matching_points = np.array(matching_points, dtype='float64')

# print all homogeneous coordinates in a diagram together with the camera centers
mp = go.Scatter3d(x=matching_points[:, 0], y=matching_points[:, 1], z=matching_points[:, 2], mode="markers", marker = dict(color = 'blue', opacity = 0.8), name='Matching Points')
c1 = go.Scatter3d(x=[cam_1_center[0]], y=[cam_1_center[1]], z=[cam_1_center[2]], mode="markers", marker = dict(color = 'red', opacity = 0.8), name='Camera 1')
c2 = go.Scatter3d(x=[cam_2_center[0]], y=[cam_2_center[1]], z=[cam_2_center[2]], mode="markers", marker = dict(color = 'orange', opacity = 0.8), name='Camera 2')
layout = go.Layout(title='3D-Projection House')
plotly.offline.plot(go.Figure(data=[mp, c1, c2], layout=layout), filename="3D_projection_house.html", auto_open=False)