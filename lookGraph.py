import numpy as np
import scipy
import algorithmsPart2 as alg
import FrameGraphPart2 as fg
import perspectivePart2 as pt
import pickle
import matplotlib.pyplot as plt

PARAMS = {
    'match_threshold': 0.9,
    'edges_num_neighbors': 3,
    'edges_inliers_threshold': 18, # 12 also works well
    'node_reference_index': 0,
    'RANSAC_inlier_threshold': 2.0,
    'RANSAC_max_iter': 700,
    'MSAC_max_iter': 1000,
    'MSAC_threshold': 5.0,
    'MSAC_confidence': 0.99,
}


with open("office/graph.pkl", "rb") as f:
    nodes = pickle.load(f)



composite_T, path_lenghts, path_costs, graph = fg.compute_composite_transformations(nodes, PARAMS)

### Plot of the path lenghts
plt.figure()
plt.plot(path_lenghts, '.')
plt.xlabel("Frame index")
plt.ylabel("Path length")
plt.grid()
plt.savefig("office/path_lenghts.png")


### Plot of the path lenghts
plt.figure()
plt.plot(path_costs, '.')
plt.xlabel("Frame index")
plt.ylabel("Path Cost")
plt.grid()
plt.savefig("office/path_costs.png")  


### Plot the graph
plt.figure()
fg.plot_graph(graph)


#### see point cloud

import open3d as o3d



### pcd_ref 
pcd_ref = o3d.geometry.PointCloud()
pcd_ref.points = o3d.utility.Vector3dVector(nodes[0].points_cloud[:, :3])
pcd_ref.colors = o3d.utility.Vector3dVector(nodes[0].points_cloud[:, 3:])


o3d.visualization.draw_geometries([pcd_ref], window_name="point cloud reference")



counter = 1

node = nodes[counter]
points3D = node.points_3D
colors3D = node.colors_3D

points_cloud = node.points_cloud

print("points3D_A.shape", points3D.shape)
print("points cloud A shape: ", colors3D.shape)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_cloud[:, :3])
pcd.colors = o3d.utility.Vector3dVector(points_cloud[:, 3:])

o3d.io.write_point_cloud("office/point_cloud.ply", pcd)


### implement trasnformations
T = composite_T[counter]

points3D_transformed = np.dot(T[:3, :3], points3D.T).T + T[:3, 3]


print(T)

## apply to point cloud
points_cloud_transformed = np.dot(T[:3, :3], points_cloud[:, :3].T).T + T[:3, 3]


print(points_cloud_transformed)

print("points3D_transformed.shape", points3D_transformed.shape)
print("points cloud transformed shape: ", points_cloud_transformed.shape)
print("clouds shape: ", points_cloud.shape)

## color is the same
pcd2_original = o3d.geometry.PointCloud()
pcd2_original.points = o3d.utility.Vector3dVector(points_cloud[:, :3])
pcd2_original.colors = o3d.utility.Vector3dVector(points_cloud[:, 3:])

o3d.visualization.draw_geometries([pcd2_original], window_name="point cloud original")


pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(points_cloud_transformed[:, :3])
pcd2.colors = o3d.utility.Vector3dVector(points_cloud[:, 3:])

o3d.io.write_point_cloud("office/point_cloud_transformed.ply", pcd2)
o3d.visualization.draw_geometries([pcd2], window_name="point cloud transformed")


### join pcd2 and pcd_ref

pcd_ref.paint_uniform_color([0.0, 0.0, 1.0])
pcd2.paint_uniform_color([1.0, 0.0, 0.0])

o3d.visualization.draw_geometries([pcd2, pcd_ref], window_name="point cloud transformed")