import numpy as np
import scipy
import algorithmsPart2PnP as alg
import FrameGraphPart2PnP as fg
import pickle
import matplotlib.pyplot as plt
import open3d as o3d

PARAMS = {
    'node_reference_index': 0,
}


with open("office/graph.pkl", "rb") as f:
    nodes = pickle.load(f)


# composite_T, path_lenghts, path_costs, graph = fg.compute_composite_transformations(nodes, PARAMS)

# ### Plot of the path lenghts
# plt.figure()
# plt.plot(path_lenghts, '.')
# plt.xlabel("Frame index")
# plt.ylabel("Path length")
# plt.grid()
# plt.savefig("office/path_lenghts.png")


# ### Plot of the path lenghts
# plt.figure()
# plt.plot(path_costs, '.')
# plt.xlabel("Frame index")
# plt.ylabel("Path Cost")
# plt.grid()
# plt.savefig("office/path_costs.png")  


# ### Plot the graph
# plt.figure()
# fg.plot_graph(graph)



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

R_mat = nodes[counter].connections[0][0]
t_vec = nodes[counter].connections[0][1]




points3D_transformed = np.dot(R_mat, points3D.T).T + t_vec

## apply to point cloud
points_cloud_transformed = np.dot(R_mat, points_cloud[:, :3].T).T + t_vec


# print(points_cloud_transformed)

print("points3D_transformed.shape", points3D_transformed.shape)
print("points cloud transformed shape: ", points_cloud_transformed.shape)
print("clouds shape: ", points_cloud.shape)

## color is the same
pcd2_original = o3d.geometry.PointCloud()
pcd2_original.points = o3d.utility.Vector3dVector(points_cloud[:, :3])
pcd2_original.colors = o3d.utility.Vector3dVector(points_cloud[:, 3:])

o3d.visualization.draw_geometries([pcd2_original, pcd_ref], window_name="point cloud original")


pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(points_cloud_transformed[:, :3])
pcd2.colors = o3d.utility.Vector3dVector(points_cloud[:, 3:])

o3d.io.write_point_cloud("office/point_cloud_transformed.ply", pcd2)
o3d.visualization.draw_geometries([pcd2, pcd_ref], window_name="point cloud transformed")


### join pcd2 and pcd_ref
pcd_ref.paint_uniform_color([0.0, 0.0, 1.0])
pcd2.paint_uniform_color([1.0, 0.0, 0.0])

o3d.visualization.draw_geometries([pcd2, pcd_ref], window_name="point cloud transformed")