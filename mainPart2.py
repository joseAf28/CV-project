import numpy as np
import scipy
import algorithmsPart2 as alg
import FrameGraphPart2 as fg
import pickle
import matplotlib.pyplot as plt
import open3d as o3d



PARAMS = {
    'match_threshold': 0.9,
    'edges_num_neighbors': 3,
    'edges_inliers_threshold': 17,
    'node_reference_index': 0,
    
    'f_threshold': 2.0,
    'f_confidence': 0.999,
    'f_max_iter': 10000,
    
    'MSAC_max_iter': 1900,
    'MSAC_threshold': 3.0,
    'MSAC_confidence': 0.999,
    
    'Cost_threshold': 0.9,
    
    'Depth_threshold': 0.75,
    
    'ICP_flag': True,
    'ICP_max_iter': 70,
    'ICP_tolerance': 1e-5,
}


cams_info_data = scipy.io.loadmat('office/cams_info_no_extr.mat')
kp_data = scipy.io.loadmat('office/kp.mat')
wrld_data = scipy.io.loadmat('office/wrld_info.mat')


### Convert the data to the appropriate format
num_images = 10

rgb_shape = cams_info_data['cams_info'][0][0]['rgb'][0][0].shape
rgb_type = cams_info_data['cams_info'][0][0]['rgb'][0][0].dtype

depth_shape = cams_info_data['cams_info'][0][0]['depth'][0][0].shape
depth_type = cams_info_data['cams_info'][0][0]['depth'][0][0].dtype

conf_shape = cams_info_data['cams_info'][0][0]['conf'][0][0].shape
conf_type = cams_info_data['cams_info'][0][0]['conf'][0][0].dtype


rgb_images = np.zeros((rgb_shape[0], rgb_shape[1], rgb_shape[2], num_images), dtype=rgb_type)
depth_images = np.zeros((depth_shape[0], depth_shape[1], num_images), dtype=depth_type)
conf_images = np.zeros((conf_shape[0], conf_shape[1], num_images), dtype=conf_type)

fx_vec = np.zeros((num_images), dtype=np.float64)
fy_vec = np.zeros((num_images), dtype=np.float64)

for i in range(num_images):
    rgb_images[..., i] = cams_info_data['cams_info'][i][0]['rgb'][0][0]
    depth_images[..., i] = cams_info_data['cams_info'][i][0]['depth'][0][0]
    conf_images[..., i] = cams_info_data['cams_info'][i][0]['conf'][0][0]
    
    fx_vec[i] = cams_info_data['cams_info'][i][0]['focal_lenght'][0,0][0,0]
    fy_vec[i] = cams_info_data['cams_info'][i][0]['focal_lenght'][0,0][0,0]


cx = rgb_shape[1] / 2
cy = rgb_shape[0] / 2


### Preprocess depth
def preprocess_depth(depth, conf, threshold=0.75):
    
    mask = conf > threshold
    depth_filtered = np.where(mask, depth, 0)
    return depth_filtered


depth_images_filtered = np.zeros_like(depth_images)
intrinsecs = []

for i in range(num_images):
    depth_images_filtered[..., i] = preprocess_depth(depth_images[..., i], conf_images[..., i], PARAMS['Depth_threshold'])
    intrinsecs.append((fx_vec[i], fy_vec[i], cx, cy))


###! create the Graph to extract the keypoint descriptor and features
nodes = fg.initialize_graph(kp_data, depth_images_filtered, rgb_images, intrinsecs)



###! compute the edges between the nodes
nodes = fg.compute_edges(nodes, PARAMS)


###! Compute the composite transformations
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



### pcd_ref 

ref = PARAMS['node_reference_index']

pcd_ref = o3d.geometry.PointCloud()
pcd_ref.points = o3d.utility.Vector3dVector(nodes[ref].points_cloud[:, :3])
pcd_ref.colors = o3d.utility.Vector3dVector(nodes[ref].points_cloud[:, 3:])


### merge the point clouds
merge_pcd = o3d.geometry.PointCloud()
merge_pcd += pcd_ref


list_nodes = np.arange(0, len(nodes))
list_nodes = list_nodes[list_nodes != ref]

print("list_nodes: ", list_nodes)


for counter in list_nodes:
    
    cost = path_costs[counter]
    
    if cost > PARAMS['Cost_threshold']:
        continue
    
    node = nodes[counter]
    points3D = node.points_3D
    colors3D = node.colors_3D

    points_cloud = node.points_cloud
    
    T = composite_T[counter]

    points_cloud_transformed = np.dot(T[:3, :3], points_cloud[:, :3].T).T + T[:3, 3]
    
    pcdcounter = o3d.geometry.PointCloud()
    pcdcounter.points = o3d.utility.Vector3dVector(points_cloud_transformed[:, :3])
    pcdcounter.colors = o3d.utility.Vector3dVector(points_cloud[:, 3:])

    merge_pcd += pcdcounter



voxel_size = 0.02
merge_pcd = merge_pcd.voxel_down_sample(voxel_size)
merge_pcd, ind = merge_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)

o3d.io.write_point_cloud("office/point_cloud_transformed.ply", merge_pcd)
o3d.visualization.draw_geometries([merge_pcd], window_name="point cloud transformed")
