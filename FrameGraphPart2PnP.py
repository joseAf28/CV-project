import numpy as np
import scipy
from scipy.spatial import Delaunay
import algorithmsPart2PnP as alg
import tqdm
import heapq


#############! FrameNode class
class FrameNode:
    
    @staticmethod
    def depth_to_points_cloud(depth, rgb, intrinsics):
        fx, fy, cx, cy = intrinsics
        
        y_indices, x_indices = np.indices(depth.shape)
        
        x = (x_indices - cx) * depth / fx
        y = (y_indices - cy) * depth / fy
        z = depth
        
        points = np.stack([x, y, z], axis=-1)

        rgb_mod = rgb.reshape(-1, 3)
        points = points.reshape(-1, 3)
        points_cloud = np.concatenate([points, rgb_mod], axis=-1)
        
        return points_cloud
    
    
    @staticmethod
    def depth_to_point3D(kp, desc, depth, rgb, intrinsics):
        fx, fy, cx, cy = intrinsics
        
        u, v = kp[:, 0], kp[:, 1]
        z = depth[v.astype(int), u.astype(int)]
        
        valid_mask = z > 0
        umod, vmod, zmod = u[valid_mask], v[valid_mask], z[valid_mask]
        
        x = (umod - cx) * zmod / fx
        y = (vmod - cy) * zmod / fy
        
        
        rgb_mod = rgb[vmod.astype(int), umod.astype(int)]
        rgb_mod = rgb_mod.reshape(-1, 3)
        
        
        points = np.stack([x, y, zmod], axis=-1)
        kp_valid = kp[valid_mask]
        desc_valid = desc[valid_mask]
        
        
        return points, rgb_mod, kp_valid, desc_valid
    
    
    def __init__(self, frame_id, keypoints, descriptors, depth_points, rgb_image, intrinsics):
        self.frame_id = frame_id
        
        points_cloud = FrameNode.depth_to_points_cloud(depth_points, rgb_image, intrinsics)
        
        points_3D, colors_3D, kp_valid, desc_valid = FrameNode.depth_to_point3D(keypoints, descriptors, depth_points, rgb_image, intrinsics)
        
        self.kp = kp_valid
        self.desc = desc_valid
        
        descriptors_norm = np.linalg.norm(desc_valid, axis=1)
        descriptors_norm[descriptors_norm == 0] = 1.0
        self.desc_weights = np.var(desc_valid/descriptors_norm[:,None], axis=1)
        
        self.depth_pts = depth_points
        self.rgb = rgb_image
        self.intrinsics = intrinsics
        
        self.points_3D = points_3D
        self.colors_3D = colors_3D
        
        self.points_cloud = points_cloud
        
        
        self.R_mat= np.eye(3)       # Rotation of the pose
        self.t_vec = np.zeros((3, 1))   # Translation of the pose
        
        
        self.inliers = {}       # key: frame_id, value: inliers
        self.stats = {}         # key: frame_id, value: stats
        self.connections = {}   # key: frame_id, value: (R, T)
    
    
    def __repr__(self):
        return f"FrameNode({self.frame_id}) with {len(self.kp)} keypoints"
    
    
    def add_connection(self, frame_id, inlier_indices, R, T, stats):
        
        self.inliers[frame_id] = inlier_indices
        self.connections[frame_id] = (R, T)
        self.stats[frame_id] = stats



#############! Initialize the graph
def initialize_graph(kp_data, depth_data, rgb_data, intrinsics_data):
    
    kp_files = [file for file in kp_data.keys() if 'img' in file]
    
    nodes = np.zeros(len(kp_files), dtype=FrameNode)
    
    for i in range(len(kp_files)):
        
        kp = kp_data[kp_files[i]][0][0][0]
        desc = kp_data[kp_files[i]][0][0][1]
        
        depth = depth_data[..., i]
        rgb = rgb_data[..., i]
        intrinsics = intrinsics_data[i]

        nodes[i] = FrameNode(i, kp, desc, depth, rgb, intrinsics)
    
    return nodes


def compute_nodes_PNP(nodes):
    
    for i in  range(len(nodes)):
        
        ##! Compute the 3D points
        points3D = nodes[i].points_3D
        points2D = nodes[i].kp
        
        intrinsics = nodes[i].intrinsics
        
        inliers, R, t = alg.RANSAC_PNP(points3D, points2D, intrinsics)
        
        if inliers is None:
            continue
        
        nodes[i].R_mat = R
        nodes[i].t_vec = t


def transform_points(points, R, t):
    
    transformed_points = np.dot(R, points.T).T + t
    
    return transformed_points


def compute_edges_PNP(nodes, reference_frame=0):
    
    
    for i in range(len(nodes)):
        
        if i == reference_frame:
            continue
        
        R_mati = nodes[i].R_mat
        t_veci = nodes[i].t_vec
        
        R_mat_ref = nodes[reference_frame].R_mat
        t_vec_ref = nodes[reference_frame].t_vec
        
        ### transform the points to the reference frame
        R = np.dot(R_mat_ref, R_mati.T)
        t = t_vec_ref - np.dot(R, t_veci)
        
        ##! invert the transformation
        R_inv = R.T
        t_inv = -np.dot(R_inv, t)
        
        ## add the connection
        # nodes[reference_frame].add_connection(i, None, R, t, 0.0)
        # nodes[i].add_connection(reference_frame, None, R_inv, t_inv, 0.0)
        
        nodes[i].add_connection(reference_frame, None, R, t, 0.0)
        nodes[reference_frame].add_connection(i, None, R_inv, t_inv, 0.0)


#############! Compute the edges

###? compute the stats for each edge
def compute_stats(matches, points3D_1, points3D_2, A, t):
    
    transformed_pts = np.dot(A, points3D_1.T).T + t

    dists = np.linalg.norm(transformed_pts - points3D_2, axis=1)
    
    stats = {
        "inliers_ratio": len(matches) / len(points3D_1),
        "mean_error": np.mean(dists), 
        "variance_error": np.var(dists),
        "dists_error": np.sum(dists)
        }
    
    return stats


###? Similarity based connections
### KNN based on the weighted centroids
###? Direct connections to the reference frame
###? Temporal connections to the previous frame
### Use best_inliers_threshold to filter the connections

def compute_edges(nodes, PARAMS):
    
    reference_index = PARAMS['node_reference_index']
    num_neighbors = PARAMS['edges_num_neighbors']
    match_threshold = PARAMS['match_threshold']
    best_inliers_threshold = PARAMS['edges_inliers_threshold']
    
    
    ###! Similarity based connections
    centroids = []
    for node in nodes:
        
        kps = node.kp
        weights = node.desc_weights
        weights /= np.sum(weights)
        
        centroid = np.sum(kps * weights[:, None], axis=0)/len(kps)
        centroids.append(centroid)
    
    distances = scipy.spatial.distance_matrix(centroids, centroids)  # Compute distance between frames
    
    ### computed edges based on the matching optional algorithm
    for i in tqdm.tqdm(range(len(nodes)), desc="Computing edges"):
        nearest_neighbors = np.argsort(distances[i])[1:num_neighbors + 1]  # Exclude self (dist = 0)
        
        for j in nearest_neighbors:
            if i != j:                
                
                # matches = alg.matching_optional(nodes[i].desc, nodes[j].desc, match_threshold)
                
                # matches = alg.cross_check_matching(nodes[i].desc, nodes[j].desc)
                matches = alg.hybrid_matching(nodes[i].desc, nodes[j].desc)
                
                print("Matches: ", matches.shape)
                
                best_inliers = alg.MSAC(matches, nodes[i].points_3D, nodes[j].points_3D, PARAMS)
                
                
                ##! Check if there are enough inliers
                if len(best_inliers) < best_inliers_threshold:
                    continue
                
                points1 = nodes[i].points_3D[best_inliers[:, 0]]
                points2 = nodes[j].points_3D[best_inliers[:, 1]]
                
                A, t = alg.estimate_affine_transformation_svd(points1, points2)
                Ainv, tinv = alg.estimate_affine_transformation_svd(points2, points1)
                
                
                T = np.eye(4)
                T[:3, :3] = A
                T[:3, 3] = t.flatten()
                
                A, t = alg.iterative_closest_point(points1, points2, T)
                
                Tinv = np.eye(4)
                Tinv[:3, :3] = Ainv 
                Tinv[:3, 3] = tinv.flatten()
                
                Ainv, tinv = alg.iterative_closest_point(points2, points1, Tinv)
                
                
                stats = compute_stats(matches, points1, points2, A, t)
                
                nodes[i].add_connection(j, best_inliers, A, t, stats)
                nodes[j].add_connection(i, best_inliers, Ainv, tinv, stats)
        
        
        ###! Temporal connections to the previous frame
        # ##? always try the homography to the previous frame
        if i > 0 and i != reference_index and i-1 not in nearest_neighbors:
            
            j = i-1
            i1 = i
            
            # matches = alg.matching_optional(nodes[i1].desc, nodes[j].desc, match_threshold)
            
            # matches = alg.cross_check_matching(nodes[i1].desc, nodes[j].desc)
            
            matches = alg.hybrid_matching(nodes[i1].desc, nodes[j].desc)
            
            best_inliers = alg.MSAC(matches, nodes[i1].points_3D, nodes[j].points_3D, PARAMS)
            
            print("Best inliers: ", best_inliers.shape)
            
            ##! Check if there are enough inliers
            if len(best_inliers) < best_inliers_threshold:
                continue
                
            points1 = nodes[i1].points_3D[best_inliers[:, 0]]
            points2 = nodes[j].points_3D[best_inliers[:, 1]]
                
            A, t = alg.estimate_affine_transformation_svd(points1, points2)
            Ainv, tinv = alg.estimate_affine_transformation_svd(points2, points1)
            
            
            T = np.eye(4)
            T[:3, :3] = A
            T[:3, 3] = t.flatten()
                
            A, t = alg.iterative_closest_point(points1, points2, T)
                
            Tinv = np.eye(4)
            Tinv[:3, :3] = Ainv 
            Tinv[:3, 3] = tinv.flatten()
                
            Ainv, tinv = alg.iterative_closest_point(points2, points1, Tinv)
            
            
            stats = compute_stats(matches, points1, points2, A, t)
                
            nodes[i1].add_connection(j, best_inliers, A, t, stats)
            nodes[j].add_connection(i1, best_inliers, Ainv, tinv, stats)
        
        
        ###! Direct connections to the reference frame
        if i != reference_index and reference_index not in nearest_neighbors:
            
            j = reference_index
            i1 = i
            
            # matches = alg.matching_optional(nodes[i1].desc, nodes[j].desc, match_threshold)
            
            # matches = alg.cross_check_matching(nodes[i1].desc, nodes[j].desc)
            
            matches = alg.hybrid_matching(nodes[i1].desc, nodes[j].desc)
            
            best_inliers = alg.MSAC(matches, nodes[i1].points_3D, nodes[j].points_3D, PARAMS)
            
            print("Best inliers: ", best_inliers.shape)
            
            ##! Check if there are enough inliers
            if len(best_inliers) < best_inliers_threshold:
                continue
                
            points1 = nodes[i1].points_3D[best_inliers[:, 0]]
            points2 = nodes[j].points_3D[best_inliers[:, 1]]
                
            A, t = alg.estimate_affine_transformation_svd(points1, points2)
            Ainv, tinv = alg.estimate_affine_transformation_svd(points2, points1)
            
            
            T = np.eye(4)
            T[:3, :3] = A
            T[:3, 3] = t.flatten()
                
            A, t = alg.iterative_closest_point(points1, points2, T)
                
            Tinv = np.eye(4)
            Tinv[:3, :3] = Ainv 
            Tinv[:3, 3] = tinv.flatten()
                
            Ainv, tinv = alg.iterative_closest_point(points2, points1, Tinv)
            
            
            
            stats = compute_stats(matches, points1, points2, A, t)
                
            nodes[i1].add_connection(j, best_inliers, A, t, stats)
            nodes[j].add_connection(i1, best_inliers, Ainv, tinv, stats)
            
            
    return nodes



#############! Search in the graph to find the best path

### Not changes here

def dijkstra(graph, start_node, end_node):

    queue = [(0, start_node, [start_node])]
    visited = set()
    min_cost = {start_node: 0}

    while queue:
        current_cost, current_node, path = heapq.heappop(queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == end_node:
            return path, current_cost

        for neighbor, cost in graph.get(current_node, {}):
            if neighbor not in visited:
                new_cost = current_cost + cost
                if new_cost < min_cost.get(neighbor, float('inf')):
                    min_cost[neighbor] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor, path + [neighbor]))

    return None, float('inf')


def cost_function(stats, max_tuple):
    
    value = stats["mean_error"]
    return value


def compute_composite_transformations(nodes, PARAMS):
    
    reference_index = PARAMS['node_reference_index']
    
    num_nodes = len(nodes)
    composite_T = {reference_index: np.eye(4)}  
    
    path_lengths = np.zeros(num_nodes)
    path_costs = np.zeros(num_nodes)
    
    mean_error_max = 0.0
    variance_error_max = 0.0
    dists_error_max = 0.0
    
    for i, node in enumerate(nodes):
        for k in node.connections.keys():
            mean_error_max = max(mean_error_max, node.stats[k]["mean_error"])
            variance_error_max = max(variance_error_max, node.stats[k]["variance_error"])
            dists_error_max = max(dists_error_max, node.stats[k]["dists_error"])
    
    max_tuple = (mean_error_max, variance_error_max, dists_error_max)
    
    graph = dict()
    for i, node in enumerate(nodes):
        transitions = [(k, cost_function(nodes[i].stats[k], max_tuple)) for k in node.connections.keys()]
        graph[i] = transitions
    
    
    pbar = tqdm.tqdm(total=num_nodes, desc="Computing composite transformations")
    for node_id in range(num_nodes):
        
        pbar.update(1)
        
        if node_id == reference_index:
            continue
        
        path, cost = dijkstra(graph, node_id, reference_index)   
        
        ##! In case of no path, enforce connection to the previous frame
        if path is None:
            
            ## create edge to the previous frame
            j = node_id - 1
            i1 = node_id
            
            
            matches = alg.matching_optional(nodes[i1].desc, nodes[j].desc, match_threshold)
            best_inliers = alg.RANSAC(matches, nodes[i1].points_3D, nodes[j].points_3D, PARAMS)
                
            ##! Check if there are enough inliers
            if len(best_inliers) < best_inliers_threshold:
                continue
                
            points1 = nodes[i1].points_3D[best_inliers[:, 0]]
            points2 = nodes[j].points_3D[best_inliers[:, 1]]
                
            A, t = alg.estimate_affine_transformation_svd(points1, points2)
            Ainv, tinv = alg.estimate_affine_transformation_svd(points2, points1)
                
            stats = compute_stats(matches, points1, points2, A, t)
                
            nodes[i1].add_connection(j, best_inliers, A, t, stats)
            nodes[j].add_connection(i1, best_inliers, Ainv, tinv, stats)
            
            
            graph[i1].append((j, cost_function(stats, max_tuple)))
            
            path, cost = dijkstra(graph, node_id, reference_index)
        
        
        path.reverse()
        
        print("Path: ", path, "Cost: ", cost, "Node: ", node_id)
        
        path_lengths[node_id] = len(path)
        path_costs[node_id] = cost
        
        A_path = np.eye(3, dtype=np.float64)
        t_path = np.zeros((3, 1), dtype=np.float64)
        
        
        T_tensor = np.eye(4, dtype=np.float64)
        
        print(len(path))
        
        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i + 1]
            
            A, t = nodes[dst].connections[src]
            
            print("A: ", A)
            print("t: ", t)
            
            T_tensor_aux = np.zeros((4, 4), dtype=np.float64)
            T_tensor_aux[:3, :3] = A
            T_tensor_aux[:3, 3] = t.flatten()
            T_tensor_aux[3, 3] = 1.0
            
            T_tensor = np.dot(T_tensor, T_tensor_aux)
        
        
        print("T_tensor: ", T_tensor)
        composite_T[node_id] = T_tensor
        
        # H = np.identity(3, dtype=np.float64)
        
        # for i in range(len(path) - 1):
        #     src = path[i]
        #     dst = path[i + 1]
            
        #     H = np.dot(H, nodes[dst].connections[src])
        #     H /= H[2, 2]
        
        # composite_homographies[node_id] = H
    
    return composite_T, path_lengths, path_costs, graph





import matplotlib.pyplot as plt
import networkx as nx

def plot_graph(graph):
    G = nx.DiGraph()
    for node, transitions in graph.items():
        for neighbor, cost in transitions:
            G.add_edge(node, neighbor, weight=cost)
    
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=200, node_color='skyblue', font_size=8, font_color='darkblue')
    labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='black', font_size=3)
    plt.savefig("office/graph.png")