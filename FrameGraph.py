import numpy as np
import scipy
from scipy.spatial import Delaunay
import algorithms as alg
import tqdm
import heapq


#############! Load Data from mat files

def load_keypoints(file_path):
    data = scipy.io.loadmat(file_path)
    
    if 'kp' not in data or 'desc' not in data:
        return None, None
    else:
        kp = data['kp']  #Keypoints (Nx2 matrix)
        desc = data['desc']  # Descritores(NxD matrix)
        return kp, desc


def load_yolo(file_path):
    data = scipy.io.loadmat(file_path)
    
    if 'xyxy' not in data or 'id' not in data:
        return None, None
    else:
        data_xyxy = data['xyxy'] 
        data_id = data['id']    
        return data_xyxy, data_id
    


#############! FrameNode class
class FrameNode:
    
    def __init__(self, frame_id, keypoints, descriptors):
        self.frame_id = frame_id
        self.keypoints = keypoints
        self.descriptors = descriptors
        descriptors_norm = np.linalg.norm(descriptors, axis=1)
        descriptors_norm[descriptors_norm == 0] = 1.0
        self.descriptor_weights = np.var(descriptors/descriptors_norm[:,None], axis=1)
        
        
        self.inliers = {}       # key: frame_id, value: inliers
        self.stats = {}         # key: frame_id, value: stats
        self.connections = {}    # key: frame_id, value: homographyxw
    
    
    def __repr__(self):
        return f"FrameNode({self.frame_id}) with {len(self.keypoints)} keypoints"
    
    
    def add_connection(self, frame_id, inlier_indices, homography, stats):
        
        self.inliers[frame_id] = inlier_indices
        self.connections[frame_id] = homography
        self.stats[frame_id] = stats



#############! Initialize the graph
def initialize_graph(reference_path, frames_path):
    
    nodes = np.zeros(len(frames_path)+1, dtype=FrameNode)
    
    kp, desc = load_keypoints(reference_path)
    nodes[0] = FrameNode(0, kp, desc)
    
    
    for i, frame in enumerate(frames_path):
        kp, desc = load_keypoints(frame)
        
        nodes[i+1] = FrameNode(i, kp, desc)
    
    return nodes




#############! Compute the edges

###? compute the stats for each edge
def compute_stats(matches, kp1, kp2, H):
    
    src_pts_hom = np.column_stack((kp1, np.ones((kp1.shape[0], 1)))).T
    transformed_pts_hom = (H @ src_pts_hom).T
    transformed_pts_hom /= transformed_pts_hom[:, 2][:, None]

    dists = np.linalg.norm(transformed_pts_hom[:, :2] - kp2, axis=1)
    
    stats = {
        "inliers_ratio": len(matches) / len(kp1),
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
        
        kps = node.keypoints
        weights = node.descriptor_weights
        weights /= np.sum(weights)
        
        centroid = np.sum(kps * weights[:, None], axis=0)/len(kps)
        centroids.append(centroid)
    
    distances = scipy.spatial.distance_matrix(centroids, centroids)  # Compute distance between frames
    
    ### computed edges based on the matching optional algorithm
    for i in tqdm.tqdm(range(len(nodes)), desc="Computing edges"):
        nearest_neighbors = np.argsort(distances[i])[1:num_neighbors + 1]  # Exclude self (dist = 0)
        
        for j in nearest_neighbors:
            if i != j:                
                
                matches = alg.matching_optional(nodes[i].descriptors, nodes[j].descriptors, match_threshold)
                best_inliers = alg.MSAC(matches, nodes[i].keypoints, nodes[j].keypoints, PARAMS)
                
                ##! Check if there are enough inliers
                if len(best_inliers) < best_inliers_threshold:
                    continue
                
                kp1 = nodes[i].keypoints[best_inliers[:, 0]]
                kp2 = nodes[j].keypoints[best_inliers[:, 1]]
                
                H = alg.getPerspectiveTransform(kp1, kp2)
                H_inv = alg.getPerspectiveTransform(kp2, kp1)
                
                stats = compute_stats(matches, kp1, kp2, H)
                
                nodes[i].add_connection(j, best_inliers, H, stats)
                nodes[j].add_connection(i, best_inliers, H_inv, stats)
        
        
        ###! Temporal connections to the previous frame
        # ##? always try the homography to the previous frame
        if i > 0 and i != reference_index and i-1 not in nearest_neighbors:
            
            j = i-1
            i1 = i
            
            matches = alg.matching_optional(nodes[i1].descriptors, nodes[j].descriptors, match_threshold)
            best_inliers = alg.MSAC(matches, nodes[i1].keypoints, nodes[j].keypoints, PARAMS)
            
            if len(best_inliers) < best_inliers_threshold:
                continue
            
            kp1 = nodes[i1].keypoints[best_inliers[:, 0]]
            kp2 = nodes[j].keypoints[best_inliers[:, 1]]
                
            H = alg.getPerspectiveTransform(kp1, kp2)
            H_inv = alg.getPerspectiveTransform(kp2, kp1)
            
            stats = compute_stats(matches, kp1, kp2, H)
            
            nodes[i1].add_connection(j, best_inliers, H, stats)
            nodes[j].add_connection(i1, best_inliers, H_inv, stats)
        
        
        ###! Direct connections to the reference frame
        if i != reference_index and reference_index not in nearest_neighbors:
            
            j = reference_index
            i1 = i
            
            matches = alg.matching_optional(nodes[i1].descriptors, nodes[j].descriptors, match_threshold)
            best_inliers = alg.MSAC(matches, nodes[i1].keypoints, nodes[j].keypoints, PARAMS)
            
            if len(best_inliers) < best_inliers_threshold:
                continue
            
            kp1 = nodes[i1].keypoints[best_inliers[:, 0]]
            kp2 = nodes[j].keypoints[best_inliers[:, 1]]
                
            H = alg.getPerspectiveTransform(kp1, kp2)
            H_inv = alg.getPerspectiveTransform(kp2, kp1)
            
            stats = compute_stats(matches, kp1, kp2, H)
            
            nodes[i1].add_connection(j, best_inliers, H, stats)
            nodes[j].add_connection(i1, best_inliers, H_inv, stats)
            
            
    return nodes



#############! Search in the graph to find the best path



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
    
    # alpha = 0.9
    # beta = 0.1
    # value = alpha * stats["mean_error"]/max_tuple[0] + beta * stats["variance_error"]/max_tuple[1]
    
    value = stats["mean_error"]
    return value


def compute_composite_homographies(nodes, PARAMS):
    
    reference_index = PARAMS['node_reference_index']
    
    num_nodes = len(nodes)
    composite_homographies = {reference_index: np.eye(3)}  
    
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
    
    
    pbar = tqdm.tqdm(total=num_nodes, desc="Computing composite homographies")
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
            
            matches = alg.matching_optional(nodes[i1].descriptors, nodes[j].descriptors, PARAMS['match_threshold'])
            best_inliers = alg.RANSAC(matches, nodes[i1].keypoints, nodes[j].keypoints, PARAMS)
            
            kp1 = nodes[i1].keypoints[best_inliers[:, 0]]
            kp2 = nodes[j].keypoints[best_inliers[:, 1]]
                
            H = alg.getPerspectiveTransform(kp1, kp2)
            H_inv = alg.getPerspectiveTransform(kp2, kp1)
            
            stats = compute_stats(matches, kp1, kp2, H)
            
            nodes[i1].add_connection(j, best_inliers, H, stats)
            nodes[j].add_connection(i1, best_inliers, H_inv, stats)
            
            graph[i1].append((j, cost_function(stats, max_tuple)))
            
            path, cost = dijkstra(graph, node_id, reference_index)
        
        
        path.reverse()
        
        path_lengths[node_id] = len(path)
        path_costs[node_id] = cost
        
        H = np.identity(3, dtype=np.float64)
        
        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i + 1]
            
            H = np.dot(H, nodes[dst].connections[src])
            H /= H[2, 2]
        
        composite_homographies[node_id] = H
    
    
    return composite_homographies, path_lengths, path_costs, graph





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
    plt.savefig("volley/graph.png")