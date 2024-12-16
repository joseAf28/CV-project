import numpy as np
import scipy
from scipy.spatial import Delaunay
import algorithms as alg


print_flag = True


def load_keypoints(file_path):
    data = scipy.io.loadmat(file_path)
    kp = data['kp']  #Keypoints (Nx2 matrix)
    desc = data['desc']  # Descritores(NxD matrix)
    return kp, desc



#############! FrameNode class
class FrameNode:
    
    def __init__(self, frame_id, keypoints, descriptors):
        self.frame_id = frame_id
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.descriptor_weights = np.var(descriptors, axis=1)
        
        self.inliers = {} # key: frame_id, value: inliers
        self.stats = {} # key: frame_id, value: stats
        self.connections = {} # key: frame_id, value: homographyxw
    
    
    def __repr__(self):
        return f"FrameNode({self.frame_id}) with {len(self.keypoints)} keypoints"
    
    
    def add_connection(self, frame_id, inlier_indices, homography, stats):
        
        self.inliers[frame_id] = inlier_indices
        self.connections[frame_id] = homography
        self.stats[frame_id] = stats



#############! Initialize the graph
def initialize_graph(frames_path):
    
    nodes = np.zeros(len(frames_path), dtype=FrameNode)
    
    for i, frame in enumerate(frames_path):
        kp, desc = load_keypoints(frame)
        
        nodes[i] = FrameNode(i, kp, desc)
    
    return nodes



#############! Compute the edges

###? Compute the edges between the frames using the previous frame
def compute_edges(nodes, threshold=0.25):
    
    for i in range(len(nodes) - 1):
        
        j = i + 1
        
        matches = alg.matching_optional(nodes[i].descriptors, nodes[j].descriptors, threshold)
        best_inliers, stats = alg.RANSAC(matches, nodes[i].keypoints, nodes[j].keypoints)
        kp1 = nodes[i].keypoints[best_inliers[:, 0]]
        kp2 = nodes[j].keypoints[best_inliers[:, 1]]
        
        H = alg.getPerspectiveTransform(kp1, kp2)
        H_inv = alg.getPerspectiveTransform(kp2, kp1)
        
        nodes[i].add_connection(j, best_inliers, H, stats)
        nodes[j].add_connection(i, best_inliers, H_inv, stats)
    
    
    return nodes


###? Compute the edges between the frames using the proximity of the frames
### Weighted Centroids
### Use best_inliers_threshold to filter the connections
def compute_proximity_edges(nodes, num_neighbors=3, threshold2=0.3, best_inliers_threshold=6):

    centroids = []
    for node in nodes:
        
        kps = node.keypoints
        weights = node.descriptor_weights
        weights /= np.sum(weights)
        
        centroid = np.sum(kps * weights[:, None], axis=0)/len(kps)
        centroids.append(centroid)

    distances = scipy.spatial.distance_matrix(centroids, centroids)  # Compute distance between frames
    
    ### computed edges based on the matching optional algorithm
    for i in range(len(nodes)):
        nearest_neighbors = np.argsort(distances[i])[1:num_neighbors + 1]  # Exclude self (dist = 0)
        
        for j in nearest_neighbors:
            if i != j:                
                
                matches = alg.matching_optional(nodes[i].descriptors, nodes[j].descriptors)
                best_inliers, stats = alg.RANSAC(matches, nodes[i].keypoints, nodes[j].keypoints)
                
                ##! Check if there are enough inliers
                if len(best_inliers) < best_inliers_threshold:
                    continue
                
                kp1 = nodes[i].keypoints[best_inliers[:, 0]]
                kp2 = nodes[j].keypoints[best_inliers[:, 1]]
                
                H = alg.getPerspectiveTransform(kp1, kp2)
                H_inv = alg.getPerspectiveTransform(kp2, kp1)
                
                nodes[i].add_connection(j, best_inliers, H, stats)
                nodes[j].add_connection(i, best_inliers, H_inv, stats)
        
        if print_flag:
            print("i:", i, "nearest_neighbors:", nearest_neighbors)
            
        ##? Additional connections in case of no edges (?)
        
        ## if there is no connection between the frames, use the previous frame
        no_edges_flag = len(nodes[i].connections) == 0
        
        if no_edges_flag and i > 0:
            
            if print_flag:
                print("used previous frame")
                
            j = i - 1
            i1 = i
        
            matches = alg.matching_optional(nodes[i1].descriptors, nodes[j].descriptors, threshold2)
            best_inliers, stats = alg.RANSAC(matches, nodes[i1].keypoints, nodes[j].keypoints)
            kp1 = nodes[i1].keypoints[best_inliers[:, 0]]
            kp2 = nodes[j].keypoints[best_inliers[:, 1]]
            
            H = alg.getPerspectiveTransform(kp1, kp2)
            H_inv = alg.getPerspectiveTransform(kp2, kp1)
            
            nodes[i1].add_connection(j, best_inliers, H, stats)
            nodes[j].add_connection(i1, best_inliers, H_inv, stats)
        
        
        if no_edges_flag and i < len(nodes) - 1:
            
            if print_flag:
                print("used next frame")
                
            j = i + 1
            i1 = i
        
            matches = alg.matching_optional(nodes[i1].descriptors, nodes[j].descriptors, threshold2)
            best_inliers, stats = alg.RANSAC(matches, nodes[i1].keypoints, nodes[j].keypoints)
            kp1 = nodes[i1].keypoints[best_inliers[:, 0]]
            kp2 = nodes[j].keypoints[best_inliers[:, 1]]
            
            H = alg.getPerspectiveTransform(kp1, kp2)
            H_inv = alg.getPerspectiveTransform(kp2, kp1)
            
            nodes[i1].add_connection(j, best_inliers, H, stats)
            nodes[j].add_connection(i1, best_inliers, H_inv, stats)
        
        
        ##? always try the homography to the first frame
        if i > 0:
            j = 0
            i1 = i
            
            matches = alg.matching_optional(nodes[i1].descriptors, nodes[j].descriptors, threshold2)
            best_inliers, stats = alg.RANSAC(matches, nodes[i1].keypoints, nodes[j].keypoints)
            
            if len(best_inliers) < best_inliers_threshold:
                continue
            
            kp1 = nodes[i1].keypoints[best_inliers[:, 0]]
            kp2 = nodes[j].keypoints[best_inliers[:, 1]]
                
            H = alg.getPerspectiveTransform(kp1, kp2)
            H_inv = alg.getPerspectiveTransform(kp2, kp1)
                
            nodes[i1].add_connection(j, best_inliers, H, stats)
            nodes[j].add_connection(i1, best_inliers, H_inv, stats)
    
    
    return nodes


###? Compute the edges between the frames using the Delaunay triangulation
### Seems not to be working very well, but in principle could be the best approach
def compute_delaunay_edges(nodes, threshold=0.5):
    
    centroids = [np.mean(node.keypoints, axis=0) for node in nodes]     # Centroids of keypoints
    triangulation = Delaunay(centroids)                                 # Perform Delaunay triangulation
    
    if print_flag:
        print("triangulation:", triangulation)
        print("simplices:", triangulation.simplices)
        print("shape: ", triangulation.simplices.shape)
        
    for simplex in triangulation.simplices:
        i, j, k = simplex  # Each simplex is a triangle with 3 points
        for (a, b) in [(i, j), (j, k), (k, i)]:
            if a != b:
                
                matches = alg.matching_optional(nodes[a].descriptors, nodes[b].descriptors)
                best_inliers, stats = alg.RANSAC(matches, nodes[a].keypoints, nodes[b].keypoints)
                
                
                ##! Start imposing stats
                ## Check if there are inliers
                if len(best_inliers) < 6:
                    continue
                
                kp1 = nodes[a].keypoints[best_inliers[:, 0]]
                kp2 = nodes[b].keypoints[best_inliers[:, 1]]
                
                H = alg.getPerspectiveTransform(kp1, kp2)
                H_inv = alg.getPerspectiveTransform(kp2, kp1)
                
                nodes[a].add_connection(b, best_inliers, H, stats)
                nodes[b].add_connection(a, best_inliers, H_inv, stats)
        
        print("simplex:", simplex)
        
    return nodes



#############! Search in the graph to find the best path

##! search in the graph to find the best path
def bfs_shortest_path(graph, start_node):
    
    queue = [start_node]
    parent = {start_node: None}  # Track the parent of each node
    while queue:
        current_node = queue.pop(0)
        
        if print_flag:
            print("current_node:", current_node)
            print("connections:", graph[current_node].connections)
            print("neighbors:", graph[current_node].connections.keys())
            print("parent:", parent)
            print("queue:", queue)
            print()
        
        
        for neighbor in graph[current_node].connections.keys():
            if neighbor not in parent:  # If not visited
                parent[neighbor] = current_node
                queue.append(neighbor)
    return parent


def compute_composite_homographies_bfs(nodes, reference_index=0):
    
    num_nodes = len(nodes)
    composite_homographies = {reference_index: np.eye(3)}  
    graph = {i: node for i, node in enumerate(nodes)} 


    parent = bfs_shortest_path(graph, reference_index)
    
    print("parent: ", parent)
    
    ## Accumulate homographies for each frame relative to the reference
    for frame_id in range(num_nodes):
        if frame_id == reference_index:
            continue
        path = []
        current_node = frame_id
        while current_node is not None:
            path.append(current_node)
            current_node = parent[current_node]
        path.reverse()  # Reverse to start from the reference frame
        
        # Compute the composite homography for this path
        H = np.eye(3)
        print("path:", path, "frame_id:", frame_id)
        
        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i + 1]
            
            H = H @ graph[dst].connections[src]
        
        composite_homographies[frame_id] = H
    
    return composite_homographies




###! Using Depth First Search to find all paths
def dfs_all_paths(graph, start_node, end_node):
    stack = [(start_node, [start_node])]
    all_paths = []

    while stack:
        (current_node, path) = stack.pop()

        for neighbor in graph.get(current_node, []):
            if neighbor in path:
                continue
            new_path = path + [neighbor]
            if neighbor == end_node:
                all_paths.append(new_path)
            else:
                stack.append((neighbor, new_path))

    return all_paths


###! Compute the composite homographies using Depth First Search
### Uses the path path that minimizes the mean error of the RANSAC
###? To be improved later
def compute_composite_homographies_dfs(nodes, reference_index=0):
    
    num_nodes = len(nodes)
    composite_homographies = {reference_index: np.eye(3)}  
    graph = {i: list(node.connections.keys()) for i, node in enumerate(nodes)} 


    for node_id in range(num_nodes):
        
        if node_id == reference_index:
            continue
        
        paths = dfs_all_paths(graph, node_id, reference_index)
        
        ## Find the path that minimizes the mean error
        mean_error_path = []
        for path in paths:
            error = 0
            for i in range(len(path) - 1):
                src = path[i]
                dst = path[i + 1]
                error += nodes[dst].stats[src]["mean_error"]
            mean_error_path.append(error)
        
        if print_flag:
            print("mean_error_path: ", mean_error_path)
            print("paths: ", paths)
            print()
        
        min_error_index = np.argmin(mean_error_path)
        
        path = paths[min_error_index]
        path.reverse()
        

        # Compute the composite homography for this path
        H = np.eye(3)
        
        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i + 1]
            
            H = H @ nodes[dst].connections[src]
        
        composite_homographies[node_id] = H
    
    return composite_homographies