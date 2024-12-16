import numpy as np
import scipy 


print_flag = False

## getting homography algorithm

def getPerspectiveTransform(src, dst):
    if len(src) == len(dst):
        # Make homogeneous coordiates if necessary
        if src.shape[1] == 2:
            src = np.hstack((src, np.ones((len(src), 1), dtype=src.dtype)))
        if dst.shape[1] == 2:
            dst = np.hstack((dst, np.ones((len(dst), 1), dtype=dst.dtype)))

        # Solve 'Ax = 0'
        A = []
        for p, q in zip(src, dst):
            A.append([0, 0, 0, q[2]*p[0], q[2]*p[1], q[2]*p[2], -q[1]*p[0], -q[1]*p[1], -q[1]*p[2]])
            A.append([q[2]*p[0], q[2]*p[1], q[2]*p[2], 0, 0, 0, -q[0]*p[0], -q[0]*p[1], -q[0]*p[2]])
        _, _, Vt = np.linalg.svd(A, full_matrices=True)
        x = Vt[-1]

        # Reorganize `x` as a matrix
        H = x.reshape(3, -1) / x[-1] # Normalize the last element as 1
        return H




### matching algorithms


def matching_optional(desc1, desc2, threshold=0.25):
    
    distances = scipy.spatial.distance.cdist(desc1, desc2, 'euclidean') ## calculate the distances between each query descriptor and each train descriptor: i -> query, j -> train
    sorted_indices = np.argsort(distances, axis=1)[:, :2]
    
    first_min_indices = sorted_indices[:, 0]
    second_min_indices = sorted_indices[:, 1]

    first_min_distances = distances[np.arange(distances.shape[0]), first_min_indices]
    second_min_distances = distances[np.arange(distances.shape[0]), second_min_indices]

    condition = first_min_distances < threshold * second_min_distances
    matches = np.column_stack((np.where(condition)[0], first_min_indices[condition]))
    
    return matches



### ransac algorithm

def compute_RANSAC_iterations(epsilon, s=4, p=0.99):
    
    if epsilon == 1:  # Avoid division by zero in case all points are outliers
        return float('inf')
    if epsilon == 0:  # If there are no outliers, only 1 iteration is required
        return 1
    
    numerator = np.log(1 - p)
    denominator = np.log(1 - (1 - epsilon) ** s)
    N = numerator / denominator
    
    return int(np.ceil(N))  # Round up to the next integer



def RANSAC(matches, kp1, kp2, inlier_threshold=4.0, epsilon_init=0.6, max_iter=500):

    inlier_threshold = 2.0
    best_inliers = []
    
    number_iteractions = compute_RANSAC_iterations(epsilon=epsilon_init, s=4, p=0.999)
    counter = 0
    
    print("matches shape: ", matches.shape)
    if matches.shape[0] < 4:
        return best_inliers, None
    
    while True:
        
        sampled_matches = matches[np.random.choice(matches.shape[0], 4, replace=False)]
        
        H = getPerspectiveTransform(kp1[sampled_matches[:, 0]], kp2[sampled_matches[:, 1]])

        src_pts_hom = np.column_stack((kp1[matches[:, 0]], np.ones((matches.shape[0], 1)))).T
        transformed_pts_hom = (H @ src_pts_hom).T
        transformed_pts_hom /= transformed_pts_hom[:, 2][:, None]

        dst_pts_all = kp2[matches[:, 1]]
        dists = np.linalg.norm(transformed_pts_hom[:, :2] - dst_pts_all, axis=1)

        inliers = matches[dists < inlier_threshold]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            
            epsilon = 1 - len(best_inliers) / len(matches)
            number_iteractions = compute_RANSAC_iterations(epsilon=epsilon, s=4, p=0.99)
            
            
        if print_flag:
            print("Number of iteractions: ", number_iteractions," epsilon: ",  epsilon,  " best inliers: ", len(best_inliers) , " inliers nb: ", len(inliers))
        
        # if counter >= number_iteractions:
        #     break
        
        if counter > max_iter:
            break
        
        counter += 1
    # print("counter: ", counter)
    
    inliers_ratio = len(best_inliers) / len(matches)
    mean_error = np.mean(dists)
    variance_error = np.var(dists)
    
    stats = {
        "inliers_ratio": inliers_ratio,
        "inliers_nb": len(best_inliers),
        "mean_error": mean_error, 
        "variance_error": variance_error
        }
        
    return best_inliers, stats