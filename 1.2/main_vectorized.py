import numpy as np
from scipy.io import loadmat
import src
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import scipy
import tqdm



def plot_matches(img1, img2, kp1, kp2, matches, max_matches=20):

    if len(matches) > max_matches:
        np.random.seed(42)
        indices = np.random.choice(len(matches), max_matches, replace=False)
        matches = matches[indices]

    plt.figure(figsize=(15, 8))
    combined_img = np.hstack((img1, img2))
    plt.imshow(combined_img, cmap='gray')

    for match in matches:
        pt1 = kp1[match[0]]
        pt2 = kp2[match[1]] + [img1.shape[1], 0]
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', alpha=0.6)

    plt.title(f" {len(matches)} correspondências")
    plt.axis('off')
    plt.show()



def load_keypoints(file_path):
    data = loadmat(file_path)
    kp = data['kp']  #Keypoints (Nx2 matrix)
    desc = data['desc']  # Descritores(NxD matrix)
    return kp, desc

# Não estou a usar isto
def match_keypoints(desc1, desc2, threshold=0.15):
    distances = cdist(desc1, desc2)
    matches = []
    for i, row in enumerate(distances):
        sorted_indices = np.argsort(row)
        if row[sorted_indices[0]] < threshold * row[sorted_indices[1]]:
            matches.append((i, sorted_indices[0]))
    return np.array(matches)



def match_keypoints_ransac(desc1, desc2, kp1, kp2, threshold=0.25, max_iter=5000, inlier_threshold=2.0):
    #calcula a distncia entre os descritores
    distances = cdist(desc1, desc2)
    matches = []


    """
    matches = []
    for i, ref_vector in enumerate(ref_desc):
        distances = np.linalg.norm(frame_desc - ref_vector, axis=1)
        sorted_indices = np.argsort(distances)
        closest_idx = sorted_indices[0]
        second_closest_idx = sorted_indices[1]
        if distances[closest_idx] < ratio_threshold * distances[second_closest_idx]:
            matches.append([i, closest_idx])
    matches = np.array(matches)"""


    for i, row in enumerate(distances):
        sorted_indices = np.argsort(row)
        if row[sorted_indices[0]] < threshold * row[sorted_indices[1]]:
            matches.append((i, sorted_indices[0]))

    matches = np.array(matches)
    best_inliers = []
    best_H = None

    for _ in range(max_iter):
        #4 matches aleatórios (usar mais?)
        sampled_matches = matches[np.random.choice(matches.shape[0], 8, replace=False)]
        src_pts = kp1[sampled_matches[:, 0]]
        dst_pts = kp2[sampled_matches[:, 1]]

        H = src.getPerspectiveTransform(src_pts,dst_pts)
        
        inliers = []
        for match in matches:
            pt1 = kp1[match[0]]
            pt2 = kp2[match[1]]
            
            
            pt1_hom = np.append(pt1, 1)  
            transformed_pt = H @ pt1_hom
            transformed_pt /= transformed_pt[2]  
            
            
            dist = np.linalg.norm(transformed_pt[:2] - pt2)
            if dist < inlier_threshold:
                inliers.append(match)

        # Se o número de inliers for maior que o melhor encontrado, atualizar
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    return best_inliers, best_H




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



def ransac_vectorized(matches, kp1, kp2, max_iter=5000, inlier_threshold=2.0):

    inlier_threshold = 2.0
    best_inliers = []
    best_H = None
    
    
    for _ in range(max_iter):
        
        sampled_matches = matches[np.random.choice(matches.shape[0], 4, replace=False)]

        H = src.getPerspectiveTransform(kp1[sampled_matches[:, 0]], kp2[sampled_matches[:, 1]])

        src_pts_hom = np.column_stack((kp1[matches[:, 0]], np.ones((matches.shape[0], 1)))).T

        transformed_pts_hom = (H @ src_pts_hom).T
        transformed_pts_hom /= transformed_pts_hom[:, 2][:, None]

        dst_pts_all = kp2[matches[:, 1]]
        dists = np.linalg.norm(transformed_pts_hom[:, :2] - dst_pts_all, axis=1)

        inliers = matches[dists < inlier_threshold]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H
            
    return best_inliers, best_H



# Inicializar o mosaico
initial_image_path = 'ISRwall/input_1/images/img_0001.jpg'
initial_image = src.image_to_matrix(initial_image_path)
width, height = (20000,10000)
start=2000
dst = np.full((height, width, initial_image.shape[2] if initial_image.ndim == 3 else 1), 0, dtype=initial_image.dtype)

for y in range(initial_image.shape[0]):
    for x in range(initial_image.shape[1]):
        if 0 <= x+12000 < width and 0 <= y+6000 < height:
            dst[y+6000, x+12000] = initial_image[y, x]

src.matrix_to_image(dst, f"final_mosaic.jpg")



H_cumulative = np.eye(3)

for i in tqdm.tqdm(range(8)):
    
    kp1, desc1 = load_keypoints(f"ISRwall/input_1/keypoints/kp_000{i+1}.mat")
    kp2, desc2 = load_keypoints(f"ISRwall/input_1/keypoints/kp_000{i+2}.mat")


    #matches = match_keypoints(desc1, desc2)

    # inliers, H_ransac = match_keypoints_ransac(desc1, desc2, kp1, kp2)
    # matches = np.array(inliers)
    
    matches_init = matching_optional(desc1, desc2)
    inliers, H_ransac = ransac_vectorized(matches_init, kp1, kp2)

    matches = np.array(inliers)
    
    #print("Correspondências encontradas:")
    #print(matches)
    print(f"Total de correspondências: {len(matches)}")

    # Certifique-se de ter as imagens carregadas como arrays numpy
    # img1 = src.image_to_matrix(f"ISRwall/input_1/images/img_000{i+1}.jpg")
    img2 = src.image_to_matrix(f"ISRwall/input_1/images/img_000{i+2}.jpg")
    # plot_matches(img1, img2, kp1, kp2, np.array(inliers))
    
    # Obter keypoints correspondentes
    matched_kp1 = kp1[matches[:, 0]]
    matched_kp2 = kp2[matches[:, 1]]

    #print(matches.shape)
    #print(matched_kp1.shape)
    #print(matched_kp2.shape)

    # Calcular a homografia
    homography_matrix = np.linalg.inv(H_ransac)  #(troquei a ordem da homografia)
    #homography_matrix = src.getPerspectiveTransform(matched_kp2, matched_kp1)

    H_cumulative = np.dot(H_cumulative, homography_matrix)

    dst = src.warp_perspective_full(img2, H_cumulative, dst)

    src.matrix_to_image(dst, f"final_mosaic{i}.jpg")

    # Imprimir resultados
    print("Matriz de homografia estimada:")
    print(homography_matrix)