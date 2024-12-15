import numpy as np
import scipy
import algorithms as alg
import matplotlib.colors as mcolors
from PIL import Image



def image_to_matrix(image_path):
    img = Image.open(image_path)  # Abrir a imagem
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img_matrix = np.array(img)   # Converter em matriz numpy
    return img_matrix



def matrix_to_image(matrix, output_path):
    img = Image.fromarray(matrix)  # Converter a matriz numpy de volta para imagem
    img.save(output_path)          # Salvar a imagem



# def warp_perspective_full(src, H, dst):
#     width, height = (20000,10000)

#     for y in range(src.shape[0]):
#         for x in range(src.shape[1]):
#             p = H @ [x, y, 1]
#             px, py = int(p[0] / p[2] + 0.5), int(p[1] / p[2] + 0.5)

#             if 0 <= px+12000 < width and 0 <= py+6000 < height:
#                 dst[py+6000, px+12000] = src[y, x]

#     return dst


### vectirized version
def warp_perspective_full(src, H, dst):
    width, height = (20000, 10000)
    
    y_indices, x_indices = np.indices((src.shape[0], src.shape[1]))
    
    flat_x = x_indices.flatten()
    flat_y = y_indices.flatten()
    
    ones = np.ones_like(flat_x)
    homogeneous_coords = np.stack([flat_x, flat_y, ones], axis=0)
    
    transformed_coords = H @ homogeneous_coords
    transformed_coords /= transformed_coords[2, :]
    
    px = np.int32(transformed_coords[0, :] + 0.5)
    py = np.int32(transformed_coords[1, :] + 0.5)
    
    px += 12000
    py += 6000
    
    valid_indices = (0 <= px) & (px < width) & (0 <= py) & (py < height)
    
    for c in range(src.shape[2]):
        dst[py[valid_indices], px[valid_indices], c] = src[:, :, c].flat[valid_indices]
    
    return dst