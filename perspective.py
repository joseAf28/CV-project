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



### vectirized version
def warp_perspective_full(src, H, dst):
    width, height = (2000, 1000)
    
    y_indices, x_indices = np.indices((src.shape[0], src.shape[1]))
    
    flat_x = x_indices.flatten()
    flat_y = y_indices.flatten()
    
    ones = np.ones_like(flat_x)
    homogeneous_coords = np.stack([flat_x, flat_y, ones], axis=0)
    
    transformed_coords = H @ homogeneous_coords
    transformed_coords /= transformed_coords[2, :]
    
    px = np.int32(transformed_coords[0, :] + 0.5)
    py = np.int32(transformed_coords[1, :] + 0.5)
    
    # px += 12000
    # py += 6000
    
    valid_indices = (0 <= px) & (px < width) & (0 <= py) & (py < height)
    
    for c in range(src.shape[2]):
        dst[py[valid_indices], px[valid_indices], c] = src[:, :, c].flat[valid_indices]
    
    return dst