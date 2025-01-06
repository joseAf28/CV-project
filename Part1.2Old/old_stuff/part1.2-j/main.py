import src
import src
import cv2 as cv
import numpy as np
import scipy 
import imageio
import tqdm
import matplotlib.pyplot as plt



if __name__ == "__main__":
    
    ### Compute the homography matrix
    
    ##! Later improve homography with RANSAC
    
    ## src: pixel_coordinates
    ## dst: maps_coordinates
    
    folder_name = "CTownAirport_1.1"
    
    data_maps = scipy.io.loadmat(f'{folder_name}/kp_gmaps.mat')
    data_maps_vec = data_maps['kp_gmaps']
    
    
    pixel_coordinates = data_maps_vec[:, 0:2]
    maps_coordinates = data_maps_vec[:, 2:]

    H = src.getPerspectiveTransform(pixel_coordinates, maps_coordinates)
    
    
    ### Load the YOLO data
    directory_path = f"{folder_name}/yolo"
    track_coordinates = src.load_yolo(directory_path)
    
    
    ###! Assumption
    card_src = (900, 1280)
    card_dst = (900, 1280)
    
    
    len_track_coordinates = len(track_coordinates)
    idx_keys = track_coordinates.keys()
    
    n = len(idx_keys)
    colors = plt.cm.tab20(np.linspace(0, 1, n))  # Use 'tab20' colormap to generate 5 colors
    
    
    image_all_combined_video = np.zeros((card_src[1], card_src[0], 3), dtype=np.uint8)
    image_all_combined_map = np.zeros((card_dst[1], card_dst[0], 3), dtype=np.uint8)
    
    
    for i in tqdm.tqdm(range(len_track_coordinates)):
        
        idx = list(idx_keys)[i]
        color_one = colors[i, :-1]*255
        
        track_coordinates_one = track_coordinates[idx][:, :-1]
        
        track_coordinates_one_img = src.coordinates2image(track_coordinates_one, card_src, color_one)
        
        track_coordinates_maps = src.warpPerspective2(track_coordinates_one_img, H, card_dst, color_one)
        
        imageio.imwrite(f'images/track_coordinates_maps_{idx}.jpg', track_coordinates_maps)
        imageio.imwrite(f'images/track_coordinates_video_{idx}.jpg', track_coordinates_one_img)
        
        image_all_combined_map += track_coordinates_maps
        image_all_combined_video += track_coordinates_one_img
        
    
    imageio.imwrite(f'images/track_coordinates_maps_all.jpg', image_all_combined_map)
    imageio.imwrite(f'images/track_coordinates_video_all.jpg', image_all_combined_video)