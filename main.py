import src
import cv2 as cv
import numpy as np
import scipy 
import imageio
import tqdm



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
    
    
    ###! assuming by now
    card = (900, 1280)
    
    
    len_track_coordinates = len(track_coordinates)
    
    idx_keys = track_coordinates.keys()
    
    for i in tqdm.tqdm(range(len_track_coordinates)):
        
        idx = list(idx_keys)[i]
        
        track_coordinates_one = track_coordinates[idx][:, :-1]
        
        track_coordinates_one_img = src.coordinates2image(track_coordinates_one, card)
        
        track_coordinates_maps = src.warpPerspective2(track_coordinates_one_img, H, card)
        
        imageio.imwrite(f'images/track_coordinates_maps_{idx}.jpg', track_coordinates_maps)
        imageio.imwrite(f'images/track_coordinates_video_{idx}.jpg', track_coordinates_one_img)
    