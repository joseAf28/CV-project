import scipy.io

def load_data():
    """
    Load all necessary data (keypoints and cameras info).
    Define the reference frame using the first frame.

    :return: A tuple containing:
             - keypoints (raw dictionary from .mat file)
             - cams_info (raw data structure from .mat file)
             - reference_frame (raw dictionary for the first frame)
    """
    # Paths to data files
    kp_path = "data_sets/office/kp.mat"  # Update with the actual path
    cams_info_path = "data_sets/office/cams_info.mat"  # Update with the actual path

    print(f"Loading data from:\nKeypoints: {kp_path}\nCameras Info: {cams_info_path}")

    # Load keypoints (no unnecessary conversions)
    kp_data = scipy.io.loadmat(kp_path)
    keypoints = {
        key: {"kp": kp_data[key]['kp'][0, 0], "desc": kp_data[key]['desc'][0, 0]}
        for key in kp_data if not key.startswith("__")
    }
    print(f"Loaded {len(keypoints)} keypoints.")

    # Load camera info (raw data, no conversions)
    cams_info_data = scipy.io.loadmat(cams_info_path)
    cams_info = cams_info_data.get('cams_info')
    if cams_info is None:
        raise ValueError("Camera info missing: 'cams_info' key not found.")
    print(f"Loaded camera info for {cams_info.shape[1]} frames.")

    # Define the reference frame as the first frame (raw format)
    reference_frame = cams_info[0, 0]
    print("Reference frame defined.")

    return keypoints, cams_info, reference_frame
