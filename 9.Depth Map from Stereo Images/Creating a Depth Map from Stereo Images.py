import cv2
import numpy as np
from google.colab import files
from google.colab.patches import cv2_imshow

# Upload both left and right images
uploaded = files.upload()

# Get file names from uploaded dictionary
file_list = list(uploaded.keys())
if len(file_list) < 2:
    print("Please upload both Left and Right stereo images.")
else:
    # Load images as grayscale
    left_img = cv2.imdecode(np.frombuffer(uploaded[file_list[0]], np.uint8), cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imdecode(np.frombuffer(uploaded[file_list[1]], np.uint8), cv2.IMREAD_GRAYSCALE)

    # Validate successful loading
    if left_img is None or right_img is None:
        print("Error: Could not load both images.")
    else:
        # Resize images to have the same dimensions
        height = min(left_img.shape[0], right_img.shape[0])
        width = min(left_img.shape[1], right_img.shape[1])
        left_img = cv2.resize(left_img, (width, height))
        right_img = cv2.resize(right_img, (width, height))

        # Create StereoBM matcher
        stereo = cv2.StereoBM_create(numDisparities=16*3, blockSize=15)

        # Compute disparity map
        disparity = stereo.compute(left_img, right_img)

        # Normalize for display
        depth_map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = np.uint8(depth_map)

        # Show the depth map
        cv2_imshow(depth_map)
        print("Depth Map Displayed")
