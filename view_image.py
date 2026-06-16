import numpy as np
import matplotlib.pyplot as plt
import os

# All project test images are 256x256 pixels
width = 256
height = 256

# Exact path to your project's test image folder
filename = 'images/lena_256x256.raw' 

if not os.path.exists(filename):
    print(f"Error: The file '{filename}' was not found.")
    print("Make sure this script is running from your main 'canny-rvv-project' directory.")
else:
    # Read the binary raw data as an 8-bit unsigned integer
    raw_data = np.fromfile(filename, dtype=np.uint8)
    
    try:
        # Reshape the flat 1D array into a 2D image matrix
        image_matrix = raw_data.reshape((height, width))
        
        # Display the image cleanly in grayscale
        plt.figure(figsize=(6, 6))
        plt.imshow(image_matrix, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Viewing Project Test Image: {os.path.basename(filename)}")
        plt.axis('off')  # Hide pixel coordinate axes
        plt.tight_layout()
        plt.show()
        
    except ValueError:
        print(f"Error: Could not reshape file. File size is {len(raw_data)} bytes.")
