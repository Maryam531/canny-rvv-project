import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# Use your actual copied image name
filename = 'test_image.jpg.jpeg'

if not os.path.exists(filename):
    print(f"Error: The file '{filename}' was not found in your directory.")
else:
    # Open the image, convert to standard grayscale, and resize for consistency
    img = Image.open(filename).convert('L').resize((256, 256))
    image_matrix = np.array(img)

    # Display the image cleanly in grayscale
    plt.figure(figsize=(6, 6))
    plt.imshow(image_matrix, cmap='gray', vmin=0, vmax=255)
    plt.title("Viewing Conan Test Image")
    plt.axis('off')  # Hide pixel coordinate axes
    plt.tight_layout()

    # Save it out where you can double-click it in Windows Explorer
    plt.savefig('output_view.png', dpi=150, bbox_inches='tight')
    print("Successfully processed! Open 'output_view.png' from Windows Explorer.")
