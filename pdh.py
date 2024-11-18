import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the cover and stego images
cover_image_path = 'peppers.tiff'
stego_image_path = 'stego_peppers.tiff'

cover_image = cv2.imread(cover_image_path, cv2.IMREAD_GRAYSCALE)
stego_image = cv2.imread(stego_image_path, cv2.IMREAD_GRAYSCALE)

# Function to calculate pixel differences
def calculate_pixel_difference(image):
    diffs = []
    rows, cols = image.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            diffs.append(int(image[i, j]) - int(image[i + 1, j]))  # Vertical difference
            diffs.append(int(image[i, j]) - int(image[i, j + 1]))  # Horizontal difference
    return diffs

# Calculate pixel differences for cover and stego images
diffs_cover = calculate_pixel_difference(cover_image)
diffs_stego = calculate_pixel_difference(stego_image)

# Generate the histograms for the pixel differences
plt.figure(figsize=(10, 5))

plt.hist(diffs_cover, bins=50, alpha=0.5, label="Peppers Cover", color='red', density=True)
plt.hist(diffs_stego, bins=50, alpha=0.5, label="Peppers Stego", color='blue', density=True)

plt.xlabel("Pixel Difference")
plt.ylabel("Occurrences")
plt.legend()
plt.grid(True)
plt.show()
