import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the cover image and stego image
cover_image = cv2.imread('baboon.tif', cv2.IMREAD_GRAYSCALE)
stego_image = cv2.imread('stego_baboon.tiff', cv2.IMREAD_GRAYSCALE)

# Calculate the histograms
cover_hist = cv2.calcHist([cover_image], [0], None, [256], [0, 256])
stego_hist = cv2.calcHist([stego_image], [0], None, [256], [0, 256])

# Calculate the difference histogram
diff_hist = np.abs(cover_hist - stego_hist)

# Plotting the results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Cover Image
axes[0, 0].imshow(cover_image, cmap='gray')
axes[0, 0].set_title('a. Cover Image (X)')
axes[0, 0].axis('off')

# Histogram of Cover Image
axes[0, 1].plot(cover_hist, color='blue')
axes[0, 1].set_title('b. Histogram of Cover Image')
axes[0, 1].set_xlabel('Pixel Intensity')
axes[0, 1].set_ylabel('Frequency')

# Histogram of Stego Image
axes[1, 0].plot(stego_hist, color='blue')
axes[1, 0].set_title('c. Histogram of Stego Image (X\')')
axes[1, 0].set_xlabel('Pixel Intensity')
axes[1, 0].set_ylabel('Frequency')

# Distribution of Differences
axes[1, 1].plot(diff_hist, color='blue')
axes[1, 1].set_title('d. Distribution of Differences')
axes[1, 1].set_xlabel('Pixel Intensity Difference')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
