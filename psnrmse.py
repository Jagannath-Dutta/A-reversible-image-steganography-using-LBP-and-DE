import cv2
import numpy as np
from math import log10, sqrt

def calculate_mse(imageA, imageB):
    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((imageA.astype(np.float64) - imageB.astype(np.float64)) ** 2)
    return mse
def calculate_psnr(mse,max_pixel_value=255.0):
    # Calculate the Peak Signal-to-Noise Ratio (PSNR)
   if mse == 0:  # Perfect match
       return float('inf')
   psnr = 10 * log10(pow(max_pixel_value, 2) / (mse))
   return psnr

def load_image(image_path):
    # Load an image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image

# Paths to cover and stego images
cover_image_path = 'baboon.tif'
stego_image_path = 'attacked_stego_image_brightening.tiff'

# Load images
cover_image = load_image(cover_image_path)
stego_image = load_image(stego_image_path)

# Calculate MSE
mse_value = calculate_mse(cover_image, stego_image)
print(f"MSE: {mse_value}")

# Calculate PSNR
psnr_value = calculate_psnr(mse_value)
print(f"PSNR: {psnr_value} dB")
