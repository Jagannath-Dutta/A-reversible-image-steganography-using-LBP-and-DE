# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 11:55:09 2024

@author: jagan
"""

import numpy as np
import math
import cv2
import random

def generate_random_secret_code(num_blocks):
    # Generate a list of random 8-bit numbers (0-255) for each 4x4 block
    return [random.randint(0, 255) for _ in range(num_blocks)]

def process_block(block, secret_code):
    # Convert block to a larger data type to prevent overflow
    block = block.astype(np.int16)

    # Step 2: Compute the average of the block values
    average_value = math.floor(np.mean(block))

    # Step 3: Compare each value with the average and represent it as 0 or 1
    binary_block = np.where(block > average_value, 0, 1)

    lbp = []

    # Step 5: Perform XOR operation on each pair of binary values in the block
    for row in binary_block:
        for i in range(0, len(row), 2):
            xor_result = row[i] ^ row[i + 1]
            lbp.append(xor_result)

    # Ensure the secret code slice matches the length of the lbp
    secret_code_slice = [int(b) for b in format(secret_code, '08b')]

    # Step 6: XOR the LBP list with the secret code slice
    xor_with_secret = [lbp[i] ^ secret_code_slice[i] for i in range(len(lbp))]

    # Step 7: Shuffle the result pairwise
    shuffled_result = xor_with_secret.copy()
    for i in range(0, len(shuffled_result), 2):
        if i + 1 < len(shuffled_result):
            shuffled_result[i], shuffled_result[i + 1] = shuffled_result[i + 1], shuffled_result[i]

    # Calculate d, v, and d_dash
    d_values = []
    v_values = []
    d_dash_values = []

    for i in range(0, block.size, 2):
        x_i = block.flat[i]
        x_i1 = block.flat[i + 1]

        # Calculate d with a large enough data type
        d = abs(int(x_i) - int(x_i1))
        d_values.append(d)

        # Calculate v
        v = math.floor((x_i + x_i1) / 2)
        v_values.append(v)

        # Calculate d_dash
        w = shuffled_result[i // 2]
        d_dash = d * 2 + w
        d_dash_values.append(d_dash)

    # Step 8: Modify the block to create the stego block
    stego_block = block.copy()
    for i in range(0, block.size, 2):
        x_i = block.flat[i]
        x_i1 = block.flat[i + 1]
        v = v_values[i // 2]
        d_dash = d_dash_values[i // 2]

        # Calculate the new pixel values while ensuring they stay within the 0-255 range
        new_x_i = max(0, min(255, v - math.floor(d_dash / 2)))
        new_x_i1 = max(0, min(255, v + math.ceil(d_dash / 2)))

        if x_i < x_i1:
            stego_block.flat[i] = new_x_i
            stego_block.flat[i + 1] = new_x_i1
        else:
            stego_block.flat[i] = new_x_i1
            stego_block.flat[i + 1] = new_x_i

    # Convert stego block back to uint8
    return stego_block.astype(np.uint8)

def process_image(image_path):
    # Load the grayscale image from the given file path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Image not found or path is incorrect.")
    
    # Ensure the image is 512x512
    if image.shape != (512, 512):
        raise ValueError("Image must be 512x512 pixels.")

    # Number of 4x4 blocks in the image
    num_blocks = (image.shape[0] // 4) * (image.shape[1] // 4)
    secret_codes = generate_random_secret_code(num_blocks)

    # Create an empty array for the stego image
    stego_image = np.zeros_like(image)

    # Process the image in 4x4 blocks
    idx = 0
    for i in range(0, image.shape[0], 4):
        for j in range(0, image.shape[1], 4):
            block = image[i:i+4, j:j+4]
            stego_block = process_block(block, secret_codes[idx])
            stego_image[i:i+4, j:j+4] = stego_block
            idx += 1

    return stego_image, secret_codes

def extract_from_stego_image(stego_image, secret_codes):
    # Initialize variables for the extraction process
    original_image = np.zeros_like(stego_image)
    extracted_bits = []

    # Process the stego image in 4x4 blocks
    for i in range(0, stego_image.shape[0], 4):
        for j in range(0, stego_image.shape[1], 4):
            stego_block = stego_image[i:i + 4, j:j + 4]
            stego_img_flat = stego_block.flatten()
            original_block = stego_img_flat.copy()

            # Apply difference expansion
            for k in range(0, stego_img_flat.size, 2):
                x_i = int(stego_img_flat[k])
                x_i1 = int(stego_img_flat[k + 1])
                v = np.floor((x_i + x_i1) / 2).astype(np.int16)  # Use int16
                d_dash = abs(x_i1 - x_i)
                w = int(bin(d_dash)[-1])
                extracted_bits.append(w)

                d = int(d_dash / 2)
                
                # Ensure values are within the valid range
                original_block[k] = np.clip(v - np.floor(d / 2), 0, 255)
                original_block[k + 1] = np.clip(v + np.ceil(d / 2), 0, 255)

            # Reshape the original block back to 4x4
            original_image[i:i + 4, j:j + 4] = original_block.reshape((4, 4))

    # Calculate average value, binary block, LBP, and final secret code
    avg_values = []
    binary_blocks = []
    lbp_list = []

    for i in range(0, original_image.size, 16):
        block = original_image.flatten()[i:i + 16].reshape((4, 4))
        average_value = math.floor(np.mean(block))
        avg_values.append(average_value)
        binary_block = np.where(block > average_value, 0, 1)
        binary_blocks.append(binary_block)

        lbp = []
        for row in binary_block:
            for k in range(0, len(row), 2):
                xor_result = row[k] ^ row[k + 1]
                lbp.append(xor_result)
        lbp_list.append(lbp)

    # Process extracted bits and LBP for final secret code
    final_secret_code = []
    for lbp, extracted_bits_block in zip(lbp_list, np.array_split(extracted_bits, len(lbp_list))):
        shuffled_result = extracted_bits_block.copy()
        
        # Swap elements pairwise
        for k in range(0, len(shuffled_result), 2):
            if k + 1 < len(shuffled_result):
                shuffled_result[k], shuffled_result[k + 1] = shuffled_result[k + 1], shuffled_result[k]

        # Final secret code calculation
        xor_with_shuffled = [lbp[i] ^ shuffled_result[i] for i in range(len(lbp))]
        final_secret_code.extend(xor_with_shuffled)

    return original_image.astype(np.uint8), final_secret_code


# Example usage
image_path = "Baboon.tif"
stego_image = process_image(image_path)

# Save the stego image
cv2.imwrite("Baboon_stego.tif", stego_image)

# Print the first and last 4x4 blocks of the stego image
print("First 4x4 Block of Stego Image:")
print(stego_image[0:4, 0:4])  # First block

print("Last 4x4 Block of Stego Image:")
print(stego_image[-4:, -4:])   # Last block

# Print the first and last 8 bits of the generated secret code
num_blocks = (stego_image.shape[0] // 4) * (stego_image.shape[1] // 4)
secret_codes = generate_random_secret_code(num_blocks)
print("First 8 bits of Secret Code:", format(secret_codes[0], '08b'))  # First 8 bits
print("Last 8 bits of Secret Code:", format(secret_codes[-1], '08b'))  # Last 8 bits


cover_image = cv2.imread('barbara.tif', cv2.IMREAD_GRAYSCALE)

stego_image = cv2.imread('barbara12.tif', cv2.IMREAD_GRAYSCALE)





def calculate_mse(imageA, imageB):
    # Ensure the images have the same dimensions
    assert imageA.shape == imageB.shape, "Images must have the same dimensions"
    # Calculate Mean Squared Error (MSE)
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def calculate_psnr(mse_value, max_pixel_value=255.0):
    if mse_value == 0:
        return float('inf')  # PSNR is infinite if MSE is zero (identical images)
    # Calculate Peak Signal-to-Noise Ratio (PSNR)
    psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse_value))
    return psnr_value

def load_image(image_path):
    # Load an image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image

# Paths to cover and stego images
"""cover_image_path = 'Baboon.tif'
stego_image_path = 'Baboon_stego.tif'

# Load images
cover_image = load_image(cover_image_path)
stego_image = load_image(stego_image_path)"""

# Calculate MSE
mse_value = calculate_mse(cover_image, stego_image)
print(f"MSE: {mse_value}")

# Calculate PSNR
psnr_value = calculate_psnr(mse_value)
print(f"PSNR: {psnr_value} dB")
