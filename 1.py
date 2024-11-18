import numpy as np
import math
import cv2

def extract_from_stego_image(stego_image):
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
                x_i = stego_img_flat[k]
                x_i1 = stego_img_flat[k + 1]
                v = math.floor((x_i + x_i1) / 2)
                d_dash = abs(x_i1 - x_i)
                w = int(bin(d_dash)[-1])
                extracted_bits.append(w)

                d = int(d_dash / 2)
                
                # Ensure values are within the valid range
                original_block[k] = np.clip(v - math.floor(d / 2), 0, 255)
                original_block[k + 1] = np.clip(v + math.ceil(d / 2), 0, 255)

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

# Example usage for extraction
stego_image_path = "Baboon12.tif"  # Make sure to use the correct path to the stego image
stego_image = cv2.imread(stego_image_path, cv2.IMREAD_GRAYSCALE)

if stego_image is None:
    raise ValueError("Stego image not found or path is incorrect.")

# Extract the original image and the secret code
original_image, secret_code = extract_from_stego_image(stego_image)

# Save the extracted original image
cv2.imwrite("Extracted_Baboon.tif", original_image)

# Optionally print first and last 4x4 blocks of the extracted image
print("First 4x4 Block of Extracted Original Image:")
print(original_image[0:4, 0:4])  # First block

print("Last 4x4 Block of Extracted Original Image:")
print(original_image[-4:, -4:])   # Last block
