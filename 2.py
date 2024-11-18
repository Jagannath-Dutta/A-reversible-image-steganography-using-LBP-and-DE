import numpy as np 
import math
import cv2
import random

def generate_random_secret_code(num_blocks):
    # Generate a list of random 8-bit numbers (0-255) for each 4x4 block
    return [random.randint(0, 255) for _ in range(num_blocks)]

def process_block(block, secret_code):
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

    return stego_block.astype(np.uint8)

def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Image not found or path is incorrect.")
    
    if image.shape != (512, 512):
        raise ValueError("Image must be 512x512 pixels.")

    num_blocks = (image.shape[0] // 4) * (image.shape[1] // 4)
    secret_codes = generate_random_secret_code(num_blocks)

    stego_image = np.zeros_like(image)

    idx = 0
    for i in range(0, image.shape[0], 4):
        for j in range(0, image.shape[1], 4):
            block = image[i:i + 4, j:j + 4]
            stego_block = process_block(block, secret_codes[idx])
            stego_image[i:i + 4, j:j + 4] = stego_block
            idx += 1

    return stego_image, secret_codes

# Example usage
image_path = "Baboon.tif"
stego_image, secret_codes = process_image(image_path)

# Save the stego image
cv2.imwrite("Baboon12.tif", stego_image)

# Print the first and last block of the stego image and first/last secret code
print("First 4x4 Block of Stego Image:")
print(stego_image[0:4, 0:4])  # First block

print("Last 4x4 Block of Stego Image:")
print(stego_image[-4:, -4:])   # Last block

print("First Secret Code (8 bits):", format(secret_codes[0], '08b'))
print("Last Secret Code (8 bits):", format(secret_codes[-1], '08b'))
