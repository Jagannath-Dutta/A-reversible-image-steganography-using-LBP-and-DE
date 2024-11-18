import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import log10, sqrt
import random
from skimage.util import random_noise  # Add this line
from skimage.metrics import structural_similarity as compare_ssim

# Function to convert the secret image pixels into a binary bit stream
def image_to_binary_bit_stream(secret_image):
    # Flatten the image and convert each pixel to its 8-bit binary representation
    flattened_pixels = secret_image.flatten()
    bit_stream = ''.join([format(pixel, '08b') for pixel in flattened_pixels])
    bit_stream_arr = np.array([int(bit) for bit in bit_stream])
    return bit_stream_arr

# Function to convert the extracted bit stream back into a 128x128 image
def binary_bit_stream_to_image(extracted_bits, image_shape=(128, 128)):
    # Group every 8 bits into a byte and convert it back to a pixel value (0-255)
    pixel_values = [int(''.join(map(str, extracted_bits[i:i+8])), 2) for i in range(0, len(extracted_bits), 8)]
    return np.array(pixel_values).reshape(image_shape)

def embed(cover_image, secret, B=4):
    H, W = cover_image.shape
    num_blocks_H = H // B
    num_blocks_W = W // B
    stego_image = cover_image.copy()
    idx = 0  # To track the secret bits

    for i in range(num_blocks_H):
        for j in range(num_blocks_W):
            block = cover_image[i * B:(i + 1) * B, j * B:(j + 1) * B].flatten()

            for k in range(len(block)):
                if idx < len(secret):
                    # Modify the LSB of each pixel in the block to embed a secret bit
                    block[k] = (block[k] & ~1) | secret[idx]
                    idx += 1

            # Reshape block back and insert it into the stego image
            stego_image[i * B:(i + 1) * B, j * B:(j + 1) * B] = block.reshape(B, B)

    print(f'Total embedded bits: {idx}')
    return stego_image

def extract(stego_image, num_bits, B=4):
    H, W = stego_image.shape
    num_blocks_H = H // B
    num_blocks_W = W // B
    extracted_bits = []

    for i in range(num_blocks_H):
        for j in range(num_blocks_W):
            block = stego_image[i * B:(i + 1) * B, j * B:(j + 1) * B].flatten()

            for k in range(len(block)):
                if len(extracted_bits) < num_bits:
                    # Extract the LSB of each pixel
                    extracted_bits.append(block[k] & 1)

    return extracted_bits

def calculate_mse(original_image, stego_image):
    # Calculate the Mean Squared Error (MSE)
    mse = np.mean((original_image.astype(np.float64) - stego_image.astype(np.float64)) ** 2)
    return mse

def calculate_psnr(mse, max_pixel_value=255.0):
    # Calculate the Peak Signal-to-Noise Ratio (PSNR)
    if mse == 0:  # Perfect match
        return float('inf')
    psnr = 10 * log10(pow(max_pixel_value, 2) / (mse))
    return psnr

# Function to calculate SSIM between two images
def calculate_ssim(imageA, imageB):
    # Use skimage's structural_similarity function
    ssim, _ = compare_ssim(imageA, imageB, full=True)
    return ssim

# Updated display function to show all required images
def display_images(cover_image, stego_image, extracted_cover_image, secret_image_before, secret_image_after):
    plt.figure(figsize=(20, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(cover_image, cmap='gray')
    plt.title('Original Cover Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(stego_image, cmap='gray')
    plt.title('Stego Image (After Embedding)')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(extracted_cover_image, cmap='gray')
    plt.title('Extracted Cover Image')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(secret_image_before, cmap='gray')
    plt.title('Secret Image Before Embedding')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(secret_image_after, cmap='gray')
    plt.title('Secret Image After Extraction')
    plt.axis('off')

    plt.show()

# Function to apply Gaussian noise
def apply_gaussian_noise(image, mean=0, var=0.01):
    sigma = var**0.5
    gaussian = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gaussian * 255, 0, 255).astype(np.uint8)
    return noisy_image

# Example usage
cover_image = cv2.imread('baboon.tif', cv2.IMREAD_GRAYSCALE)
secret_image = cv2.imread('facebook.png', cv2.IMREAD_GRAYSCALE)

# Convert the secret image into a binary bit stream
secret_data = image_to_binary_bit_stream(secret_image)

# Embed the secret into the cover image
stego_image = embed(cover_image, secret_data)
cv2.imwrite('stego_image_with_secret.tiff', stego_image)

# Calculate and display MSE, PSNR, and SSIM values
mse_value = calculate_mse(cover_image, stego_image)
psnr_value = calculate_psnr(mse_value)
ssim_value = calculate_ssim(cover_image, stego_image)
print(f'MSE: {mse_value}')
print(f'PSNR: {psnr_value} dB')
print(f'SSIM: {ssim_value}')

print('-----------------------EXTRACTION--------------------------------')

# Apply Gaussian noise to the stego image (variance = 0.01)
noisy_stego_image = apply_gaussian_noise(stego_image, var=0.01)

# Save and display the noisy stego image
cv2.imwrite('noisy_stego_image_gaussian.tiff', noisy_stego_image)
plt.imshow(noisy_stego_image, cmap='gray')
plt.title('Stego Image with Gaussian Noise (Variance 0.01)')
plt.axis('off')
plt.show()

# Extract the secret bits from the noisy stego image
extracted_bits_from_noisy = extract(noisy_stego_image, len(secret_data))

# Convert the extracted bits back into a    28x128 image
extracted_secret_image_from_noisy = binary_bit_stream_to_image(extracted_bits_from_noisy)

# Display the original cover, noisy stego image, and extracted secret image after noise
display_images(cover_image, noisy_stego_image, cover_image, secret_image, extracted_secret_image_from_noisy)

# Save the extracted secret image after noise
cv2.imwrite('extracted_secret_image_from_noisy_gaussian.png', extracted_secret_image_from_noisy)

# Function to calculate the percentage of recovered data and Bit Error Rate (BER)
def calculate_recovery_percentage_and_ber(original_bits, extracted_bits):
    # Total number of bits (T_total)
    total_bits = len(original_bits)
    
    # Calculate the number of erroneous bits (Err)
    num_errors = np.sum(np.array(original_bits) != np.array(extracted_bits))
    
    # Calculate the recovery percentage
    num_correct_bits = total_bits - num_errors
    recovery_percentage = (num_correct_bits / total_bits) * 100
    
    # Calculate Bit Error Rate (BER)
    ber = num_errors / total_bits
    return recovery_percentage, ber

# Calculate the recovery percentage and BER after Gaussian noise
recovery_percentage, ber_value = calculate_recovery_percentage_and_ber(secret_data, extracted_bits_from_noisy)
print(f"Data Recovery Percentage after Gaussian Noise (Variance 0.01): {recovery_percentage:.2f}%")
print(f"Bit Error Rate (BER) after Gaussian Noise (Variance 0.01): {ber_value:.6f}")

# Function to apply Salt-and-Pepper noise
def apply_salt_and_pepper_noise(image, amount=0.1):
    noisy_image = random_noise(image, mode='s&p', amount=amount)
    noisy_image = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)
    return noisy_image



# Apply Salt-and-Pepper noise (10% noise)
noisy_salt_pepper_image = apply_salt_and_pepper_noise(stego_image, amount=0.1)

# Save and display the noisy stego image
cv2.imwrite('noisy_stego_image_salt_pepper.tiff', noisy_salt_pepper_image)
plt.imshow(noisy_salt_pepper_image, cmap='gray')
plt.title('Stego Image with Salt-and-Pepper Noise (10% noise)')
plt.axis('off')
plt.show()

# Extract the secret bits from the noisy Salt-and-Pepper image
extracted_bits_from_salt_pepper = extract(noisy_salt_pepper_image, len(secret_data))

# Convert the extracted bits back into a 128x128 image
extracted_secret_image_from_salt_pepper = binary_bit_stream_to_image(extracted_bits_from_salt_pepper)

# Display the original cover, noisy stego image, and extracted secret image after Salt-and-Pepper noise
display_images(cover_image, noisy_salt_pepper_image, cover_image, secret_image, extracted_secret_image_from_salt_pepper)

# Save the extracted secret image after Salt-and-Pepper noise
cv2.imwrite('extracted_secret_image_from_salt_pepper.png', extracted_secret_image_from_salt_pepper)

# Calculate the recovery percentage and BER after Salt-and-Pepper noise
recovery_percentage_salt_pepper, ber_salt_pepper = calculate_recovery_percentage_and_ber(secret_data, extracted_bits_from_salt_pepper)
print(f"Data Recovery Percentage after Salt-and-Pepper Noise (10%): {recovery_percentage_salt_pepper:.2f}%")
print(f"Bit Error Rate (BER) after Salt-and-Pepper Noise (10%): {ber_salt_pepper:.6f}")


# Function to apply Sharpening attack
def apply_sharpening_attack(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpening kernel
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


# Apply Sharpening attack
sharpened_stego_image = apply_sharpening_attack(stego_image)

# Save and display the sharpened stego image
cv2.imwrite('sharpened_stego_image.tiff', sharpened_stego_image)
plt.imshow(sharpened_stego_image, cmap='gray')
plt.title('Stego Image with Sharpening Attack')
plt.axis('off')
plt.show()

# Extract the secret bits from the sharpened stego image
extracted_bits_from_sharpened = extract(sharpened_stego_image, len(secret_data))

# Convert the extracted bits back into a 128x128 image
extracted_secret_image_from_sharpened = binary_bit_stream_to_image(extracted_bits_from_sharpened)

# Display the original cover, sharpened stego image, and extracted secret image after Sharpening attack
display_images(cover_image, sharpened_stego_image, cover_image, secret_image, extracted_secret_image_from_sharpened)

# Save the extracted secret image after Sharpening attack
cv2.imwrite('extracted_secret_image_from_sharpened.png', extracted_secret_image_from_sharpened)

# Calculate the recovery percentage and BER after Sharpening attack
recovery_percentage_sharpened, ber_sharpened = calculate_recovery_percentage_and_ber(secret_data, extracted_bits_from_sharpened)
print(f"Data Recovery Percentage after Sharpening Attack: {recovery_percentage_sharpened:.2f}%")
print(f"Bit Error Rate (BER) after Sharpening Attack: {ber_sharpened:.6f}")

import matplotlib.pyplot as plt
from skimage.util import random_noise

# Function to apply Speckle noise
def apply_speckle_noise(image, var=0.01):
    speckle = random_noise(image, mode='speckle', var=var)
    speckle_noisy_image = np.clip(speckle * 255, 0, 255).astype(np.uint8)
    return speckle_noisy_image

# Function to apply Cropping attack
def apply_cropping(image,block_size=100): 
   H, W = image.shape

    # Create a white block of specified size
   block = 255 * np.ones((block_size, block_size), dtype=np.uint8)
    
    # Add the block to the top-left corner of the image
   modified_image = image.copy()
   modified_image[0:block_size, 0:block_size] = block

   return modified_image

# Function to apply Rotation attack
def apply_rotation(image, angle=10):
    H, W = image.shape
    center = (W // 2, H // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (W, H), borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

# Function to apply Scaling (Resizing) attack
def apply_scaling(image, scale_factor=1.5):
    H, W = image.shape
    resized_image = cv2.resize(image, (int(W * scale_factor), int(H * scale_factor)))
    resized_image = cv2.resize(resized_image, (W, H))
    return resized_image

# Function to apply Blurring attack
def apply_blurring(image, kernel_size=(3, 3)):
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image

# Function to apply Histogram Equalization attack
def apply_histogram_equalization(image):
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

# Function to apply Median Filtering attack
def apply_median_filtering(image, kernel_size=3):
    median_filtered_image = cv2.medianBlur(image, kernel_size)
    return median_filtered_image

# Function to apply Combined Rotation and Scaling attack
def apply_rotation_scaling(image, angle=10, scale_factor=1.5):
    rotated_image = apply_rotation(image, angle)
    scaled_image = apply_scaling(rotated_image, scale_factor)
    return scaled_image
# Function to apply Opaque attack
def apply_opaque_attack(image, opacity_ratio=0.1):
    H, W = image.shape
    # Calculate the size of the opaque block
    block_height = int(H * np.sqrt(opacity_ratio))
    block_width = int(W * np.sqrt(opacity_ratio))
    # Determine the position for the block (center of the image)
    start_y = (H - block_height) // 2
    start_x = (W - block_width) // 2
    # Create a copy of the image and add a black block
    opaque_image = image.copy()
    opaque_image[start_y:start_y + block_height, start_x:start_x + block_width] = 0
    return opaque_image
# Function to apply Poisson noise
def apply_poisson_noise(image):
    poisson_noisy_image = np.random.poisson(image).astype(np.uint8)
    return poisson_noisy_image

# Function to apply Sharpening attack
def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image
# Function to apply Brightening attack
def apply_brightening(image, increase=30):
    brightened_image = np.clip(image + increase, 0, 255).astype(np.uint8)
    return brightened_image
# Function to apply an attack and extract the secret image
def test_attack(attack_func, stego_image, secret_data, attack_name, **kwargs):
    # Apply the attack
    attacked_stego_image = attack_func(stego_image, **kwargs)
    
    # Save the attacked stego image
    attacked_image_filename = f'attacked_stego_image_{attack_name}.tiff'
    cv2.imwrite(attacked_image_filename, attacked_stego_image)
    
    # Extract the secret from the attacked stego image
    extracted_bits = extract(attacked_stego_image, len(secret_data))
    extracted_secret_image = binary_bit_stream_to_image(extracted_bits)
    
    # Display the attacked stego image and extracted secret image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(attacked_stego_image, cmap='gray')
    plt.title(f'Stego Image with {attack_name} Attack')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(extracted_secret_image, cmap='gray')
    plt.title(f'Extracted Secret Image after {attack_name} Attack')
    plt.axis('off')
    plt.show()
    
    # Save the extracted secret image
    extracted_secret_filename = f'extracted_secret_{attack_name}.png'
    cv2.imwrite(extracted_secret_filename, extracted_secret_image)
    
    return attacked_image_filename, extracted_secret_filename

# Example usage: Apply each attack and extract the secret image
attack_functions = [
    (apply_speckle_noise, 'Speckle Noise', {'var': 0.01}),
    (apply_cropping, 'Cropping', {'block_size': 100}),
    (apply_rotation, 'Rotation', {'angle': 10}),
    (apply_scaling, 'Scaling', {'scale_factor': 1.5}),
    (apply_blurring, 'Blurring', {'kernel_size': (3, 3)}),
    (apply_histogram_equalization, 'Histogram Equalization', {}),
    (apply_median_filtering, 'Median Filtering', {'kernel_size': 3}),
    (apply_rotation_scaling, 'Rotation and Scaling', {'angle': 10, 'scale_factor': 1.5}),
    (apply_poisson_noise, 'Poisson Noise', {}),
    (apply_sharpening, 'Sharpening', {}),
    (apply_opaque_attack, 'Opaque', {'opacity_ratio': 0.1}),
    (apply_brightening, 'Brightening', {'increase': 30})
]

for func, name, params in attack_functions:
    test_attack(func, stego_image, secret_data, name, **params)
    
# Function to apply an attack, extract the secret image, and calculate BER and recovery percentage
def test_attack_with_metrics(attack_func, stego_image, secret_data, attack_name, **kwargs):
    # Apply the attack
    attacked_stego_image = attack_func(stego_image, **kwargs)
    
    # Save the attacked stego image
    attacked_image_filename = f'attacked_stego_image_{attack_name}.tiff'
    cv2.imwrite(attacked_image_filename, attacked_stego_image)
    
    # Extract the secret from the attacked stego image
    extracted_bits = extract(attacked_stego_image, len(secret_data))
    extracted_secret_image = binary_bit_stream_to_image(extracted_bits)
    
    # Calculate BER and recovery percentage
    recovery_percentage, ber_value = calculate_recovery_percentage_and_ber(secret_data, extracted_bits)
    
    # Display the attacked stego image and extracted secret image
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(attacked_stego_image, cmap='gray')
    plt.title(f'Stego Image with {attack_name} Attack')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(extracted_secret_image, cmap='gray')
    plt.title(f'Extracted Secret Image after {attack_name} Attack')
    plt.axis('off')
    plt.show()
    
    # Save the extracted secret image
    extracted_secret_filename = f'extracted_secret_{attack_name}.png'
    cv2.imwrite(extracted_secret_filename, extracted_secret_image)
    
    # Print BER and recovery percentage
    print(f"Data Recovery Percentage after {attack_name}: {recovery_percentage:.2f}%")
    print(f"Bit Error Rate (BER) after {attack_name}: {ber_value:.6f}")
    
    return recovery_percentage, ber_value

# Example usage: Apply each attack, extract the secret image, and calculate BER and recovery percentage
attack_functions = [
    (apply_speckle_noise, 'Speckle Noise', {'var': 0.01}),
    (apply_cropping, 'Cropping', {'block_size': 100}),
    (apply_rotation, 'Rotation', {'angle': 10}),
    (apply_scaling, 'Scaling', {'scale_factor': 1.5}),
    (apply_blurring, 'Blurring', {'kernel_size': (3, 3)}),
    (apply_histogram_equalization, 'Histogram Equalization', {}),
    (apply_median_filtering, 'Median Filtering', {'kernel_size': 3}),
    (apply_rotation_scaling, 'Rotation and Scaling', {'angle': 10, 'scale_factor': 1.5}),
    (apply_poisson_noise, 'Poisson Noise', {}),
    (apply_sharpening, 'Sharpening', {}),
    (apply_opaque_attack, 'Opaque', {'opacity_ratio': 0.1}),
    (apply_brightening, 'Brightening', {'increase': 30})
]

# Iterate over each attack and test it
results = []
for func, name, params in attack_functions:
    recovery_percentage, ber_value = test_attack_with_metrics(func, stego_image, secret_data, name, **params)
    results.append((name, recovery_percentage, ber_value))

# Display the results in a tabular format
import pandas as pd

results_df = pd.DataFrame(results, columns=['Attack Name', 'Recovery Percentage (%)', 'Bit Error Rate (BER)'])
print(results_df)


