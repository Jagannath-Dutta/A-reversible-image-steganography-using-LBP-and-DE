# -- coding: utf-8 --
"""
Created on Sat Sep 28 20:40:47 2024

@author: jagan
"""

import numpy as np
import math

# Given 4x4 stego block
stego_block = np.array([
    [157,  71,  83,  33],
     [103,  56,  71,  52],
     [124,  46,  80,  48],
     [ 68, 131,  59,  57]
])

# Flatten the block for easier processing
stego_img = stego_block.flatten()

# Initialize variables
extracted_bits = []
original_block = stego_img.copy()

# Apply difference expansion
for i in range(0, stego_img.size, 2):
    x_i = stego_img[i]
    x_i1 = stego_img[i + 1]
    v = math.floor((x_i + x_i1) / 2)
    d_dash = abs(x_i1 - x_i)
    w = int(bin(d_dash)[-1])
    extracted_bits.append(w)
    d = int(d_dash / 2)
    if x_i < x_i1:
        original_block[i] = v - math.floor (d / 2)
        original_block[i + 1] = v + math.ceil(d / 2)
    else:
        original_block[i] = v + math.ceil(d / 2)
        original_block[i + 1] = v - math.floor(d / 2)

# Reshape the original block back to 4x4
original_block = original_block.reshape((4, 4))

# Print the results
print("d_dash values:", [abs(stego_img[i + 1] - stego_img[i]) for i in range(0, stego_img.size, 2)])
print("v values:", [math.floor((stego_img[i] + stego_img[i + 1]) / 2) for i in range(0, stego_img.size, 2)])
print("d values:", [int(abs(stego_img[i + 1] - stego_img[i]) / 2)  for i in range(0, stego_img.size, 2)])
print("Extracted secret bits:", extracted_bits)
print("Extracted original block values:\n", original_block)

average_value = math.floor(np.mean(original_block))
print("\nAverage Value:", average_value)

binary_block = np.where(original_block> average_value, 0, 1)
print("\nBinary Block:")
print(binary_block)

lbp = []

# Iterate through each row in the binary block
for row in binary_block:
    # Iterate through the row pairwise
    for i in range(0, len(row), 2):
        # Perform XOR operation on each pair
        xor_result = row[i] ^ row[i + 1]
        # Append the result to the list
        lbp.append(xor_result)

print("\nLBP:", lbp)

shuffled_result = extracted_bits.copy()

# Swap elements pairwise
for i in range(0, len(shuffled_result), 2):
    if i + 1 < len(shuffled_result):
        shuffled_result[i], shuffled_result[i + 1] = shuffled_result[i + 1], shuffled_result[i]

print("\nShuffled Result of extracted bits:", shuffled_result)

xor_with_suffled = [lbp[i] ^ shuffled_result[i] for i in range(len(lbp))]
print("\nfinal Secret Code:", xor_with_suffled)