# -- coding: utf-8 --
"""
Created on Fri Sep 27 17:21:07 2024

@author: jagan
"""
import math
import numpy as np

# Step 1: Create the 4x4 block as a NumPy array
block = np.array([
    [136,  93,  71,  46],
       [ 91,  68,  66,  57],
       [105,  66,  72,  56],
       [ 84, 115,  59,  58]
])

# Step 2: Compute the average of the 16 values
average_value = math.floor(np.mean(block))

# Step 3: Compare each value with the average and represent it as 0 or 1
binary_block = np.where(block > average_value, 0, 1)

# Step 4: Store the new values in a 4x4 block
print("Original Block:")
print(block)
print("\nAverage Value:", average_value)
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

# Step 6: XOR the LBO list with the 8-bit binary representation of 65
secret_code = [0, 0, 0, 1, 1, 0, 0, 1]
xor_with_secret = [lbp[i] ^ secret_code[i] for i in range(len(lbp))]
print("\nsecret code : =", secret_code)
print("\nXOR with Secret Code:", xor_with_secret)

# Step 7: Shuffle the final result pairwise
shuffled_result = xor_with_secret.copy()

# Swap elements pairwise
for i in range(0, len(shuffled_result), 2):
    if i + 1 < len(shuffled_result):
        shuffled_result[i], shuffled_result[i + 1] = shuffled_result[i + 1], shuffled_result[i]

print("\nShuffled Result:", shuffled_result)

# Calculate d, v, and d_dash
d_values = []
v_values = []
d_dash_values = []

for i in range(0, block.size, 2):
    x_i = block.flat[i]
    x_i1 = block.flat[i + 1]
    
    # Calculate d
    d = abs(x_i - x_i1)
    d_values.append(d)
    
    # Calculate v
    v = math.floor((x_i + x_i1) / 2)
    v_values.append(v)
    
    # Calculate d_dash
    w = shuffled_result[i // 2]
    d_dash = d * 2 + w
    d_dash_values.append(d_dash)

print("\nd values:", d_values)
print("v values:", v_values)
print("d_dash values:", d_dash_values)

stego_img = block.copy()

for i in range(0, block.size, 2):
    x_i = block.flat[i]
    x_i1 = block.flat[i + 1]
    v = v_values[i // 2]
    d_dash = d_dash_values[i // 2]
    
    if x_i < x_i1:
        stego_img.flat[i] = v - math.floor(d_dash / 2)
        stego_img.flat[i + 1] = v + math.ceil(d_dash / 2)
    else:
        stego_img.flat[i] = v + math.ceil(d_dash / 2)
        stego_img.flat[i + 1] = v - math.floor(d_dash / 2)
        
print("\nStego Image Block:")
print(stego_img)