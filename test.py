import numpy as np

# Function to calculate triangular membership function
def triangular_membership(x, a, b, c):
    return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

# Example pairwise comparison matrix
pairwise_matrix = np.array([
    [1, 3, 5],
    [1/3, 1, 3],
    [1/5, 1/3, 1]
])

# Define parameters for triangular membership function
a, b, c = 0.2, 0.5, 0.8

# Apply triangular membership function to pairwise comparison matrix
fuzzy_weights = triangular_membership(pairwise_matrix, a, b, c)

# Normalize the fuzzy weights
normalized_fuzzy_weights = fuzzy_weights / np.sum(fuzzy_weights, axis=1)[:, np.newaxis]

# Output the normalized fuzzy weights
print("Normalized Fuzzy Weights (Triangular):")
for i in range(len(pairwise_matrix)):
    print(f"Criterion {i+1}: {normalized_fuzzy_weights[i]}")
