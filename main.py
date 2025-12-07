# Adaptive Kalman Filter-based SLAM in LiDAR Degenerated Environments

import numpy as np
import matplotlib.pyplot as plt
import cv2

def analyze_degeneracy(H_matrix, threshold=10.0):
    print("Analyzing environment...")

    # 1. Calculate Eigenvalues (w) and Eigenvectors (v)
    # eigh is used for symmetric matrices (like Hessians)
    eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)

    # 2. Sort Eigenvalues and Eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # 3. Calculate Condition Number
    condition_number = sorted_eigenvalues[0] / sorted_eigenvalues[-1]
    print(f"Condition Number: {condition_number}")

    # 4. Calculate Eigenvalue Ratios
    ratios = sorted_eigenvalues[1:] / sorted_eigenvalues[:-1]
    print("Eigenvalue Ratios:", ratios)

    # 5. Visualize Eigenvalues
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.stem(sorted_eigenvalues)
    plt.title("Sorted Eigenvalues")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")

    plt.subplot(1, 2, 2)
    plt.stem(ratios)
    plt.title("Eigenvalue Ratios")
    plt.xlabel("Index")
    plt.ylabel("Ratio")
    plt.show()

    # 6. Check for Degeneracy
    if condition_number > threshold:
        print("Degenerate environment detected!")
    else:
        print("Non-degenerate environment.")

    return condition_number

if __name__ == "__main__":
    # Example usage
    H = np.array([[4, 2], [2, 3]])
    analyze_degeneracy(H)