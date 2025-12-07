
import numpy as np

class DegeneracyAnalyzer:
    def __init__(self, threshold=10.0):
        self.threshold = threshold

    def analyze(self, H_matrix):
        """
        Analyzes the Hessian/Information matrix for degeneracy.
        Returns:
            is_degenerate (bool): True if condition number > threshold
            condition_number (float): The ratio of max/min eigenvalues
            eigenvalues (np.array): Sorted eigenvalues
        """
        # 1. Calculate Eigenvalues
        # eigh is used for symmetric matrices
        eigenvalues, _ = np.linalg.eigh(H_matrix)

        # 2. Sort Eigenvalues (descending)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]

        # Avoid division by zero
        if sorted_eigenvalues[-1] == 0:
            condition_number = float('inf')
        else:
            condition_number = sorted_eigenvalues[0] / sorted_eigenvalues[-1]

        is_degenerate = condition_number > self.threshold

        return is_degenerate, condition_number, sorted_eigenvalues
