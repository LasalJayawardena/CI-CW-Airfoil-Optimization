import numpy as np


def gaussJordonElimination(matrix1, matrix2):
    # Combine Cup and bup into an augmented matrix
    augmented_matrix = np.column_stack((matrix1, matrix2))

    # Perform Gauss-Jordan elimination
    n = len(augmented_matrix)
    for i in range(n):
        # Partial pivot
        max_row = np.argmax(np.abs(augmented_matrix[i:, i])) + i
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        pivot = augmented_matrix[i, i]
        augmented_matrix[i] = augmented_matrix[i] / pivot

        for j in range(n):
            if i != j:
                factor = augmented_matrix[j, i]
                augmented_matrix[j] = augmented_matrix[j] - factor * augmented_matrix[i]

    # The last column of the augmented matrix contains the solution (aup)
    result = augmented_matrix[:, -1]
    return result


def euclidean_distance(min_chromosome, max_chromosome, array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)

    # Ensure arrays have the same length
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length")

    # Clip values to be within the specified range
    array1 = np.clip(array1, min_chromosome, max_chromosome)
    array2 = np.clip(array2, min_chromosome, max_chromosome)

    # Calculate Euclidean distance
    distance = np.linalg.norm(array1 - array2)

    return distance
