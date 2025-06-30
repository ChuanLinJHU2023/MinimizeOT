import numpy as np

# Example shapes
M, I, N, J = 4, 3, 5, 4

# Generate sample data
A1 = np.random.rand(M, I, N, J)
V1 = np.array([0, 2, 3])  # Length I
V2 = np.array([3, 2, 1, 0])     # Length J

# Create index grids for I and J
i_indices = np.arange(I)  # shape (I,)
j_indices = np.arange(J)  # shape (J,)

# Use broadcasting to create index matrices
# Shape (I, J)
A2 = A1[V1[:, np.newaxis], i_indices[:, np.newaxis], V2[np.newaxis, :], j_indices[np.newaxis, :]]

print("A2 shape:", A2.shape)
print(A2)
print(A1[2,1,2,1]==A2[1,1])