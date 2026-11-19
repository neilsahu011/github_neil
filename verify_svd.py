import numpy as np
from PIL import Image

from svd_compression import (
    load_image,
    svd_scratch,
    svd_numpy,
    rebuild,
)

# Load the same image at the same size as the scratch test.
A = load_image("image.jpg", max_size=128)

k = 20

# Calculate the leading singular components using both methods.
U1, S1, Vt1 = svd_scratch(A, k, verbose=False)
U2, S2, Vt2 = svd_numpy(A, k)

# Reconstruct the image matrices.
A_scratch = rebuild(U1, S1, Vt1, k)
A_numpy = rebuild(U2, S2, Vt2, k)

# Calculate relative reconstruction errors.
denominator = np.linalg.norm(A)

error_scratch = np.linalg.norm(A - A_scratch) / denominator
error_numpy = np.linalg.norm(A - A_numpy) / denominator

# Check the orthogonality of the computed singular vectors.
identity = np.eye(k)

orthogonality_U = np.linalg.norm(U1.T @ U1 - identity)
orthogonality_V = np.linalg.norm(Vt1 @ Vt1.T - identity)

print("SVD Verification Results")
print("------------------------")
print(f"Custom SVD relative error: {error_scratch:.6f}")
print(f"NumPy SVD relative error:  {error_numpy:.6f}")
print(f"U orthogonality deviation: {orthogonality_U:.6f}")
print(f"V orthogonality deviation: {orthogonality_V:.6f}")
print(
    "Maximum singular-value difference:",
    f"{np.max(np.abs(S1 - S2)):.6f}"
)