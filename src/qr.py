import numpy as np
import helper_functions as hf

# --------------------------- QR ---------------------------

def qr_algorithm(A, tol=1e-8, max_iter = 10000):
    n = len(A)
    A_i = A.copy()
    V = np.identity(n)
    if np.allclose(A, A.T):
        for _ in range(max_iter):
            # perform QR decomposition of the matrix A_i
            Q, R = np.linalg.qr(A_i)
            # Update the matrix A_i and V
            A_i = R @ Q
            V = V @ Q
            # check if the matrix A_i is upper triangular
            if np.allclose(A_i, np.triu(A_i), atol=tol):
                break
        # Extract the eigenvalues from the diagonal of the matrix A_i.
        # The eigenvectors are stored in the matrix V
        eigenvalues = np.diag(A_i)
        sort_indices = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = V[:, sort_indices]
    
    else:
        for _ in range(max_iter):
            Q, R = np.linalg.qr(A_i)
            A_i = R @ Q
            if np.allclose(A_i, np.triu(A_i), atol=tol):
                break
        eigenvalues = np.diag(A_i)
        sort_indices = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = hf.solve_eigenvalue_equation(A, eigenvalues)
                
    return eigenvalues, eigenvectors

# ----------------------------------------------------------

