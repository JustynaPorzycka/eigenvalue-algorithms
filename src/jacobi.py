import numpy as np
import helper_functions as hf

# --------------------------------- Jacobi ---------------------------------

def jacobi_iteration_max(A, tol=1e-8, max_iter = 10000):
    n = len(A)
    A_i = A.copy()
    V = np.identity(n)
    
    for _ in range(max_iter):
        max_off_diag = 0
        k, l = 0, 0
        for p in range(n):
            for q in range(p+1, n):
                a_pq = A_i[p, q]
                # Update the maximum off-diagonal element if the current element is larger
                if abs(a_pq) > abs(max_off_diag):
                    max_off_diag = a_pq
                    k, l = p, q
        # Check if the maximum off-diagonal element is smaller than the tolerance
        if abs(max_off_diag) < tol:
            break
        
        R = hf.calculate_rotation_matrix(A_i, k, l)
        # Update the matrix A_i and V
        # R.T * A * R
        A_i = hf.multiply_from_right(A_i, R, k, l)
        A_i = hf.multiply_from_left(A_i, R.T, k, l)
        # V * R
        V = hf.multiply_from_right(V, R, k, l)
    # Extract the eigenvalues from the diagonal of the matrix A_i.
    # The eigenvectors are stored in the matrix V
    eigenvalues = np.diag(A_i)
    sort_indices = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = V[:, sort_indices]
                
    return eigenvalues, eigenvectors

def jacobi_iteration_cyclic(A, max_iter = 10000):
    n = len(A)
    A_i = A.copy()
    V = np.identity(n)
    for _ in range(max_iter):
        if hf.is_almost_diagonal(A_i):
            break
        k, l = 0, 0
        ik = 0
        jk = 1
        while ik < n - 1 or jk < n:
            k, l = ik, jk
            if jk < n - 1:
                jk += 1
            else:
                ik += 1
                jk = ik + 1
            if A_i[k,l] == 0:
                continue
            else:
                R = hf.calculate_rotation_matrix(A_i, k, l)
                # Update the matrix A_i and V
                # R.T * A * R
                A_i = hf.multiply_from_right(A_i, R, k, l)
                A_i = hf.multiply_from_left(A_i, R.T, k, l)
                # V * R
                V = hf.multiply_from_right(V, R, k, l)
    # Extract the eigenvalues from the diagonal of the matrix A_i.
    # The eigenvectors are stored in the matrix V
    eigenvalues = np.diag(A_i)
    sort_indices = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = V[:, sort_indices]
                
    return eigenvalues, eigenvectors

# --------------------------------------------------------------------------