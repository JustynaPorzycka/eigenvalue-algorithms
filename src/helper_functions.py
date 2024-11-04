import numpy as np
import time
import qr
import jacobi
import NM
import warnings

# ---------------------------- Helper functions ----------------------------

def allclose(A, eigenvalues, eigenvectors):
    # Check if A*eigenvecotr == eigenvalue*eigenvector for all eigenvectors
    list = []
    bool = False
    n = A.shape[0]
    for i in range(len(A)):
        eigenvector = eigenvectors[:, i]
        if np.count_nonzero(eigenvector) == 0:
            return False
        else:
            product = np.dot(A, eigenvector)
            expected_product = eigenvalues[i] * eigenvector
            bool = np.allclose(product, expected_product)
            list.append(bool)
    return all(list)

def read_matrix_file(path_file):
    with open(path_file,'r') as f:
        size = int(f.readline().strip())
        matrix = []
        for i in range(size):
            line = f.readline().strip()
            row = [float(x) for x in line.split()]
            matrix.append(row)
        return np.array(matrix)

def print_matrix_info(A):
    # Check if matrix is symmetric
    is_symmetric = np.allclose(A, A.T)
    
    if is_symmetric:
        print("\n*******************************************************\n\
               \nThe matrix is symmetric, so three efficient methods for\
               \ncomputing eigenvalues and eigenvectors are presented.\n")
        
        print("Jacobi rotations method with maximum off-diagonal element:\n")
        print_result(A, jacobi.jacobi_iteration_max)
        
        print("Jacobi rotations method with cyclic iteration:\n")
        print_result(A, jacobi.jacobi_iteration_cyclic)
        
        print("QR method:\n")
        print_result(A, qr.qr_algorithm)
        print("*******************************************************")
    else:
        print_result_not_sym(A)

        print("********************************************************")

def print_result(A, f):
    start = time.perf_counter_ns()
    eigenvalues, eigenvectors = f(A)
    stop = time.perf_counter_ns()
    print(f"Eigenvalues:\n {eigenvalues}")
    print(f"Eigenvectors:\n {eigenvectors}")
    print("A*eigenvector == eigenvalue*eigenvector for all eigenvectors:", allclose(A, eigenvalues, eigenvectors))
    print("Time in ns:", stop-start)
    print("\n")
    
def print_result_not_sym(A):
    start = time.perf_counter_ns()
    eigenvalues, eigenvectors = NM.startNewtonMethod(A)
    stop = time.perf_counter_ns()
    if eigenvectors == []:
        print("\n*******************************************************\n\
        \nMatrix is not symmetric and has complex eigenvalues, eigenvectors won't be calculated")
        print(f"Eigenvalues:\n {eigenvalues}")
    else:
        print("\n*******************************************************\n\
        \nThe matrix is not symmetric, so two efficient methods for\
        \ncomputing eigenvalues are presented. The eigenvectors were\
        \ncalculated by solving the characteristic equation for each\
        \neigenvalue.\n")
        
        print("Newton-Raphson method for approximating polynomial roots\
              \n(in our case, the characteristic polynomial):\n")
        
        print(f"Eigenvalues:\n {eigenvalues}")
        print(f"Eigenvectors:\n {eigenvectors}")
        print("A*eigenvector == eigenvalue*eigenvector for all eigenvectors:", allclose(A, eigenvalues, eigenvectors))
        print("Time in ns:", stop-start)
        print("\n")
        print("QR method:\n")
        print_result(A, qr.qr_algorithm)

# --------------------------------------------------------------------------

def is_almost_diagonal(matrix, threshold=1e-8):
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if i != j and abs(matrix[i][j]) >= threshold:
                return False
    return True

def calculate_rotation_matrix(A_i, k, l):
    x, y = A_i[k, k], A_i[l, l]
    if A_i[k,k] == A_i[l,l]:
        c = s = np.cos(np.pi/4)
        if A_i[k,l] > 0:
            s = np.sin(np.pi/4)
        else:
            s = np.sin(-np.pi/4)
    else:
        d = (y - x) / (2 * A_i[k, l])
        t = np.sign(d) / (abs(d) + np.sqrt(d**2 + 1))
        c = 1 / np.sqrt(t**2 + 1)
        s = c * t
    R = np.identity((len(A_i)))
    R[k, k] = c
    R[l, l] = c
    R[k, l] = s
    R[l, k] = -s
    return R

def multiply_from_right(M, R, k, l):
    M_copy = M.copy()
    M[:,k] = M_copy @ R[:,k]
    M[:,l] = M_copy @ R[:,l]
    return M

def multiply_from_left(M, R, k, l):
    M_copy = M.copy()
    M[k,:] = R[k, :] @ M_copy
    M[l,:] = R[l, :] @ M_copy
    return M

warnings.filterwarnings("ignore")
def solve_eigenvalue_equation(A, eigenvalues):
    n = A.shape[0]
    eigenvectors = np.zeros((n, n))
    
    for i in range(n):
        M = A - eigenvalues[i] * np.eye(A.shape[0]) # Subtract lambda_i * I from A
        _, _, V = np.linalg.svd(M) # Find null space of M
        null_space_vector = V[-1]  # Eigenvector corresponding to the smallest singular value
        
        eigenvectors[:, i] = null_space_vector

    return eigenvectors

# --------------------------------------------------------------------------