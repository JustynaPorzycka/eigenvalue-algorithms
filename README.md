# Algorithms for Calculating Eigenvalues and Eigenvectors of Matrices

## Project Overview
The main functionality of this project is to compute the eigenvalues and eigenvectors of matrices (read from a file). Several algorithms are utilized in the program, depending on the type of matrix provided:

### 1. Non-Symmetric Matrix with Real Eigenvalues – Algorithms Used:
- **QR Decomposition Algorithm**
- **Characteristic Polynomial Method** using the Newton-Raphson method for root approximation (along with Banach's contraction theorem). In both cases, eigenvectors are computed by solving the characteristic equation `Ax = λx` for each eigenvalue `λ`.
  - **Note**: For non-symmetric matrices with complex eigenvalues, only the eigenvalues are calculated (using the Newton method). An appropriate message is displayed.

### 2. Symmetric Matrix – Algorithms Used:
- **QR Decomposition Algorithm** with dynamic determination of eigenvectors.
- **Jacobi Rotation Method** with selection of the largest off-diagonal value.
- **Jacobi Rotation Method** with cyclic selection.



