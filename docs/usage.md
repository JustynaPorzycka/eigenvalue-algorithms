# Usage Guide

## Running the Program

To run the program, use the terminal and provide the path to the `main.py` file along with the path to the matrix input file.

```bash
python main.py <path_to_matrix_file>
```

### Input

The input file should have the following structure:
- The first line should contain the dimension `N` of the matrix.
- Each subsequent line should contain a row of the matrix, with values separated by a single space.

The program only reads up to the last row of the matrix, so comments can follow after the matrix data.

### Output

The output depends on whether the input matrix is symmetric or not. It will provide a brief summary of the methods used, followed by the eigenvalues and eigenvectors of the matrix, the results of a validation test for the computed values, and the time taken for the computations.


## References

1. “On the convergence of the classical Jacobi method for real symmetric matrices with non-distinct eigenvalues”, H. P. M. van Kempen
2. “The cyclic Jacobi method for computing the principal values of a complex matrix”, G. E. Forsythe and P. Henrici
