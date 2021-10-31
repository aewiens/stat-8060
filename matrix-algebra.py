#!/usr/bin/env python3
import numpy as np


def gauss_elim(A, pivot=True):
    """ Perform gaussian elimination on an augmented matrix A
    
    Parameters
    ----------
    A : np.array
        Augmented matrix
    pivot : bool
        Perform gaussian elimination with pivoting
        Default: true
        If false, naive gaussian elimination is performed

    Return
    ------
    U : np.array
        Upper triangular matrix
    """
    n, m = A.shape
    U = A.copy()

    for i in range(0, n):

        if pivot:

            # Find max in this column
            col = np.fabs(U[i:,i])
            max_row = i + np.argmax(col)

            # Swap if necessary
            if max_row != i:
                swap = True
                UU = U.copy()

                U[max_row,i:m] = UU[i, i:]
                U[i,i:m] = UU[max_row, i:]

        # Forward elimination
        for k in range(i+1, n):
            c = U[k,i] / U[i,i]
            for j in range(i, m):
                U[k,j] -= c * U[i,j]

    return U


def calc_determinant(A, pivot=True):
    """ Find determinant of a square matrix using gaussian elimination

    Parameters
    ----------
    A : np.array
        square matrix (2d) for which determinant is desired
    
    pivot : bool
        Use pivoting in gaussian elimination algorithm?

    Return
    ------
    det : float
        Determinant of square matrix A
    """
    n, m = A.shape
    assert m == n, "The determinant is only defined for square matrices!"
    U = A.copy()
   
    swap = 0
    for i in range(0, n):

        if pivot:

            # Find max in this column
            max_entry = np.fabs(U[i,i])
            max_row = i
            for k in range(i+1, n):
                if np.fabs(U[k,i]) > max_entry:
                    max_entry = np.fabs(U[k,i])
                    max_row = k

            if max_row != i:
                swap += 1

            # Swap
            for k in range(i, m):
                tmp = U[max_row,k]
                U[max_row,k] = U[i,k]
                U[i,k] = tmp


        # Forward elimination
        for k in range(i+1, n):
            c = U[k,i] / U[i,i]
            for j in range(i, m):
                U[k,j] -= c * U[i,j]

    # naive gaussian elimination routine
    for i in range(n):
        for k in range(i+1, n):
            A[k] -= A[i] * A[k, i] / A[i,i]

    det = np.prod(np.diag(A))
    return det * (-1)**swap



def back_solve(U):
    """ Solve system of equations contained in upper triangular matrix U

    Parameters
    ----------
    U : np.ndarray
        Augmented matrix U

    Return
    ------
    x : np.ndarray
       numpy 1D array containing the solution for the system 
    """
    n = len(U)
    x = np.zeros(n) 
    for i in range(n-1, -1, -1):
        x[i] = U[i,n] / U[i,i]
        for k in range(i-1, -1, -1):
            U[k,n] -= U[k,i] * x[i]
    return x



def LU_decomposition(A, pivot=False):
    """ Perform LU decomposition using Crout's algorithm

    Parameters
    ----------
    A : ndarray
        Square matrix

    Return
    ------
    L, U : ndarray
        Square matrices s.t. LU = A
    """
    n, m = A.shape
    assert m==n, "LU decomposition is only meant for square matrices!"

    U = np.zeros_like(A)
    L = np.identity(n)

    if pivot:
        for j in range(n):
            for i in range(j+1):
                s1 = sum(U[k,j] * L[i,k] for k in range(i))
                U[i,j] = A[i,j] - s1

            for i in range(j, n):
                s2 =  sum(U[k,j] * L[i,k] for k in range(j))
                L[i,j] = (A[i,j] - s2) # / U[j,j]


            # Find max for partial pivot
            max_row = j
            max_entry = np.fabs(U[j,j])
            for i in range(j+1, n):
                if np.fabs(L[i,j]) > max_entry:
                    max_entry = L[i,j]
                    max_row = i

            # Swap
            for i in range(j, n):
                tmp = U[max_row,j]
                U[max_row,j] = U[i,j]
                U[i,j] = tmp / U[j,j]

    else:

        for j in range(n):
            for i in range(j+1):
                s1 = sum(U[k,j] * L[i,k] for k in range(i))
                U[i,j] = A[i,j] - s1

            for i in range(j, n):
                s2 =  sum(U[k,j] * L[i,k] for k in range(j))
                L[i,j] = (A[i,j] - s2) / U[j,j]

    return L, U



def cholesky(A):
    """ Perform a Cholesky decomposition of matrix A

    Paramters
    ---------
    A : numpy array
        Positive definite square matrix for which decomposition is requried

    Return
    ------
    L : numpy array
        Lower-triangular matrix
        
    """
    m, n = A.shape
    assert m==n, "Square matrix required for Cholesky decomposition!"
    # assert positive definite
    L = np.zeros_like(A)

    # Cholesky decomposition algorithm
    for i in range(n):
        for k in range(i+1):
            s = sum(L[i,j] * L[k,j] for j in range(k))

            if i == k: 
                L[i,k] = np.sqrt(A[i,i] - s)
            else:
                L[i,k] = (1.0 / L[k,k] * (A[i,k] - s))
    return L




""" Gaussian elimination example """
A = np.array([[20, 15, 10, 45],
              [-3, -2.249, 7, 1.751],
              [5, 1, 3, 9]])
B = gauss_elim(A, pivot=True)
print(B)
x = back_solve(B)
line = "Gauss Elim. Solution: [" + "{:10.5f}" * len(x) + "]"
print(line.format(*x))


""" Determinant  Example """
a = np.array([[25, 5, 1],
              [64, 8, 1],
              [144, 12, 1]], float)

det_a = calc_determinant(a)
#print("Determinant of matrix a: {:7.4f}".format(det_a))
print(pivot_matrix(a))


""" LU Decomposition example """
C = np.array([[11,9,24,2],
              [1,5,2,6],
              [3,17,18,1],
              [2,5,7,1]], float)

L, U = LU_decomposition(C, pivot=False)


""" Cholesky example """
D = np.array([[6, 3, 4, 8], 
              [3, 6, 5, 1], 
              [4, 5, 10, 7], 
              [8, 1, 7, 25]], float)
E = cholesky(D)
