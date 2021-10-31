#!/usr/bin/env python3
""" STAT 8060 Final
    Avery Wiens
    December 8, 2017
"""
import numpy as np

def pivot_matrix(i, A):
    """ Build the pivoting matrix for A """
    M = A.copy()
    m = len(M)
    I = np.identity(m)

    col = np.fabs(M[:,i])
    row = np.argmax(col)
    
    if row != i: 
        II = I.copy()
        I[i], I[row] = II[row], II[i]

    return I


############################################################################
############################ PROBLEM 1 #####################################
############################################################################
# Write a function that does LU decomposition of a square matrix A


def LUdecomp(A, pivot=False): 
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
        P = pivot_matrix(0, A)
        A = P @ A

    for j in range(n):
        for i in range(j+1):
            U[i,j] =  A[i,j] 
            if j > 0:
                U[i,j] -= sum(U[k,j] * L[i,k] for k in range(i))

        for i in range(j, n):
            L[i,j] = (A[i,j] - sum(U[k,j] * L[i,k] for k in range(j))) / U[j,j]


    L -= np.identity(n)
    return P @ (L + U)




############################################################################
############################ PROBLEM 2 #####################################
############################################################################

# Write a function that takes the LU decomposition for matrix A and solves the
# system of equations AX = Y for the vector X. 


def LUsolve(B, Y):
    """ Solve system of equations AX = Y (where B is LU decomposition of A)

    Parameters
    ----------
    B : np.array
        Square matrix corresponding to the LU decomposition of A, with elements of
        the L matrix below the diagonal and elements of the U matrix above the
        diagonal

    Y : np.array
        Column vector Y for which the solution AX = Y is desired

    Return
    ------
    x : np.ndarray
       numpy 1D array containing the solution for the system 
    """

    n = len(B)
    U = np.zeros_like(B)
    L = np.identity(n)

    for i in range(n):
        for j in range(i):
            L[i,j] = B[i,j]
        
        for k in range(i, n):
            U[i,k] = B[i,k]

    LL = np.concatenate((L, Y), axis=1)
    Z = forward_substitute(LL)

    UU = np.concatenate((U, np.array([Z]).T), axis=1)
    X = back_substitute(UU)

    return X


def forward_substitute(L):
    """ Solve the system of equations contained in a lower triangular, augmented
        matrix L
    """
    m, n = L.shape
    x = np.zeros(m)
    for i in range(m):
        x[i] = L[i, n-1] / L[i,i]
    for i in range(1, m):
        x[i] -= sum(L[i,k] * x[k] for k in range(i))

    return x


def back_substitute(U):
    """ Solve the system of equations contained in an upper triangular, augmented
        matrix U
    """
    n = len(U)
    x = np.zeros(n) 
    for i in range(n-1, -1, -1):
        x[i] = U[i,n] / U[i,i]
        for k in range(i-1, -1, -1):
            U[k,n] -= U[k,i] * x[i]
    return x



if __name__ == '__main__':

    """ test cases from class notes  """ 
    A = np.array([[25, 5, 1],
                  [64, 8, 1],
                  [144, 12, 1]], float)

    Y = np.array([[106.8], 
                  [177.2], 
                  [279.2]])

    B = LUdecomp(A, pivot=True)
    print(B)
    test = LUsolve(B, Y)
    print(test)

    """
    a = np.array([[20, 15, 10],
                   [-3, -2.249, 7],
                   [5, 1, 3]])
    b = LUdecomp(a, pivot=False)
    print(b)
    """
