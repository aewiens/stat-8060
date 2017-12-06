#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Set seed so that every run has the same output.
np.random.seed(0)   


def f(x):
    """ test function """
    return (x**3 - 3)


def derivative(f, x0, h):
    """ Differentiate f (1 variable) using central difference
        
    Parameters
    ----------
    f : function (1-dimensional)
        Function for which derivative is desired
    x0 : float
        Point at which derivative is desire
    h : float
        Step size

    Return
    ------
    float
        Approximate derivative of f at x0
    """

    return (f(x0 + h) - f(x0 - h)) / (2 * h)


# central difference example for 1d function
print("\n-----------------------------------------------------")
print("Differentiate f(x) = x^3 -3 using central difference")
print("-----------------------------------------------------\n")
deriv = derivative(f, 3, 1e-4)
print("Approximate derivative = {:<12.7f}".format(deriv))
print("Exact derivative       = 27.")


def f2(x):
    return np.sum(x**2) + 2


def gradient(f, x0, h):
    """ Calculate the gradient of f (multivariable) using central difference

    Parameters
    ----------
    f : function (multivariable)
        Function for which gradient is desired
    x0 : ndarray
        Point at which gradient is desired
    h : float
        Step size for central difference routine

    Return
    grad: ndarray
        Approximate gradient of f at x0
    """

    n = len(x0)
    grad = np.zeros(n)

    # loop over dimensions
    for i in range(n):
       x1 = x0[i] - h
       x2 = x0[i] + h
       grad[i] = (f(x2) - f(x1)) / (2*h)

    return grad


# central difference example for multivariable function
print("\n------------------------------------------------------------")
print("Differentiate f2(x) = [Norm(x)]^2 + 2 using central difference")
print("-------------------------------------------------------------\n")
point = np.array([10, 10, 10])
grad = gradient(f2, point, 1e-4)
print("Approximate gradient =", grad)
print("Exact gradient       = [20., 20., 20.]")
