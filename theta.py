import numpy as np
import itertools as it
import random as r
import math as m
from joblib import Parallel, delayed
from npy_append_array import NpyAppendArray
import matplotlib.pyplot as plt

# Creates tuples of integer points in the box [-R,R]^2
# We cut off at R = 15 since the function is small outside of this region

d, R = 2, 15
pts = np.array([i for i in it.product(*[[i for i in range(-R, R)] for j in range(d)])])


def _mat(a, b, c, d):
    return np.array([a, b, c, d]).reshape(2, 2)


# Defining helper functions to sample points uniformly at random

_S_mats = np.array([_mat(0, 0, 0, 0), _mat(1, 0, 0, 0), _mat(-1, 0, 0, 0),
                   _mat(0, 0, 0, 1), _mat(0, 0, 0, -1), _mat(1, 0, 0, 1),
                   _mat(-1, 0, 0, -1), _mat(1, 0, 0, -1), _mat(-1, 0, 0, 1),
                   _mat(0, 1, 1, 0), _mat(0, -1, -1, 0), _mat(1, 1, 1, 0),
                   _mat(-1, -1, -1, 0), _mat(0, 1, 1, 1), _mat(0, -1, -1, -1)])

_h_list = [[[0.5, 0.5], [0.0, 0.0]], [[0.0, 0.5], [0.5, 0.0]],
           [[0.0, 0.5], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]],
           [[0.0, 0.0], [0.5, 0.5]], [[0.0, 0.0], [0.0, 0.5]],
           [[0.5, 0.0], [0.0, 0.0]], [[0.5, 0.5], [0.5, 0.5]],
           [[0.5, 0.0], [0.0, 0.5]], [[0.0, 0.0], [0.5, 0.0]]]


def _random_X():
    return _mat(r.random()-1/2, *[r.random()-1/2]*2, r.random()-1/2)


def _random_Y():
    x = 9/16*1/(r.random())**(1/3)
    u = r.random()/2
    y = (1-u**2)**(1/2)/r.random()
    return x*_mat(1, u, 0, 1)@_mat(y, 0, 0, 1/y)@_mat(1, 0, u, 1)


def _random_Z():
    return _random_X()+_random_Y()*1j


def _random_H():
    x = [r.random()-1/2, r.random()-1/2]
    y = [r.random()-1/2, r.random()-1/2]
    return np.array([x, y])


def _F(h, X):
    return np.einsum('ij,ij->i', pts+h, np.einsum('ij,...j', X, pts+h))


def _ab_det(Z, i):
    return np.abs(np.linalg.det(Z+_S_mats[i]))


# _test if the matrix X + i Y lies in the fundamental domain
def _test(Z):
    return all([np.abs(Z[i, i]) > 1 for i in range(2)]
               + [np.abs(Z[0, 0]+Z[1, 1]-2*Z[0, 1] + i) > 1 for i in [-1, 1]]
               + [_ab_det(Z, i) > 1 for i in range(15)])


# Defining the absolute value of the theta function
def theta_abs(h, X, Y):
    phase = m.pi*_F(h[0], X)*1j+2*m.pi*pts@h[1]*1j - m.pi*_F(h[0], Y)
    return np.linalg.det(Y)**(1/4)*np.abs(np.sum(np.exp(phase)))


# We create a potential sample of N points, and then will keep the ones that pass _test(Z)
def generate_theta_0(N: int, file_name: str, mode='a'):
    """
    First, sample N complex matrices from the Siegel upper-half plane, and keep those which lie in the fundamental domain for Theta.
    Use these to generate a uniform sample of the fundamental domain, compute the Siegel theta function at each of these points, and append the results to the file "file_name".
    Argument 'mode' can be set to 'w' to overwrite any existing data in the file.
    """
    modes = ['a', 'w']
    if mode not in modes:
        raise ValueError("Invalid mode. Expected one of 'a' or 'w'")
    potential_sample = (_random_Z() for i in range(N))
    sample = (Z for Z in potential_sample if _test(Z))
    with NpyAppendArray(file_name, delete_if_exists=(mode == 'w')) as file:
        file.append(np.array(Parallel(n_jobs=-1)(delayed(theta_abs)(x, np.real(y), np.imag(y)) for x in _h_list for y in sample)))


def generate_theta(N, file_name, mode='a'):
    """
    First, sample N complex matrices from the Siegel upper-half plane, and keep those which lie in the fundamental domain for Theta.
    Use these to generate a uniform sample of the orbit of x=0, y=0 in the fundamental domain, compute the Siegel theta function at each of these points, and append the results to the file "file_name".
    Argument 'mode' can be set to 'w' to overwrite any existing data in the file.
    """
    modes = ['a', 'w']
    if mode not in modes:
        raise ValueError("Invalid mode. Expected one of 'a' or 'w'")
    potential_sample = (_random_Z() for i in range(N))
    sample = (Z for Z in potential_sample if _test(Z))
    with NpyAppendArray(file_name, delete_if_exists=(mode == 'w')) as file:
        file.append(np.array(Parallel(n_jobs=-1)(delayed(theta_abs)(_random_H(), np.real(y), np.imag(y)) for i in range(100) for y in sample)))


def histogram(file_name, bins=1000, axes=None, cumulative=False):
    """
    Produces a histogram from data stored in "file_name"
    """
    data = np.load(file_name)
    plt.figure(figsize=(16, 9), dpi=80)
    plt.hist(data, density=True, bins=bins, cumulative=cumulative)
    if axes is not None:
        plt.axis(axes)
    plt.xlabel("s")
    plt.ylabel("Density")
    plt.title(f"Probability Density for Absolute Value of Theta Function with Gaussian Cutoff - N = {len(data)}")
    plt.show()
