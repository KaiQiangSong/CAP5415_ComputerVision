import numpy as np
import numpy.linalg as la
import scipy as sci
import scipy.ndimage as ndi
import math, time
from scipy.ndimage import *
from scipy.misc.pilutil import imshow

from task1 import Normalize, dualG, Conv2D_1D
from task2 import histogram

def derivative_x(I):
    shp = I.shape
    I_padding = np.concatenate([I, np.zeros((1, shp[1]), dtype = np.float32)], axis = 0)
    I_x = I_padding[1:] - I_padding[:-1]
    return I_x

def derivative_y(I):
    return np.transpose(derivative_x(np.transpose(I)))

def derivative(I):
    if type(I) == list:
        return [derivative(item) for item in I]
    return [derivative_x(I), derivative_y(I)]

def derivative2Hessian_(M):
    if type(M[0]) != list:
        return np.stack(M)
    MM = [derivative2Hessian_(item) for item in M]
    return np.stack(MM)

def derivative2Hessian(M):
    return np.transpose(derivative2Hessian_(M), (2, 3, 1, 0))

def Gaussian_x(I ,filter):
    return np.transpose(Conv2D_1D(np.transpose(I), filter))

def Gaussian_y(I, filter):
    return Conv2D_1D(I, filter)

def Gaussian(I, filter, n = 1):
    G_x = Gaussian_x(I,filter)
    G_y = Gaussian_y(I,filter)
    return np.transpose(np.stack([np.asarray(G_x), np.asarray(G_y)]), (1,2,0))

def GaussianHessian(I, filter = dualG(), n = 1):
    gI = Gaussian(I, filter)
    gI_ = np.reshape(gI, gI.shape[:-1] + (1, gI.shape[-1]))
    gI_ = np.concatenate([gI_, np.zeros_like(gI_)], axis = gI_.ndim - 2)
    gnI = gI
    for i in range(1,n):
        gnI = np.reshape(gnI, gnI.shape + (1,))
        gnI = np.concatenate([gnI, np.zeros_like(gnI)], axis = gnI.ndim - 1)
        
        ndim = gnI.ndim
        gnI = np.matmul(gnI, gI_)
        gnI = np.reshape(gnI, gnI.shape[:-2] + (gnI.shape[-2] * gnI.shape[-1], ))
        
    return gnI

def Cornor(M):
    hist = histogram(np.floor(Normalize(M)))
    hist_sum = np.cumsum(hist)
    hist_total = hist.sum()

    Thresholds = (hist_sum >= (0.98 * hist_total)).astype(int)
    T = np.min(np.argwhere(Thresholds > 0))


    Cornor = (Normalize(M) >= T).astype(int)
    return Cornor

if __name__ == '__main__':
    I = imread('input3.png')
    if I.ndim == 3:
        I = I[:,:,0]
    imshow(I)


    # Task 3.1
    dI = derivative(I)
    d2I = derivative(dI)

    H = derivative2Hessian(d2I)

    eigvals = la.eigvals(H)
    min_eigvals = eigvals.min(axis = 2)
    
    
    C1 =  Cornor(min_eigvals)
    imshow(C1)

    # Task 3.2
    filter = dualG(sigma = 1)
    g2I = np.reshape(GaussianHessian(I, filter, 2), I.shape + (2, 2))
    
    st = time.time()
    score_1 = la.det(g2I) - 0.04 * np.trace(g2I, axis1 = g2I.ndim-2, axis2 = g2I.ndim-1)
    C2 = Cornor(score_1)
    print time.time() - st
    imshow(C2)
    
    # Task 3.3
    st = time.time()
    eigvals_ = la.eigvals(g2I)
    score_2 = np.prod(eigvals_, axis = 2) - 0.04 * np.sum(eigvals_, axis = 2)
    C3 = Cornor(score_2)
    print time.time() - st
    imshow(C3)