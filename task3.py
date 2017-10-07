import numpy as np
import scipy as sci
import scipy.ndimage as ndi
import math, time
from scipy.ndimage import *
from scipy.misc.pilutil import imshow

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
    return np.transpose(derivative2Hessian_(M), (2,3,1,0))

def histogram(I):
    Ones = np.ones_like(I)
    result = ndi.sum(Ones, I, index = range(256))
    return result.astype(int)

def Normalize(I):
    '''
    Due to the difference on value scale,
    We need Normalization on Image scale [0..255]
    '''
    low = I.min()
    high = I.max()
    I = (I-low) * 1.0 / (high - low)
    return I * 255


I = imread('Image1.jpg')


# Task 3.1
dI = derivative(I)
d2I = derivative(dI)

H = derivative2Hessian(d2I)

eigvals = np.linalg.eigvals(H)
min_eigvals = eigvals.min(axis = 2)
hist = histogram(np.floor(Normalize(min_eigvals)))
hist_sum = np.cumsum(hist)
hist_total = hist.sum()

Thresholds = (hist_sum >= (0.98 * hist_total)).astype(int)
T = np.min(np.argwhere(Thresholds > 0))


Cornor = (Normalize(min_eigvals) >= T).astype(int) 

print Cornor
imshow(Cornor)

# Task 3.2

