import numpy as np
import numpy.linalg as la
import scipy as sci
import scipy.ndimage as ndi
import math, time
from scipy.ndimage import *
from scipy.misc.pilutil import imshow

from task1 import Normalize, dualG, Conv2D_1D
from task2 import histogram


def duplicate2D_2D(I, mask, padding = True):
    '''
    Prepare a 3D-Tensor on Image I for Conv or other operation with a 2D Mask
    '''
    
    if padding:
        pad_x = mask.shape[0]/2
        pad_y = mask.shape[1]/2
        I_padding = np.zeros((I.shape[0] + pad_x * 2, I.shape[1] + pad_y * 2), dtype = np.float32)
        I_padding[pad_x:I.shape[0]+pad_x, pad_y:I.shape[1]+pad_y] = I
    else:
        I_padding = I
    
    
    highI = np.empty(I.shape+(0,), dtype = np.float32)
    
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j] > 0:
                highI = np.append(highI, np.reshape(I_padding[i:i+I.shape[0], j:j+I.shape[1]], I.shape+(1,)), axis = 2)
                
    return highI

def dFilter():
    '''
    derivative Filter
    '''
    return np.asarray([-1, 0, 1], dtype = np.float32)

def derivative_x(I):
    '''
    apply derivative Filter on x axis
    '''
    return np.asarray(Conv2D_1D(I, dFilter()))

def derivative_y(I):
    '''
    apply derivative Filter on y axis
    '''
    return np.asarray(np.transpose(Conv2D_1D(np.transpose(I), dFilter())))

def derivative(I):
    '''
    get next order derivatives 
    '''
    if type(I) == list:
        return [derivative(item) for item in I]
    return [derivative_x(I), derivative_y(I)]

def derivative2Hessian_(M):
    '''
    build a Hessian Matrix 
    '''
    if type(M[0]) != list:
        return np.stack(M)
    MM = [derivative2Hessian_(item) for item in M]
    return np.stack(MM)

def derivative2Hessian(M):
    '''
    build a Hessian Matrix and transpose to a readable format
    '''
    return np.transpose(derivative2Hessian_(M), (2, 3, 1, 0))

def Gaussian2D(x, y, sigma = 1):
    '''
    2D Gaussian Function
    '''
    return 1.0 / (2.0 * math.pi * sigma * sigma) * np.exp(-1.0 / (2.0 * sigma * sigma) * (x * x + y * y))

def GuassianFilter2D(size = 5, sigma = 1):
    '''
    Build a 2D Gaussian Filter use 2D Gaussian Function
    '''
    dx = np.tile(np.asarray(range(-(size/2), size/2+1), dtype = np.float32), (size,1))
    dy = np.transpose(dx)
    return Gaussian2D(dx, dy, sigma)

def Conv2D_2D(I, filter):
    '''
    Conv on  2D Image with a 2D kernel
    '''
    mask = np.ones_like(filter, dtype = np.int)
    highI = duplicate2D_2D(I, mask)
    ConvI = np.dot(highI, np.reshape(filter, (filter.shape[0] * filter.shape[1], 1)))
    return np.reshape(ConvI, I.shape)

def GaussianHessian(I, filter):
    '''
    Build a Hessian which apply Gaussian Smothing
    '''
    I_x, I_y = derivative(I)
    A = Conv2D_2D(I_x * I_x, filter)
    B = Conv2D_2D(I_y * I_y, filter)
    C = Conv2D_2D(I_x * I_y, filter)
    
    H = np.stack([np.stack([A, C]), np.stack([C, B])])
    H = np.transpose(H, (2, 3, 0, 1))
    return H

def Corner_(M, ratio = 0.998):
    '''
    A Threshold Method to dectect the Corner
    '''
    hist = histogram(np.round(Normalize(M)))
    hist_sum = np.cumsum(hist)
    hist_total = hist.sum()

    Thresholds = (hist_sum >= (ratio * hist_total)).astype(int)
    T = np.min(np.argwhere(Thresholds > 0))
    print 'Thresholds =', T

    Corner = (Normalize(M) >= T).astype(int)
    return Corner

if __name__ == '__main__':
    I = imread('input3.png')
    if I.ndim == 3:
        I = I[:,:,0]
    imshow(I)


    # Task 3.1
    st = time.time()
    dI = derivative(I)
    d2I = derivative(dI)

    H = derivative2Hessian(d2I)

    eigvals = la.eigvals(H)
    min_eigvals = eigvals.min(axis = 2)
    
    
    C1 =  Corner_(min_eigvals)
    print time.time() - st
    imshow(C1)

    # Task 3.2
    '''
    # It seems that there's no much differences when I fine-tune my alpha value
    '''
    filter = GuassianFilter2D()
    g2I = GaussianHessian(I, filter)
    
    st = time.time()
    score_1 = la.det(g2I) - 0.04 * np.trace(g2I, axis1 = g2I.ndim-2, axis2 = g2I.ndim-1)
    C2 = Corner_(score_1)
    print time.time() - st
    imshow(C2)
    
    # Task 3.3
    '''
    Basicly it's theoretically the same.
    But the 1st (Task 3.2) method runs much quicker than the 2nd (Task 3.3) method.
    It takes 0.106s for 1st method, and 0.532s for 2nd method 
    '''
    st = time.time()
    eigvals_ = la.eigvals(g2I)
    score_2 = np.prod(eigvals_, axis = 2) - 0.04 * np.sum(eigvals_, axis = 2)
    C3 = Corner_(score_2)
    print time.time() - st
    imshow(C3)