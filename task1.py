import numpy as np
import scipy as sci
import scipy.ndimage as ndi
import math
from scipy.ndimage import *
from scipy.misc.pilutil import imshow
from copy import deepcopy


def Gaussian(x, sigma = 1):
    y = 1/(math.sqrt(2*math.pi) * sigma) * math.exp(- x*x / (2* sigma * sigma))
    return y

def dGaussian(x, sigma = 1):
    y = - x/(math.sqrt(2*math.pi) * sigma * sigma * sigma) * math.exp(- x*x / (2* sigma * sigma))
    return y

def G(size = 3, sigma = 1):
    x = range(size)
    g = [Gaussian(i, sigma) for i in x]
    return g

def dG(size = 3, sigma = 1):
    x = range(size)
    g = [dGaussian(i, sigma) for i in x]
    return g

def dualG(size = 5, sigma = 1):
    g = G(size/2+1, sigma)
    g = g[::-1]+g[1:]
    return g

def dualdG(size = 5, sigma = 1):
    g = dG(size/2+1, sigma)
    g_ = [-x for x in g[::-1]]
    g = g_+g[1:]
    return g

def Conv2D_1D(I, filter, padding = True):
    size = len(filter)
    shp = I.shape
    if padding:
        I = np.concatenate([np.zeros((shp[0], size/2), dtype = np.float32),
                            I,
                            np.zeros((shp[0], size/2), dtype = np.float32)], axis = 1)
    
    highI = np.empty((shp[0], shp[1], 0), dtype = np.float32)
    for i in range(size):
        subI = I[:,i:i+shp[1]]
        subI = np.expand_dims(subI, axis = 2)
        highI = np.append(highI, subI, axis = 2)
        
    newI = np.dot(highI, np.transpose(np.asmatrix([filter], dtype = np.float32)))
    return newI

def doFilter(I):
    filter_G = dualG(sigma = 1)
    filter_dG = dualdG(sigma = 1)
    
    I_x = np.transpose(Conv2D_1D(np.transpose(I), filter_G))
    I_y = Conv2D_1D(I, filter_G)
    
    I_x_2 = np.transpose(Conv2D_1D(np.transpose(I_x), filter_dG))
    I_y_2 = Conv2D_1D(I_y, filter_dG)
    return I_x, I_y, I_x_2, I_y_2
    

def Magnitude(I_x_2, I_y_2):
    M_x = np.multiply(I_x_2, I_x_2)
    M_y = np.multiply(I_y_2, I_y_2)
    M = np.sqrt(M_x + M_y)
    return M

dx = np.asarray([-1, -1, 0, 1, 1, 1, 0, -1], dtype = np.int32)
dy = np.asarray([0, -1, -1, -1, 0, 1, 1, 1], dtype = np.int32)

def hysteresis_threshold(I, low, high):
    low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
    mask_low = I > low
    mask_high = I > high
    # Connected components of mask_low
    labels_low, num_labels = ndi.label(mask_low)
    # Check which connected components contain pixels from mask_high
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]
    return thresholded

def Normalize(I):
    low = I.min()
    high = I.max()
    I = (I-low) * 1.0 / (high - low)
    return I * 255


def Candy(I):

    I_x, I_y, I_x_2, I_y_2 = doFilter(I)
    M = Magnitude(I_x_2, I_y_2)
    theta = np.arctan2(I_y_2, I_x_2) / math.pi * 4 + 4
    switcher = np.rint(theta).astype(int)
    switcher = switcher - (switcher == 8).astype(int) * 8
    
    print switcher
    
    f_x = dx[switcher]
    f_y = dy[switcher]
    
    M_padding = np.zeros((M.shape[0]+2, M.shape[1]+2), dtype = np.float32)
    M_padding[1:-1,1:-1] = M
    
    ID_x = np.transpose(np.tile(np.asarray(range(M_padding.shape[0]), dtype = np.int32), (M_padding.shape[1],1)))
    ID_y = np.tile(np.asarray(range(M_padding.shape[1]), dtype = np.int32), (M_padding.shape[0],1))
    
    P_x = ID_x 
    P_y = ID_y
    P_x[1:-1,1:-1] += f_x
    P_y[1:-1,1:-1] += f_y
    
    Q_x = ID_x
    Q_y = ID_y
    Q_x[1:-1,1:-1] -= f_x
    Q_x[1:-1,1:-1] -= f_y
    
    M_P = M_padding[P_x, P_y]
    M_Q = M_padding[Q_x, Q_y]
    
    MM = np.maximum(M_P, M_Q)
    MM = np.maximum(MM, M_padding)
    Exist = (M_padding == MM).astype(int)
    print Exist[1:-1,1:-1], Exist[1:-1,1:-1].sum()
    M_new = np.multiply(M_padding, Exist)[1:-1, 1:-1]
    
    M_threshold = hysteresis_threshold(Normalize(M_new), 25, 64).astype(int)
    print M_threshold, M_threshold.sum()
    return [I_x, I_y, I_x_2, I_y_2], [I, M, M_new, M_threshold]

I = imread('Image3.jpg')
print type(I)
n, m = I.shape
print n, m, I

IG, Candies = Candy(I)
outputImage = np.concatenate([np.concatenate([Normalize(img) for img in IG], axis = 1),
                              np.concatenate([Normalize(img) for img in Candies], axis = 1)],
                             axis = 0)
imshow(outputImage)

