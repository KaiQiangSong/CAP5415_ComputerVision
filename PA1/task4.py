import numpy as np
import numpy.ma as ma
import numpy.linalg as la
import scipy as sci
import scipy.ndimage as ndi
import math, time
from scipy.ndimage import *
from scipy.misc.pilutil import imshow
from task1 import Normalize
from task2 import histogram
from task3 import duplicate2D_2D, Corner_

eps = 1e-8

mask = np.asarray([[0, 0, 1, 1, 1, 0, 0],
                   [0, 1, 1, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 0, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 1, 1, 0],
                   [0, 0, 1, 1, 1, 0, 0]],
                  dtype = np.int)

mask_flat = mask.flatten()

fx = np.reshape(np.repeat(np.asarray(range(-3, 4), dtype = np.float32), 7), (7,7))
fy = np.transpose(fx)

dx = np.asarray([fx.flatten().tolist()[i] for i in range(mask_flat.shape[0]) if mask_flat[i] > 0])
dy = np.asarray([fy.flatten().tolist()[i] for i in range(mask_flat.shape[0]) if mask_flat[i] > 0])
dx = np.reshape(dx, (dx.shape[0], 1))
dy = np.reshape(dy, (dy.shape[0], 1))

def USAN(I, mask, t = 32, ratio = 0.5):
    '''
    Get the USAN
    '''
    
    highI = duplicate2D_2D(I, mask)
    distance = np.exp(-np.power((highI - I[:,:,None]) / t, 6))
    
    N = distance.sum(axis = 2)
    g = ratio * N.max()
    
    R = (N <= g).astype(int) * (g - N)
    
    return N, R, distance

def SUSAN_Corner(I, mask):
    '''
    Corner Detection, which Ratio is 0.5 (Edge is 0.75)
    Apply Non-max Suppression (a trick version)
    '''
    N, R, distance = USAN(I, mask, ratio = 0.5)
    
    # Calculate the center point of gravity 
    distance_x = np.reshape(np.dot(distance, dx), distance.shape[:-1])
    distance_y = np.reshape(np.dot(distance, dy), distance.shape[:-1])
    
    center_x = distance_x / (N + eps)
    center_y = distance_y / (N + eps)
    
    # Calculate the distance between mask center and gravity center
    away = np.sqrt(center_x * center_x + center_y * center_y)
    
    # Non-max Suppression
    #Corner = (away > 0.05).astype(int)
    Corner = Corner_(away, 0.98)
    
    return Corner

def Median(I, mask = mask):
    '''
    Median Filter
    '''
    highI = duplicate2D_2D(I, mask)
    M = np.median(highI, axis = 2)
    return M
    
if __name__ == '__main__':
    
    #task 4.1
    I = imread('susan_input1.png')
    if I.ndim == 3:
        I = I[:,:,0]
    imshow(I)
    
    Corner = SUSAN_Corner(I, mask)
    imshow(Corner)
    
    
    #task 4.2
    '''
    SUSAN Corner Detection method works badly on noise
    '''
    I = imread('susan_input2.png')
    if I.ndim == 3:
        I = I[:,:,0]
    imshow(I)
    
    Corner = SUSAN_Corner(I, mask)
    imshow(Corner)
    
    #task 4.3

    '''
    Median Filter can work well on smoothing the Image, but lead some noise on the edges,
    which lead the corner detector becomes an edge detector.
    '''

    I = Median(I)
    imshow(I)
    
    Corner = SUSAN_Corner(I, mask)
    imshow(Corner)
    