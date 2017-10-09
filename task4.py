import numpy as np
import numpy.ma as ma
import numpy.linalg as la
import scipy as sci
import scipy.ndimage as ndi
import math, time
from scipy.ndimage import *
from scipy.misc.pilutil import imshow
from task1 import Normalize

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



def duplicate2D_2D(I, mask, padding = True):
    
    
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

def USAN(I, mask, t = 255, ratio = 0.5):
    
    highI = duplicate2D_2D(I, mask)
    distance = np.power((highI - I[:,:,None]) / t, 6)
    
    N = distance.sum(axis = 2)
    g = ratio * N.max()
    
    
    R = (N <= g).astype(int) * (g - N)
    
    return N, R, distance

def SUSAN_Corner(I, mask):
    N, R, distance = USAN(I, mask, ratio = 0.5)
    
    distance_x = np.reshape(np.dot(distance, dx), distance.shape[:-1])
    distance_y = np.reshape(np.dot(distance, dy), distance.shape[:-1])
    
    center_x = distance_x / (N + eps)
    center_y = distance_y / (N + eps)
    
    away = np.sqrt(center_x * center_x + center_y * center_y)
    Corner = (away > 3).astype(int)
    
    return Corner

def Median(I):
    highI = duplicate2D_2D(I, mask)
    M = np.median(highI, axis = 2)
    return M
    
if __name__ == '__main__':
    
    #task 4.1
    I = imread('susan_input1.png')
    if I.ndim == 3:
        I = I[:,:,0]
    
    Corner = SUSAN_Corner(I, mask)
    imshow(Corner)
    
    
    #task 4.2
    '''
    SUSAN Corner Detection method works badly on noise
    '''
    I = imread('susan_input2.png')
    if I.ndim == 3:
        I = I[:,:,0]
    
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
    