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

def Histogram_Equalization(I):
    size = I.shape[0] * I.shape[1]
        
    hist = histogram(I)
    hist_sum = np.cumsum(hist)
    
    map = np.asarray([round(255.0 * hist_sum[i] / size) for i in range(256)], dtype = np.float32)
    
    return map[Normalize(I).astype(int)]

def Clipping(I, a = 50, b = 150, beta = 2):
    
    mask_a = ((a <= I) & (I < b)).astype(int)
    mask_b = (I >= b).astype(int)
    
    return mask_a * I * beta + mask_b * np.ones_like(I) * beta * (b - a)

def Range_Compression(I, c):
    map = c * np.log10(1 + np.asarray(range(256), dtype = np.float32))
    return map[Normalize(I).astype(int)]

if __name__ == '__main__':
    
    I = imread('Image1.jpg')
    if I.ndim == 3:
        I = I[:,:,0]
        
    imshow(I)
    
    # task 5.1
    
    I_HE = Histogram_Equalization(I)
    imshow(I_HE)
    
    # task 5.2
    '''
    The result seems to much more clear than original Image,
    since it stretches the sensitive pixels to a larger brightness range
    and block all the non-sensitive pixels 
    '''
    
    I_Clip = Clipping(I)
    imshow(I_Clip)
    
    #task 5.3
    '''
    Much brighter than before.
    For the reason above, there's no much difference using different c.
    '''
    for c in [1, 10, 100, 1000]:
        I_RC = Range_Compression(I, c)
        imshow(I_RC)
    
    
    