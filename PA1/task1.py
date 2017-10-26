import numpy as np
import scipy as sci
import scipy.ndimage as ndi
import math
from scipy.ndimage import *
from scipy.misc.pilutil import imshow


def Gaussian(x, sigma = 1):
    '''
    Gaussian function
    
    '''
    y = 1/(math.sqrt(2*math.pi) * sigma) * math.exp(- x*x / (2* sigma * sigma))
    return y

def dGaussian(x, sigma = 1):
    '''
    derivative of Gaussian function 
    '''
    y = - x/(math.sqrt(2*math.pi) * sigma * sigma * sigma) * math.exp(- x*x / (2* sigma * sigma))
    return y

def G(size = 3, sigma = 1):
    '''
    1D-Gaussian Filter
    '''
    x = range(size)
    g = [Gaussian(i, sigma) for i in x]
    return g

def dG(size = 3, sigma = 1):
    '''
    1st Derivative of Gaussian Filter
    '''
    x = range(size)
    g = [dGaussian(i, sigma) for i in x]
    return g

def dualG(size = 11, sigma = 1):
    '''
    Get the Gaussian filter from -size/2 to size/2
    '''
    g = G(size/2+1, sigma)
    g = g[::-1]+g[1:]
    return g

def dualdG(size = 11, sigma = 1):
    '''
    Get the 1st Dirivative of Gaussian filter from -size/2 to size/2
    '''
    g = dG(size/2+1, sigma)
    g_ = [-x for x in g[::-1]]
    g = g_+g[1:]
    return g

def Conv2D_1D(I, filter, padding = True):
    '''
    Apply filter on Image with Same-size Padding
    '''
    size = len(filter)
    shp = I.shape
    
    if padding:
        I = np.concatenate([np.zeros((shp[0], size/2), dtype = np.float32),
                            I,
                            np.zeros((shp[0], size/2), dtype = np.float32)], axis = 1)
    
    # Stack size layers Images together
    highI = np.empty((shp[0], shp[1], 0), dtype = np.float32)
    for i in range(size):
        subI = I[:,i:i+shp[1]]
        subI = np.expand_dims(subI, axis = 2)
        highI = np.append(highI, subI, axis = 2)
    # Do a dot product instead of using for loops
    newI = np.dot(highI, np.transpose(np.asmatrix([filter], dtype = np.float32)))
    return newI

def doFilter(I, sigma = 1):
    '''
    Apply Gaussian and first derivative Gaussian filters on Image
    '''
    filter_G = dualG(sigma = sigma)
    filter_dG = dualdG(sigma = sigma)
    
    I_x = np.transpose(Conv2D_1D(np.transpose(I), filter_G))
    I_y = Conv2D_1D(I, filter_G)
    
    I_x_2 = np.transpose(Conv2D_1D(np.transpose(I_x), filter_dG))
    I_y_2 = Conv2D_1D(I_y, filter_dG)
    return I_x, I_y, I_x_2, I_y_2
    

def Magnitude(I_x_2, I_y_2):
    '''
    Get the Magnitude filed of Image
    '''
    M_x = np.multiply(I_x_2, I_x_2)
    M_y = np.multiply(I_y_2, I_y_2)
    M = np.sqrt(M_x + M_y)
    return M

'''
0 for -pi and pi
1 for -3/4 pi
2 for -1/2 pi
3 for -1/4 pi
4 for 0
5 for 1/4 pi
6 for 1/2 pi
7 for 3/4 pi
'''
# the direction of neighbor pixels
dx = np.asarray([-1, -1, 0, 1, 1, 1, 0, -1], dtype = np.int32)
dy = np.asarray([0, -1, -1, -1, 0, 1, 1, 1], dtype = np.int32)

def hysteresis_threshold(I, low, high):
    '''
    Hysteresis Thresholding Method
    '''
    mask_low = I > low
    mask_high = I > high
    # Connected components of mask_low (Flood Fill)
    labels_low, num_labels = ndi.label(mask_low)
    # Check which connected components contain pixels from mask_high
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]
    return thresholded

def Normalize(I):
    '''
    Due to the difference on value scale,
    We need Normalization on Image scale [0..255]
    '''
    low = I.min()
    high = I.max()
    I = (I-low) * 1.0 / (high - low)
    return I * 255


def Candy(I, sigma = 1):

    I_x, I_y, I_x_2, I_y_2 = doFilter(I, sigma)
    
    # Get the Magnitude Filed
    M = Magnitude(I_x_2, I_y_2)
    
    # Get the direction of gradient
    theta = np.arctan2(I_y_2, I_x_2) / math.pi * 4 + 4
    
    # Do a Mapping from arc degree to Index of direction
    switcher = np.rint(theta).astype(int)
    switcher = switcher - (switcher == 8).astype(int) * 8
    
    # Directions at every pixels
    f_x = dx[switcher]
    f_y = dy[switcher]
    
    # Padding one pixel around
    M_padding = np.zeros((M.shape[0]+2, M.shape[1]+2), dtype = np.float32)
    M_padding[1:-1,1:-1] = M
    
    # The position of pixel itself
    ID_x = np.transpose(np.tile(np.asarray(range(M_padding.shape[0]), dtype = np.int32), (M_padding.shape[1],1)))
    ID_y = np.tile(np.asarray(range(M_padding.shape[1]), dtype = np.int32), (M_padding.shape[0],1))
    
    # The Positive Gradient Direction
    P_x = ID_x 
    P_y = ID_y
    P_x[1:-1,1:-1] += f_x
    P_y[1:-1,1:-1] += f_y
    
    # The Negative Gradient Direction
    Q_x = ID_x
    Q_y = ID_y
    Q_x[1:-1,1:-1] -= f_x
    Q_x[1:-1,1:-1] -= f_y
    
    # M_P for Positive Pixel, M_Q for Negative Pixel 
    M_P = M_padding[P_x, P_y]
    M_Q = M_padding[Q_x, Q_y]
    
    #Non-Maximum Suppression
    #Get the Maximum of 3, M
    MM = np.maximum(M_P, M_Q)
    MM = np.maximum(MM, M_padding)
    
    # Check if equal to maximum
    Exist = (( M_padding - MM < 1e-6) & (MM - M_padding < 1e-6 )).astype(int)
    
    # After Non-Maximum Suppression
    M_new = np.multiply(M_padding, Exist)[1:-1, 1:-1]
    
    # Do Hysteresis Thresholding
    M_threshold = hysteresis_threshold(Normalize(M_new), 25, 64).astype(int)
    
    #Pack and Return All the Result Graphs For Demo
    return [I_x, I_y, I_x_2, I_y_2], [I, M, M_new, M_threshold]

if __name__ == '__main__':
    # Input the Image I
    I = imread('Image2.jpg')

    # Get the Graphs
    IG, Candies = Candy(I, 0.1)

    # Do concatenation to demo
    outputImage = np.concatenate([np.concatenate([Normalize(img) for img in IG], axis = 1),
                                  np.concatenate([Normalize(img) for img in Candies], axis = 1)],
                                 axis = 0)

    # Demo
    imshow(outputImage)


    # It seems the less sigma value is the better performance we get
    # less sigma value can obtain more details

    sigmas = [0.1, 1 ,10, 100]
    outputs = []
    for sig in sigmas:
        _, Candies = Candy(I, sig)
        outputs.append(Candies[3])

    outputImages = np.concatenate(outputs, axis = 1)
    imshow(outputImages)
