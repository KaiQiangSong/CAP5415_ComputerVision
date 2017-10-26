import matplotlib.pyplot as plt
import cv2
import numpy as np
import numpy.linalg as la
import scipy as sci
import scipy.ndimage as ndi

def showImg(img, save = None, t = 1000):
    cv2.imshow('basketball',img)
    cv2.waitKey(t)
    if save is not None:
        cv2.imwrite(save, img)
        
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

def buildMask(size = 5):
    mask = np.ones((size, size), dtype = np.int)
    return mask

def getTMatrix(f_x, f_y, f_t, mask):
    high_x = duplicate2D_2D(f_x, mask)
    high_y = duplicate2D_2D(f_y, mask)
    high_t = duplicate2D_2D(f_t, mask)
    
    T_xx = np.sum(np.multiply(high_x, high_x), axis = 2)
    T_xy = np.sum(np.multiply(high_x, high_y), axis = 2)
    T_yy = np.sum(np.multiply(high_y, high_y), axis = 2)
    T_xt = np.sum(np.multiply(high_x, high_t), axis = 2)
    T_yt = np.sum(np.multiply(high_y, high_t), axis = 2)
    
    u = (T_yt * T_xy - T_xt * T_yy) / (T_xx * T_yy - T_xy * T_xy)
    v = (T_xt * T_xy - T_yt * T_xx) / (T_xx * T_yy - T_xy * T_xy)
    return u,v 

def Pyramids(img, level = 3):
    I = img
    result = [I]
    for i in range(level):
        I = cv2.pyrDown(I)
        result.append(I)
    return result

def OpticalFlow(img1, img2):
    f_x, f_y = derivative(img1)
    f_t = img1 - img2
    filter = buildMask(3)
    u, v = getTMatrix(f_x, f_y, f_t, filter)
    Corners = cv2.goodFeaturesToTrack(img1, 100, 0.3, 7, blockSize = 7)
    Corners = np.reshape(Corners, (Corners.shape[0], Corners.shape[2])).astype(int)
    
    result = []
    for p in Corners.tolist():
        y, x = p
        p0 = (x, y)
        dp = (u[x, y], v[x, y])
        result.append((p0,dp))
    return result

def Arrow(ax, line):
    p0 = line[0]
    dp = line[1]
    print line
    ax.arrow(p0[0], p0[1], dp[0], dp[1], head_width=0.05, head_length=0.1, fc='k', ec='k')
    return ax

def Question1(prefix):
    f1Name = prefix+'1.png'
    f2Name = prefix+'2.png'
    img1 = cv2.imread(f1Name)
    img2 = cv2.imread(f2Name)
    img = img1
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    result = OpticalFlow(img1, img2)
    ax = plt.axes()
    ax.set_xlim([img.shape[0], 0])
    ax.set_ylim([0, img.shape[1]])
    
    for line in result:
        Arrow(ax, line)
    plt.show()
    
    
if __name__ == '__main__':
    Question1('basketball')
    Question1('grove')
    
        