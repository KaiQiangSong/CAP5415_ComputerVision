import numpy as np
import scipy as sci
import scipy.ndimage as ndi
import math, time
from scipy.ndimage import *
from scipy.misc.pilutil import imshow

def histogram(I):
    Ones = np.ones_like(I)
    result = ndi.sum(Ones, I, index = range(256))
    return result.astype(int)

EPS = 1e-8

def Entropy(hist, hist_sum):
    '''
    256 X Speed than Calculate One By One
    
    
    H_A_i = -1/P * (Sigma (P_j Log P_j) - Sigma (P_j) Log P) (j = 0..i)
    P = Sigma P_j (j = 0..i)
    
    Let S_i = Sigma(P_j) (j from 0 to i)
    Let Q_i = Sigma(P_j Log P_j) (j from 0 to i)
    
    H_A_i = - (Q_i - S_i log S_i) / S_i
    
    Similar to H_B_i
    
    Since We only need the prefix summation of a list of values, We can solve it in O(Range of Colors)
    '''
    hist_log = hist * np.log(hist + EPS)
    hist_log_sum = np.cumsum(hist_log)
    
    H_A = - (hist_log_sum - hist_sum * np.log(hist_sum + EPS)) / (hist_sum + EPS)
    #print H_A
    
    hist_total = hist.sum()
    hist_log_total = hist_log.sum()
    
    hist_sum_ = hist_total - hist_sum
    hist_log_sum_ = hist_log_total - hist_log_sum
    
    H_B = - (hist_log_sum_ - hist_sum_ * np.log(hist_sum_ + EPS)) / (hist_sum_ + EPS)
    #print HB
    H = H_A + H_B
    return H

if __name__ == '__main__':
    st = time.time()
    I = imread('Image1.jpg')
    hist = histogram(I)
    hist_sum = np.cumsum(hist)
    H = Entropy(hist, hist_sum)
    print np.argmax(H)
    I_Thresh = (I>= np.argmax(H))
    print time.time() - st
    imshow(I_Thresh)