import os
import cPickle as Pickle
import numpy as np


def saveToPKL(filename, data):
    with open(filename,'wb')as f:
        Pickle.dump(data, f)
    return 

def loadFromPKL(filename):
    with open(filename,'rb') as f:
        data = Pickle.load(f)
    return data

actions = ['Diving','Golf-Swing','Kicking','Lifting','Riding-Horse','Run','SkateBoarding','Swing-Bench','Swing-SideAngle','Walk']

if __name__ == '__main__':
    i2l = loadFromPKL('i2l.pkl')
    count = loadFromPKL('count.pkl')
    print i2l
    print count
    
    N = len(i2l)
    M = len(actions)
    map = []
    for i in range(N):
        for j in range(M):
            if actions[j] in i2l[i]:
                map.append(j)
                break
    
    print N, M, map
    count_new = np.zeros((M, M), dtype = np.int)
    for i in range(N):
        for j in range(N):
            count_new[map[i],map[j]] += count[i,j]
    
    print count_new
    correct = np.sum(np.diag(count_new))
    all = np.sum(count_new)
    print correct * 1.0 / all
    
    precision = np.diag(count_new) * 1.0 / np.sum(count_new, axis = 0)
    recall = np.diag(count_new) * 1.0 / np.sum(count_new, axis = 1)
    
    print precision
    print recall
    print 2 * (precision * recall) / (precision + recall)