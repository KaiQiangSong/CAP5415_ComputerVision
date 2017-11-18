import os
import cPickle as Pickle
import numpy as np

from sklearn import svm

def saveToPKL(filename, data):
    with open(filename,'wb')as f:
        Pickle.dump(data, f)
    return 

def loadFromPKL(filename):
    with open(filename,'rb') as f:
        data = Pickle.load(f)
    return data

actions = ['Diving','Golf-Swing','Kicking','Lifting','Riding-Horse','Run','SkateBoarding','Swing','Walk']

if __name__ == '__main__':
    dataset = loadFromPKL('dataset.pkl')
    features = loadFromPKL('features.pkl')
    
    # Mapping Label to Index
    n_Label = 0
    Labels = []
    l2i = {}
    i2l = []
    for data in dataset:
        if data[0] not in l2i:
            l2i[data[0]] = n_Label
            i2l.append(data[0])
            n_Label += 1
        Labels += [l2i[data[0]]] * len(data[2])
                
    Labels = np.asarray(Labels, dtype = np.int)
    
    count = np.zeros((len(i2l), len(i2l)), dtype = np.int)
    
    st = 0
    for testVideo in dataset:
        ed = st + len(testVideo[2])
        trainX = np.concatenate([features[:st], features[ed:]], axis = 0)
        trainY = np.concatenate([Labels[:st], Labels[ed:]], axis = 0)
        testX = features[st:ed]
        testY = Labels[st:ed]
        print trainX.shape, trainY.shape
        print testX.shape, testY.shape
        print st, ed
        st = ed
        
        clf = svm.SVC(kernel = 'linear', decision_function_shape='ovo')
        clf.fit(trainX,trainY)
        y = clf.predict(testX)
        predict = np.argmax(np.bincount(y))
        
        print y
        print testY
        print 'Predict %s -> %s'%(i2l[testY[0]], i2l[predict])
        
        count[testY[0], predict] += 1
        
    print count
    print i2l
    saveToPKL('i2l.pkl', i2l)
    saveToPKL('count.pkl', count)
    
    