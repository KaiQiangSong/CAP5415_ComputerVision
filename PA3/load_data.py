import os
import cPickle as Pickle
import numpy as np

from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input, decode_predictions

from keras.models import Model

def saveToPKL(filename, data):
    with open(filename,'wb')as f:
        Pickle.dump(data, f)
    return 

def loadFromPKL(filename):
    with open(filename,'rb') as f:
        data = Pickle.load(f)
    return data

def extracFeature(fileName, model):
    img = image.load_img(fileName, target_size=(224,224))
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    features = model.predict(x)
    return features

if __name__ == '__main__':
    path = './ucf_action'
    labels = os.listdir(path)
    Index = 0
    dataset = []
    for label_i in labels:
        label_path = os.path.join(path, label_i)
        videos = os.listdir(label_path)
        for video_j in videos:
            video_path = os.path.join(label_path, video_j)
            frames = os.listdir(video_path)
            frames = [f for f in frames if f.endswith(".jpg")]
            frames = sorted(frames)
            dataset.append((label_i, video_j, frames))
    
    
    saveToPKL('dataset.pkl', dataset)
    
    base_model = VGG16(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
    
    features = []
    for data in dataset:
        for frame in data[2]:
            fileName = path+'/'+data[0]+'/'+data[1]+'/'+ frame
            feature = extracFeature(fileName, model)
            features.append(feature)
            print fileName, feature.shape
    
    features = np.concatenate(features, axis = 0)
    print features.shape
    
    saveToPKL('features.pkl', features)