# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:16:50 2021

@author: Param
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as aug
import cv2

def get_data(path):
    columns=['Center','Left','Right','Steering','Throttle','Brake','Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names = columns)
    data['Center']=data['Center'].apply(get_file)
   
    return data
    
def get_file(filePath):
    return filePath.split('\\')[-1]

def data_balancing(data,display=True):
    
    Bins=31
    samples_per_Bin=35
    hist, bins = np.histogram(data['Steering'], Bins)
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samples_per_Bin, samples_per_Bin))
        plt.show()
        
    removeindexList = []
    for j in range(Bins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samples_per_Bin:]
        removeindexList.extend(binDataList)
 
    print('Removed Images:', len(removeindexList))
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data)) 
    if display:
        hist, _ = np.histogram(data['Steering'], (Bins))
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samples_per_Bin, samples_per_Bin))
        plt.show()

def load_data(path, data):
  imagesPath = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    imagesPath.append(f'{path}/IMG/{indexed_data[0]}')
    steering.append(float(indexed_data[3]))
  imagesPath = np.asarray(imagesPath)
  steering = np.asarray(steering)
  return imagesPath, steering

# Augment the images randomly
def augment_img(img_path,steering):
    img=mpimg.imread(img_path)
    
    if np.random.rand(),0.5:
     pan=aug.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
     img=pan.augment_image(img)
   
    if np.random.rand(),0.5: 
     zoom=aug.Affine(scale=(1,1.4))
     img=zoom.augment_image(img)
    
    if np.random.rand(),0.5: 
     brightness=aug.multiply((0.3,1.3))
     img=brightness.augment_image(img)
    
    if np.random.rand(),0.5: 
     img=cv2.flip(img,1)
     steering=-steering
    
    return img, steering 

def preprocess(img):
    img=img[60:135,:,:]
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img=cv2.GaussianBlur(img,(3,3),0)
    img=cv2.resize(img,(200,66))
    img=img/255
    return img

def batch_gen(img_paths,steeeringlist,batch_size,training):
    while True:
        img_batch=[]
        steering_batch=[]
        for i in range(batch_size):
            idx=random.randint(0,len(img_paths)-1)
            if training:
              img,steering=augment_img(img_paths[idx],steeringlist[idx])
            else:# we do not want to augment our validation data
                img=mpimg.imread(img_paths[idx])
                steering=steeringlist[idx]
            img=preprocess(img)
            img_batch.append(img)
            steering_batch.append(steering)
        yield(np.asarray(img_batch),np.asarray(steering_batch))    








    
    














