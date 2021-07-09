# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:16:10 2021

@author: Param
"""
from helper_functions import *
# importing data
data=get_data("simulation Data")
# balancing data
balanced_data=data_balancing(data)

#data.drop(data.loc[data['Steering']<-0.9].index, inplace=True)
#data.drop(data.loc[data['Steering']>0.9].index, inplace=True)
#data = data.reset_index(drop=True)

imagesPath, steerings = load_data("simulation Data",data)
print(imagesPath[0],steerings[0])

# Training and Validation Split
from sklearn.model_selection import train_test_split
xtrain, xval, ytrain, yval = train_test_split(imagesPath, steerings, test_size=0.2,random_state=42)
print('Total Training Images: ',len(xtrain))
print('Total Validation Images: ',len(xval))
