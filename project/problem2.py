#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 23:39:29 2020

@author: nadhiraqilah
"""
import pandas as pd
import matplotlib.pyplot as plt
from regularize import regExt
import numpy as np

# define dataset
def classify(doc, features):
    #open doc and read it first
    data = pd.read_csv(doc, sep="\t", header=None)
    t = data[0].value_counts(sort=True)
    print('Total number of students: {}'.format(len(t)))
    #select students that watch at least half of the videos
    numVid = len(data[1].value_counts(sort=True))
    val = [t.index[i] for i in range(len(t)) if t.values[i] >= int(numVid/2)]
    print('Total number of students who watch at least half of the videos or more: {}'.format(len(val)))
    
    #create a new series that only have students with video count at least half
    data.columns = list(data.iloc[0])
    df = data[:][1:].astype({'fracSpent':'float', 'fracComp':'float','fracPaused':'float','numPauses':'float', 
                'avgPBR':'float', 'numRWs':'float', 'numFFs':'float','s': 'float'})
    
    train_x = df[features]
    train_x = train_x[train_x['userID'].isin(val)]
    #print(train_x)
    train_x.set_index('userID',inplace=True)
    train_x = train_x.groupby('userID').agg(np.mean)
    
    train_y = df[['userID','s']]
    train_y = train_y[train_y['userID'].isin(val)]
    #print(train_y)
    train_y.set_index('userID',inplace=True)
    train_y = train_y.groupby('userID').agg(np.mean)
    #print(train_y)
    
    return train_x, train_y
    
    '''
    train_x = df.groupby('userID').agg({'fracSpent':'mean', 'fracComp':'mean','fracPaused':'mean','numPauses':'mean', 
                'avgPBR':'mean', 'numRWs':'mean', 'numFFs':'mean', 's':'mean'}).reset_index()

    select_df = df1[df1['userID'].isin(val)]
    train_x = select_df[features]
    train_y = select_df['s']
        
    train_x = df[features]
    train_x.set_index('VidID',inplace=True)
    train_x = train_x.groupby('VidID',as_index=False).apply(lambda x: x.values.tolist())#.reset_index()#apply(list).reset_index()
    
    train_y = df[['VidID','s']]
    train_y.set_index('VidID',inplace=True)
    train_y = train_y.groupby('VidID').apply(lambda x: x.values.tolist())

    return train_x, train_y
    '''

def predict(train_x,train_y,features):

    model_best, mse, lmbda_best, lmbda, MSE = regExt(train_x,train_y)
    test_x = train_x[int(len(train_x) * 0.91):]
    predicted_y = model_best.predict(test_x)
    actual_y = train_y[int(len(train_y) * 0.91):]
    thres = 0.2
    numCorr = sum(abs(predicted_y-actual_y) < thres)
    print('\nPredicting students by given behavioral features {}'.format(features))
    print('Number of Training Data: {} \n Number of Testing Data:{}' .format(len(train_x), len(test_x)))
    print('Threshold Value: %.1f' %thres)
    print('Model Accuracy: %.2f %%' %(numCorr / len(actual_y) * 100))
    print('Model Coefficients: {}' .format(model_best.coef_))
    print('MSE : %.2f' %mse)
    print('R^2 on test data: %.2f' %(model_best.score(test_x, actual_y)))
    print('Best Lambda Value: {}\n' .format(lmbda_best))
    plt.xlabel('Lambda')
    plt.ylabel('MSE')
    plt.title('MSE values vs Different Lambdas when Training Model')
    plt.plot(lmbda, MSE)
    plt.show()


if __name__ == '__main__' :
    doc = 'behavior-performance.txt'
    features = ['fracSpent', 'fracComp', 'fracPaused',
                'numPauses', 'avgPBR', 'numRWs', 'numFFs']
    features1 = ['userID','fracSpent', 'fracComp', 'fracPaused',
                'numPauses', 'avgPBR', 'numRWs', 'numFFs','s']
    train_x, train_y = classify(doc,features1)
    
    x = train_x.to_numpy(dtype=float)
    y = train_y.to_numpy(dtype=float)
    predict(x,y,features)
    
    