#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 23:39:29 2020

@author: nadhiraqilah
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint as pprint
from kmeans import kmeansExt
from pprint import pprint as pprint
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

def find_numcluster(data):
    #normalize the data first
    #X = normalize(data)
    #X = StandardScaler().fit_transform(data)
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(data)
    #choose the appropriate number of clusters
    kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,}

    # A list holds the SSE values for each k
    sse = []
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 20), sse)
    plt.xticks(range(1, 20))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()    

# define dataset
def classify(doc, features):

    #open doc and read it first
    data = pd.read_csv(doc, sep="\t", header=None)
    t = data[0].value_counts(sort=True)
    print('Total number of students: {}'.format(len(t)))

    #select students with video count >= 5
    val = [t.index[i] for i in range(len(t)) if t.values[i] >= 5]
    print('Total number of students who watch 5 videos or more: {}'.format(len(val)))
    #create a new series that only have students with video count >= 5
    #print(data[0])
    #print(data.loc[data[0].isin(val)])
    data.columns = list(data.iloc[0])
    df = data[:][1:].astype({'fracSpent':'float', 'fracComp':'float','fracPaused':'float','numPauses':'float', 
                'avgPBR':'float', 'numRWs':'float', 'numFFs':'float'})
    df1 = df.groupby('userID').agg({'fracSpent':'mean', 'fracComp':'mean','fracPaused':'mean','numPauses':'mean', 
                'avgPBR':'mean', 'numRWs':'mean', 'numFFs':'mean'}).reset_index()

    select_df = df1[df1['userID'].isin(val)]
    select_df = select_df[features]

    return select_df


def cluster(cluster_val, numFeature, numCluster):
    avgDist = []
    diffDist = []
    sumDist = 0
    print('Clustering {} students by given behavioral features' .format(len(cluster_val)))
    
    for i in numCluster:
        initClusters = np.zeros(shape=(i,numFeature))
        clust = kmeansExt(cluster_val, initClusters)
    
        #Returns the average distance from all of the points in self.points to
        #the current center
        for c in clust:
            if(len(c.points) is not 0):
                sumDist += c.avgDistance
        avgDist.append(sumDist / i)
        sumDist = 0
    
    return avgDist, clust

def test(X):
    #X = data#normalize(data)
    #X = StandardScaler().fit_transform(data)
    #y_pred = KMeans(n_clusters=9).fit_predict(X)
    kmeans = KMeans(n_clusters=6)
    model = kmeans.fit(X)
    print("model\n", model)
    centers = model.cluster_centers_
    print(centers)
    plt.scatter(X[:,0],X[:,1],c=model.labels_);
    plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=100, color="red")
    plt.grid(True)
    plt.show()

if __name__ == '__main__' :
    doc = 'behavior-performance.txt'
    features = ['fracSpent', 'fracComp','fracPaused','numPauses', 
                'avgPBR', 'numRWs', 'numFFs']
    cluster_val = classify(doc,features)
    
    scaler = preprocessing.MinMaxScaler()
    features_normal = scaler.fit_transform(cluster_val)
    
    #find the appropriate kmean number of cluster, uncomment if necessary
    find_numcluster(cluster_val)
    #temp = cluster_val.to_numpy(dtype=float)
    test(features_normal)
    
    avgDist, final_cluster = cluster(features_normal,len(features),range(1,11))
    pprint(avgDist)
    pprint(final_cluster)
    
    '''
    for c in final_cluster:
        c.printAllPoints()
    '''
    plt.xlabel('Number of clusters')
    plt.ylabel('AvgDist of Points to Cluster')
    plt.title('AvgDist of Points to Cluster over Different Number of Clusters')
    plt.plot(range(1,11), avgDist)
    plt.show()
    
    