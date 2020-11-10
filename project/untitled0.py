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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from pandas.plotting import parallel_coordinates
import seaborn as sns
from kmeans import kmeansExt

def pd_centers(featuresUsed, centers):
	colNames = list(featuresUsed)
	colNames.append('prediction')

	# Zip with a column called 'prediction' (index)
	Z = [np.append(A, index) for index, A in enumerate(centers)]

	# Convert to pandas data frame for plotting
	P = pd.DataFrame(Z, columns=colNames)
	P['prediction'] = P['prediction'].astype(int)
	return P

def parallel_plot(data):
	my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
	plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])
	parallel_coordinates(data, 'prediction', color = my_colors, marker='o')

def find_numcluster(data):
    
    X = StandardScaler().fit_transform(data)
    #print(X)
    #choose the appropriate number of clusters
    kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,}

    # A list holds the SSE values for each k
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
    
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()    

# define dataset
def classify(doc, features):

    #open doc and read it first
    data = pd.read_csv(doc, sep="\t", header=None)

    '''
    t = data[0].value_counts(sort=True)
    
    val = [t.index[i] for i in range(len(t)) if t.values[i] >= 5]
    data.sort_values(by=0,inplace=True)    

    
    #plot scatterplot based on each video-watching behavior
    temp = data[data[0].isin(val)]
    print(len(temp))
    #print(temp)
    #print(len(data))
    for col in range(2, len(temp.columns)-1):
        x = np.array(temp[col].values,dtype='float64')
        y = np.array(temp[col+1].values,dtype='float64')

        plt.scatter(x, y)
        #plt.scatter(X[:,0],X[:,1],c=model.labels_);
        #plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=100, color="red")
        plt.xlabel(col_name[col])
        plt.ylabel(col_name[col+1])
        plt.grid(True)
        #plt.show()
        #print(X)
    '''
    '''
    #data.columns = list(data.iloc[0])
    tempX = np.array(data,dtype=float)
    X = StandardScaler().fit_transform(tempX)
    print(X)
    model = KMeans(n_clusters=4)
    model.fit(X)
    print(model.cluster_centers_)
    '''
    t = data[0].value_counts(sort=True)
    print('Total number of students: {}'.format(len(t)))
    #select students with video count >= 5
    val = [t.index[i] for i in range(len(t)) if t.values[i] >= 5]
    print('Total number of students who watch 5 videos or more: {}'.format(len(val)))
    #data.sort_values(by=0,inplace=True)    
    
    #plot scatterplot based on each video-watching behavior
    temp = data[data[0].isin(val)]
    temp.columns = list(data.iloc[0])
    select_df = temp[features][1:].astype(float)
    find_numcluster(select_df)

    X = StandardScaler().fit_transform(select_df)
    kmeans = KMeans(n_clusters=8).fit(X)
    #calculate the center for the initial cluster
    centers = kmeans.cluster_centers_
    P = pd_centers(features, centers)
    print(P)
    
    labels = pd.DataFrame(kmeans.labels_)
    print(labels)
    '''
    labels.sort_values(by=0, inplace=True)
    pop = labels.to_numpy()
    print(pop)
    '''
    
    labeledData = pd.concat((select_df,labels),axis=1)
    labeledData = labeledData.rename({0:'labels'},axis=1)
    print(labeledData.iloc[:24575])
    #sns.pairplot(labeledData.iloc[:24575],hue='labels')
    labeledData.sort_values(by='labels',inplace=True)
    print(labeledData)
    help = labeledData['labels'].value_counts(sort=True)
    print(help)
    
    return select_df
    '''
    labeledData = pd.concat((features,labels),axis=1)
    labeledData = labeledColleges.rename({0:'labels'},axis=1)
    print(labeledData.head())
    
    model = kmeans.fit(X)
    y_kmeans = kmeans.fit_predict(X)
    '''

    

    

    #sns.lmplot(x='Top10perc',y='S.F.Ratio',data=labeledData,hue='labels',fit_reg=False)
    
    
    '''
    kmeans = KMeans(n_clusters=9)
    model = kmeans.fit(X)
    print("model\n", model)
    centers = model.cluster_centers_
    print(centers)
    plt.scatter(X[:,0],X[:,1],c=model.labels_);
    plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=100, color="red")
    plt.grid(True)
    '''


def cluster(cluster_val, numFeature, numCluster):
    avgDist = []
    diffDist = []
    sumDist = 0
    print('Clustering {} students by given behavioral features' .format(len(cluster_val)))
    #for i in numCluster:
    initClusters = np.zeros(shape=(numCluster,numFeature))
    clust = kmeansExt(cluster_val, initClusters)
    
    #Returns the average distance from all of the points in self.points to
    #the current center
    for c in clust:
        if(len(c.points) is not 0):
            sumDist += c.avgDistance
        avgDist.append(sumDist / numCluster)
        sumDist = 0
    '''
    for c in clust:
        c.printAllPoints()
    
    for c in clust:
        for p in c.points:
            diffDist.append(p.distFrom(p.closest()) - p.distFrom(p.secondClosest()))
    diffDist = np.divide(sum(diffDist), len(diffDist))
    print(diffDist)
    '''
    return avgDist#, diffDist
    
    

    
#using path 2

#How well can the students be naturally grouped or clustered by their video-watching behavior 
#(fracSpent, fracComp, fracPaused, numPauses, avgPBR, numRWs, and numFFs)? 
#You should use all students that complete at least five of the videos in your analysis.

if __name__ == '__main__' :
    doc = 'behavior-performance.txt'
    features = ['fracSpent', 'fracComp','fracPaused','numPauses', 
                'avgPBR', 'numRWs', 'numFFs']
    cluster_val = classify(doc,features)
    temp = cluster_val.to_numpy(dtype=float)
    dist = cluster(temp,len(features),8)