from cluster import *
from point import *

def kmeansExt(pointdata, clusterdata) :
    #Fill in
    
    #1. Make list of points using makePointList and pointdata
    points = makePointList(pointdata)
    #2. Make list of clusters using createClusters and clusterdata
    clusters = createClusters(clusterdata)
    #3. As long as points keep moving:
    while(1):
        #A. Move every point to its closest cluster (use Point.closest and
        #   Point.moveToCluster)
        #   Hint: keep track here whether any point changed clusters by
        #         seeing if any moveToCluster call returns "True"
        
        changes = set()
        #get the centers/points in the cluster
        clust = [c for c in clusters]
        #print(clust)
            
        for p in points:
            #find the closest cluster
            temp = p.closest(clust)
            #move the point to its closest cluster
            changes.add(p.moveToCluster(temp))
        
        #B. Update the centers for each cluster (use Cluster.updateCenter)   
        for c in clusters:
            c.updateCenter()
        
        #if no points are moving then break the loop
        if not True in changes:
            break
            
    #4. Return the list of clusters, with the centers in their final positions
    return clusters
    
    
    
if __name__ == '__main__' :
    data = np.array([[0.5, 2.5], [0.3, 4.5], [-0.5, 3], [0, 1.2], [10, -5], [11, -4.5], [8, -3]], dtype=float)
    centers = np.array([[0, 0], [1, 1]], dtype=float)
    
    clusters = kmeansExt(data, centers)
    print(clusters)
    for c in clusters :
        c.printAllPoints()
