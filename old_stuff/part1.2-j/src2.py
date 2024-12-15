import scipy 
import numpy as np


class DMatch:
    def __init__(self, _queryIdx, _trainIdx, _distance):
        self.queryIdx = _queryIdx
        self.trainIdx = _trainIdx
        self.distance = _distance
    
    
    def __repr__(self):
        return 'DMatch(queryIdx: %s, trainIdx: %s, distance: %s)' % (self.queryIdx, self.trainIdx, self.distance)


# Function to calculate Euclidean distance
def euclidean_distance(descriptor1, descriptor2):
    return np.sqrt(np.sum((descriptor1 - descriptor2) ** 2))


# Function to find the K nearest neighbors
def knn_match_vectorized(descriptors_query, descriptors_train, k):
    distances = scipy.spatial.distance.cdist(descriptors_query, descriptors_train, 'euclidean') ## calculate the distances between each query descriptor and each train descriptor: i -> query, j -> train
    indices = np.argsort(distances, axis=1) ## order for each row (query descriptor) the indices of the train descriptors
    matches = np.zeros((descriptors_query.shape[0], k), dtype=object)
    
    for i in range(descriptors_query.shape[0]):
        k_nearest_neighbors = [(index, distances[i, index]) for index in indices[i, :k]]
        match = np.array([DMatch(_queryIdx=i, _trainIdx=neighbor[0], _distance=neighbor[1]) for neighbor in k_nearest_neighbors])
        matches[i, :] = match
        
    return matches



def ratio_test(matches, ratio=0.75):
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
            
    return np.array(good_matches)