import numpy as np

class KNN:
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        assert weights == 'uniform' or weights == 'distance', 'weights support uniform or distance'
        self.weights = weights
        self.X = None
        self.y = None
        
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, test):
        assert (self.X != None).any() and (self.y != None).any(), 'please fit model'
        distances = np.zeros(len(self.X))
        for i, x in enumerate(self.X):
            distances[i] = np.linalg.norm(x - test)  # ユークリッド距離
        argsort = np.argsort(distances)
        neighbors = argsort[ :self.n_neighbors]
        labels = self.y[neighbors]
        
        if self.weights == 'uniform':
            uni_labels, votes = np.unique(labels, return_counts=True)
            predict = uni_labels[np.argmax(votes)]
            
        elif self.weights == 'distance':
            uni_labels = np.unique(labels)
            neighbor_distances = distances[neighbors]
            neighbor_distances = 1-(neighbor_distances/sum(neighbor_distances))
            mat = np.concatenate([labels.reshape(-1, 1), neighbor_distances.reshape(-1, 1)], axis=1)
            
            best_score = 0
            for label in uni_labels:
                score = mat[mat[:, 0]==label][:, 1].sum()
                if best_score < score:
                    best_score = score
                    best_label = label
                
            predict = best_label

        return predict