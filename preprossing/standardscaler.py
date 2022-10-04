import numpy as np
class StandardScaler():
    """
    This class assumes that X is the following
    X: (n_samples, n_features)
    """
    def __init__(self, ):
        self.mean = None
        self.std = None
    def fit(self, x, axis=0):
        self.mean = np.mean(x, axis=axis)
        self.std = np.std(x, axis=axis)
        return None
    def transform(self, x):
         return (x - self.mean) / self.std
    def fit_transform(self, x, axis=0):
        self.mean = np.mean(x, axis=axis)
        self.std = np.std(x, axis=axis)    
        return (x - self.mean) / self.std