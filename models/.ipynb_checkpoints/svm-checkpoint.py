import numpy as np

class HardMarginSVM:
    def __init__(self, lr=0.001, epoch=1000, random_state=None):
        self.lr = lr
        self.epoch = epoch
        self.random_state = random_state
        self.is_trained = False
        
        
    def fit(self, X, y):
        self.num_samples = X.shape[0]
        self.num_features = X.shape[1]
        self.w = np.zeros(self.num_features)
        self.b = 0
        
        rgen = np.random.RandomState(self.random_state)
        self.alpha = rgen.normal(loc=0.0, scale=0.01, size=self.num_samples)
        
        for _ in range(self.epoch):
            self._cycle(X, y)
            
        indexes_sv = [i for i in range(self.num_samples) if self.alpha[i] != 0]
        for i in indexes_sv:
            self.w += self.alpha[i] * y[i] * X[i]
        for i in indexes_sv:
            self.b += y[i] - (self.w @ X[i])
        self.b /= len(indexes_sv)
        self.is_trained = True
        
        
    def predict(self, X):
        assert self.is_trained == True, 'Model not fitted'
        logit = X @ self.w + self.b
        result = np.where(logit > 0, 1, -1)
        return result
    
    
    def _cycle(self, X, y):
        y = y.reshape([-1, 1])  # (num_samples, 1)の行列にreshape
        H = (y @ y.T) * (X @ X.T)
        grad = np.ones(self.num_samples) - H @ self.alpha
        self.alpha += self.lr * grad
        self.alpha = np.where(self.alpha < 0, 0, self.alpha)
        
class SoftMarginSVM:
    # C: lambda
    def __init__(self, lr=0.001, epoch=10000, C=1.0, random_state=None):
        self.lr = lr
        self.epoch = epoch
        self.C = C
        self.random_state = random_state
        self.is_trained = False
    
    def fit(self, X, y):
        self.num_samples = X.shape[0]
        self.num_features = X.shape[1]
        # 重みの初期化
        self.w = np.zeros(self.num_features)
        self.b = 0
        
        # alpha(未定乗数)を初期化
        rgen = np.random.RandomState(self.random_state)
        self.alpha = rgen.normal(loc=0.0, scale=0.01, size=self.num_samples)
        
        for _ in range(self.epoch):
            self._cycle(X, y)
            
        # サポートベクトル / 超平面内部の点のindexを取得
        # 外側は正しく分類できている
        indexes_sv = []
        indexes_inner = []
        for i in range(self.num_samples):
            if 0 < self.alpha[i] < self.C: 
                indexes_sv.append(i)
            elif self.alpha[i] == self.C: #誤って分類
                indexes_inner.append(i)
        for i in indexes_sv + indexes_inner:
            self.w += self.alpha[i] * y[i] * X[i]
        for i in indexes_sv:
            self.b += y[i] - (self.w @ X[i])
        self.b /= len(indexes_sv)
        
        self.is_trained = True 
        
    def predict(self, X):
        assert self.is_trained == True, 'Model not fitted'
        logit = X @ self.w + self.b
        result = np.where(logit > 0, 1, -1)
        return result
        
    def _cycle(self, X, y):
        y = y.reshape([-1, 1])  # (num_samples, 1)
        H = (y @ y.T) * (X @ X.T)
        grad = np.ones(self.num_samples) - H @ self.alpha
        self.alpha += self.lr * grad
        # 制約 0 <= alpha(未定乗数) <= lambda
        self.alpha = np.where(self.alpha < 0, 0, self.alpha)
        self.alpha = np.where(self.alpha > self.C, self.C, self.alpha)