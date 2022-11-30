import numpy as np

class LogisticRegression:
    def __init__(self, lr, num_iter):
        self.lr = lr
        self.num_iter = num_iter
        self.theta = None
        
    def sigmoid(self, x):  # 活性化関数
        return 1/(1+np.exp(-x))
    
    def crossentropy(self, x, y):
        return (-y * np.log(x) - (1-y) * np.log(1-x)).mean()  # クロスエントロピー
    
    def fit(self, X, y):
        y = y.reshape(-1, 1)
        self.theta = np.zeros((X.shape[1], 1))  # 重みの初期化
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        # 学習
        for i in range(self.num_iter):
            gradient = np.dot(X.T, (h-y)) / y.shape[0]
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            loss = self.crossentropy(h, y)
            
            if i%10000 == 0:
                print(f'loss: {loss}')
                
    def predict_prob(self, X):
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        return h
                
    def predict(self, X):
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        return h.round()