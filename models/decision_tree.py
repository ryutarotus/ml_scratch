import numpy as np

class DecisionTree():
    def __init__(self, data, target, max_depth):
        self.data = data #再帰的にクラスを定義するためここで定義する
        self.target = target
        self.max_depth = max_depth
        self.depth = None
        self.threshold = None
        self.feature = None
        self.gini_min = None
        self.left = None
        self.right = None
        self.label = np.argmax(np.bincount(target))
    
    def fit(self, depth):
        self.depth = depth
        self.gini_min, self.threshold, self.feature = self.search_best_split()
        
        if self.depth == self.max_depth or self.gini_min == 0: #停止条件
            return
        
        idx_left =  self.data[:, self.feature] >= self.threshold
        idx_right =  self.data[:, self.feature] < self.threshold
        
        self.left = DecisionTree(self.data[idx_left], self.target[idx_left], self.max_depth)
        self.right = DecisionTree(self.data[idx_right], self.target[idx_right], self.max_depth)
        self.left.fit(self.depth+1)
        self.right.fit(self.depth+1)
        
    def predict(self, data):
        #print(f'gini_min: {self.gini_min}, depth: {self.depth}')
        if self.gini_min == 0.0 or self.depth == self.max_depth:
            return self.label
        else:
            if data[self.feature] > self.threshold:
                return self.left.predict(data)
            else:
                return self.right.predict(data)
    
    def search_best_split(self):
        features = self.data.shape[1] #説明変数の数
        best_threshold = None
        best_feat = None
        min_gini =  1
        
        for feat_idx in range(features):
            values = self.data[:, feat_idx] #説明変数の選択
            for val in values:
                gini = self.gini_score(feat_idx, val)
                if min_gini > gini:
                    min_gini = gini
                    best_threshold = val
                    best_feat = feat_idx
        return min_gini, best_threshold, best_feat
    
    
    def gini_score(self, feat_idx, threshold):
        gini = 0
        sample_num = len(target)
        # div target left or right
        left_target = self.target[self.data[:, feat_idx] >= threshold]
        right_target = self.target[self.data[:, feat_idx] < threshold] 
        
        classes = np.unique(self.target)
        left_score = 0
        right_score = 0
        for cl in classes:
            left_score += (np.sum(left_target == cl)/len(left_target))**2 
            right_score += (np.sum(right_target == cl)/len(right_target))**2 
        gini = (1 - left_score) * (len(left_target)/sample_num) + (1- right_score) * (len(right_target)/sample_num)
            
        return gini
        
class DesicionTreeClassifier():
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.tree = None
   
    def fit(self, data, target):
        initial_depth = 0
        self.tree = DecisionTree(data, target, self.max_depth)
        self.tree.fit(initial_depth)
   
    def predict(self, data):
        pred = []
        for s in data:
            pred.append(self.tree.predict(s))
        return np.array(pred)