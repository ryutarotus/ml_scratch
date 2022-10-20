import numpy as np

class DecisionTreeNode():
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
        if len(self.target) != 0:
            self.label = np.argmax(np.bincount(target))
        else:
            self.label = None
    
    def fit(self, depth):
        self.depth = depth
        self.gini_min, self.threshold, self.feature = self.search_best_split()
        #print(self.threshold)
        if self.depth == self.max_depth or self.gini_min == 0 or len(self.target) == 0:  # 停止条件
            return
        #print(type(self.data[:, self.feature]), type(self.threshold))
        idx_left =  self.data[:, self.feature] >= self.threshold
        idx_right =  self.data[:, self.feature] < self.threshold
        
        self.left = DecisionTreeNode(self.data[idx_left], self.target[idx_left], self.max_depth)
        self.right = DecisionTreeNode(self.data[idx_right], self.target[idx_right], self.max_depth)
        
        self.left.fit(self.depth+1)
        self.right.fit(self.depth+1)
        
        
    def predict(self, data):
        #print(f'gini_min: {self.gini_min}, depth: {self.depth}')
        if self.gini_min == 0.0 or self.depth == self.max_depth or len(self.target) == 0:
            """
            if self.label == None:
                print(self.label)
            """
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
        sample_num = len(self.target)
        # div target left or right
        left_target = self.target[self.data[:, feat_idx] >= threshold]
        right_target = self.target[self.data[:, feat_idx] < threshold] 
        
        classes = np.unique(self.target)
        left_score = 0
        right_score = 0
        for cl in classes:
            if len(left_target) != 0:
                left_score += (np.sum(left_target == cl)/len(left_target))**2 
            if len(right_target) != 0:
                right_score += (np.sum(right_target == cl)/len(right_target))**2 
        gini = (1 - left_score) * (len(left_target)/sample_num) + (1- right_score) * (len(right_target)/sample_num)
            
        return gini

class DecisionTree():
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.tree = None
   
    def fit(self, data, target):
        initial_depth = 0
        self.tree = DecisionTreeNode(data, target, self.max_depth)
        self.tree.fit(initial_depth)
   
    def predict(self, data):
        pred = []
        for s in data:
            pred.append(self.tree.predict(s))
        return np.array(pred)
    
class RandomForest():
    """
    label の予測データ が None だった場合は random_state を変更してください。
    """
    
    def __init__(self, n_tree=10, max_depth=3):
        self.n_tree = n_tree
        self.forest = [None] * self.n_tree
        self._used_features_set = [None] * self.n_tree
        self.max_depth = max_depth
        
    def _bootstrap_sample(self, X, y, random_state=None):
        n_features = X.shape[1]
        n_features_forest = int(np.floor(np.sqrt(n_features)))  # それぞれ決定木で特徴量を何個使うか
        bootstrapped_X = []  # subset
        bootstrapped_y = []
        
        np.random.seed(random_state)

        for i in range(self.n_tree):
            idx = np.random.choice(len(y), size=len(y))
            col_idx = np.random.choice(n_features, size=n_features_forest, replace=False)
            bootstrapped_X.append(X[idx][:, col_idx])
            bootstrapped_y.append(y[idx])
            self._used_features_set[i] = col_idx
        return bootstrapped_X, bootstrapped_y

    def fit(self, X, y, random_state=None):
        self._classes = np.unique(y) #一意な要素を抽出
        sampled_X, sampled_y = self._bootstrap_sample(X, y, random_state=random_state)
        for i, (sampled_Xi, sampled_yi) in enumerate(zip(sampled_X, sampled_y)):
            tree = DecisionTree(self.max_depth)
            tree.fit(sampled_Xi, sampled_yi)
            self.forest[i] = tree
            
    def predict(self, X):
        prob = self.predict_prob(X)
        return self._classes[np.argmax(prob, axis=1)]

    def predict_prob(self, X): 
        if self.forest[0] is None:
            raise ValueError('Model not fitted yet')

        # 決定木群による投票
        votes = []
        for i, (tree, used_features) in enumerate(zip(self.forest, self._used_features_set)):
            votes.append(tree.predict(X[:, used_features]))
        votes_array = np.array(votes)
        #print(votes_array.shape)
        # 投票結果を集計
        counts_array = np.zeros((len(X), len(self._classes)), dtype=int)
        for c in self._classes:
            counts_array[:, c] = np.sum(np.where(votes_array==c, 1, 0), axis=0)

        # 予測クラス毎に割合を計算し、probとして返す
        prob = counts_array / self.n_tree
        return prob