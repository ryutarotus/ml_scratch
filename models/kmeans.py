import numpy as np

class KMeans:
    """
    init: kmeans or random is available
    """
    def __init__(self, cluster_num=4, max_iter=100, init='kmeans++'):
        self.cluster_num = cluster_num
        self.max_iter = max_iter
        self.centroids = None
        self.init = init
        
    def fit(self, X):
        data_num = X.shape[0]
        feature_num = X.shape[1]
        
        if self.init == 'random':
            indexs = np.arange(data_num)  # データのindex

            init_indexs = np.random.choice(indexs, self.cluster_num, replace=False)  # 初期値をデータからクラスター数分重複なしで抽出
            centroids = X[init_indexs, :]  # 重心の初期値
            
        elif self.init == 'kmeans++':
            distance = np.zeros((data_num, self.cluster_num))  # 重心の距離の初期化
            centroids = np.zeros((self.cluster_num, feature_num))  # 初期値の初期化
            
            #1つ目の重心をランダムに決める
            pr = np.repeat(1/data_num, data_num)
            centroids[0, :] = X[np.random.choice(np.arange(data_num), 1, p=pr), :]
            distance[:, 0] = np.sum((X-centroids[0, :])**2, axis=1)  # 一つ目の重心からの距離を計算
            
            #2つ目の重心は1つ目の重心からの距離によって確率を変更
            choiced = []
            for i in range(1, self.cluster_num):
                pr = np.sum(distance, axis=1) / np.sum(distance)
                #確率に従って2つ目の点を選ぶ
                choice = np.random.choice(np.arange(data_num), 1, p=pr)
                choiced.append(choice)
                centroids[i,:] = X[choice, :]
                distance[:,i] = np.sum((X-centroids[i, :])**2, axis=1)
                distance[choiced, :] = 0.0  # 一度選んだ重心は確率0になるようにする
                
        else:
            raise ValueError('init is must be kmeans++ or random')
        
        idx = np.zeros(data_num)  # どこのクラスターに入るかの空の配列
        
        for _ in range(self.max_iter):
            for i in range(data_num):
                idx[i] = np.argmin(np.sum((X[i, :] - centroids)**2, axis=1))  # それぞれの重心からの距離を計算し一番近いクラスターに割り当てる
            
            pre_centroids = centroids.copy()
            
            for k in range(self.cluster_num):  # 重心を更新
                centroids[k, :] = X[idx==k, :].mean(axis=0)
                
            if np.all(centroids == pre_centroids):  # 重心が更新されない場合
                break
                
        self.centroids = centroids
        
    def predict(self, X):
        data_num = X.shape[0]
        
        idx = np.zeros(data_num)  # どこのクラスターに入るかの空の配列
        
        for i in range(data_num):
            idx[i] = np.argmin(np.sum((X[i, :] - self.centroids)**2, axis=1))  # それぞれの重心からの距離を計算し一番近いクラスターに割り当てる
            
        return idx
    
    def fit_predict(self, X):
        data_num = X.shape[0]
        feature_num = X.shape[1]
        
        if self.init == 'random':
            indexs = np.arange(data_num)  # データのindex

            init_indexs = np.random.choice(indexs, self.cluster_num, replace=False)  # 初期値をデータからクラスター数分重複なしで抽出
            centroids = X[init_indexs, :]  # 重心の初期値
            
        elif self.init == 'kmeans++':
            distance = np.zeros((data_num, self.cluster_num))  # 重心の距離の初期化
            centroids = np.zeros((self.cluster_num, feature_num))  # 初期値の初期化
            
            #1つ目の重心をランダムに決める
            pr = np.repeat(1/data_num, data_num)
            centroids[0, :] = X[np.random.choice(np.arange(data_num), 1, p=pr), :]
            distance[:, 0] = np.sum((X-centroids[0, :])**2, axis=1)  # 一つ目の重心からの距離を計算
            
            #2つ目の重心は1つ目の重心からの距離によって確率を変更
            choiced = []
            for i in range(1, self.cluster_num):
                pr = np.sum(distance, axis=1) / np.sum(distance)
                #確率に従って2つ目の点を選ぶ
                choice = np.random.choice(np.arange(data_num), 1, p=pr)
                choiced.append(choice)
                centroids[i,:] = X[choice, :]
                distance[:,i] = np.sum((X-centroids[i, :])**2, axis=1)
                distance[choiced, :] = 0.0  # 一度選んだ重心は確率0になるようにする
                
        else:
            raise ValueError('init is must be kmeans++ or random')
        
        idx = np.zeros(data_num)  # どこのクラスターに入るかの空の配列
        
        for _ in range(self.max_iter):
            for i in range(data_num):
                idx[i] = np.argmin(np.sum((X[i, :] - centroids)**2, axis=1))  # それぞれの重心からの距離を計算し一番近いクラスターに割り当てる
            
            pre_centroids = centroids.copy()
            
            for k in range(self.cluster_num):  # 重心を更新
                centroids[k, :] = X[idx==k, :].mean(axis=0)
                
            if np.all(centroids == pre_centroids):  # 重心が更新されない場合
                break
                
        self.centroids = centroids
        return idx