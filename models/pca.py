import numpy as np

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components
        self.W = None
        
    def fit(self, X):
        # 分散共分散行列の作成
        cov_mat = np.cov(X.T)
        # 分散共分散行列の固有値、固有ベクトルを作成
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        # 固有値、固有ベクトルのペアを作成
        eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
        # 上記のペアを固有値 (寄与率) の大きい順にソート
        eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        
        # 寄与率の高い順に固有ベクトルをn_components分獲得
        ws = []
        for i in range(self.n_components):
            ws.append(eigen_pairs[i][1])
        # 射影行列の作成
        self.W = np.array(ws).T
        
    def transform(self, X):
        # 射影
        X_pca = X @ self.W
        return X_pca
    
    def fit_transform(self, X):
        # 分散共分散行列の作成
        cov_mat = np.cov(X.T)
        # 分散共分散行列の固有値、固有ベクトルを作成
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        # 固有値、固有ベクトルのペアを作成
        eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
        # 上記のペアを固有値の大きい順にソート
        eigen_pairs.sort(key=lambda k: k[0], reverse=True)
        
        # 寄与率の高い順に固有ベクトルをn_components分獲得
        ws = []
        for i in range(self.n_components):
            ws.append(eigen_pairs[i][1])
        # 射影行列の作成
        self.W = np.array(ws).T
        # 射影
        X_pca = X @ self.W
        return X_pca