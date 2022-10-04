import random
def train_test_split(X, y, test_size=0.2, shuffle=True, random_state=None):
    assert len(X) == len(y), 'X and y must have the same length'
    idx = np.arange(len(X))
    np.random.seed(seed=random_state)
    np.random.shuffle(idx)
    train_idx = idx[: int(len(idx)*(1-test_size))]
    test_idx = idx[int(len(idx)*(1-test_size)): ]
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    return X_train, X_test, y_train, y_test