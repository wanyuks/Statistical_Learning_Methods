# coding: utf-8
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from collections import Counter


class OriginalPerceptron(object):
    def __init__(self):
        self.w = 0
        self.b = 0

    def fit(self, X, y, lr=0.01, iterations=500):
        # 初始化w和b
        self.w = np.random.rand(X.shape[1])
        self.b = 0
        X = np.array(X)
        for iteration in range(iterations):
            for i in range(len(X)):
                sample = X[i].T
                label = y[i]
                y_ = np.dot(self.w, sample) + self.b
                if label * y_ <= 0:
                    self.w = self.w + lr * label * sample
                    self.b = self.b + lr * label
            print('iteration', iteration)
        return self

    def test(self, X, y):
        right_count = 0
        for i in range(len(X)):
            y_ = np.dot(self.w, X[i].T) + self.b
            if y_ * y[i] > 0:
                right_count += 1
        return right_count


def get_data():
    '''
    构造一个用于二分类的数据
    '''
    data = load_iris()
    X = data['data'][:100]
    y = data['target'][:100]
    y = np.array([1 if i == 1 else -1 for i in y])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    op = OriginalPerceptron()
    X_train, X_test, y_train, y_test = get_data()
    op.fit(X_train, y_train)
    right_count = op.test(X_test, y_test)
    print(op.w, op.b)
    print(right_count)
