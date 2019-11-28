# coding: utf_8
import numpy as np
from collections import Counter

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


class KdNode(object):
    def __init__(self, ele, left, right):
        '''
        :param dom_elt:  K维向量节点（K维空间中的一个样本点）
        :param left: 分割超平面产生的左子空间构成的kd树
        :param right:分割超平面产生的右子空间构成的kd树
        '''
        self.ele = ele
        self.left = left
        self.right = right


class KdTree(object):
    def __init__(self, data):
        self.data = data
        self.k = len(data[0])

    def create_node(self, data_set, split=0):
        if data_set == None:
            return None
        else:
            data_set = sorted(data_set, key=lambda x: x[split])  #将数据按split位置排序
            split_pos = len(data_set) // 2  # 分割点
            data_split = data_set[split_pos]  # 得到分割点
            return KdNode(
                data_split,
                self.create_node(data_set[:split_pos], split+1),
                self.create_node(data_set[split_pos+1:], split+1)
            )

    # kd树的前序遍历
    def preorder(self, root):
        print(root.ele)
        if root.left:
            print(root.left)
        if root.right:
            print(root.right)


# python实现knn算法
class KNN(object):
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def _distance(self, v1, v2, distance_type='euclidean'):
        if distance_type == 'euclidean':
            distance = (np.sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))])) ** 0.5
            return distance
        if distance_type == 'manhattan':
            distance = np.sum([abs(v1[i] - v2[i]) for i in range(len(v1))])
            return distance
        else:
            raise TypeError('distance_type must be euclidean or manhattan')

    def predict(self, x):
        y_pred = []
        for i in range(len(x)):
            dist_arr = [self._distance(x[i], self.X[j]) for j in range(len(self.X))]
            # 获取距离排序后的索引
            '''
            np.argsort():
            l = np.array([1,4,3,-1,6,9])
            l_index_sort = np.argsort(l): array([3, 0, 2, 1, 4, 5], dtype=int64)
            l_index_sort[0]=3表示l[3]在l中最小
            '''
            sort_index = np.argsort(dist_arr)
            # 获取top_k的索引
            top_k = sort_index[:self.k]
            # 获取最相近k个实例的类别
            label_k = self.y[top_k]
            # 投票决定类别
            label_k_count = Counter(label_k)
            label_k_count_most = sorted(label_k_count.items(), key=lambda x:x[1], reverse=True)[0][0]
            y_pred.append(label_k_count_most)
        return np.array(y_pred)

    def score(self, y_pred=None, y_true=None):
        if y_pred is None or y_true is None:
            y_pred = self.predict(self.X)
            y_true = self.y
        count = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                count += 1
        score = count / len(y_pred)
        return round(score, 3)

if __name__ == '__main__':
    # 加载数据
    data = load_iris()
    X = data['data']
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=33)
    # 对数据进行标准化处理
    X_train = scale(X_train)
    X_test = scale(X_test)
    clf = KNN()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.score(y_pred, y_test))
