from collections import Counter
import math
import numpy as np
import pandas as pd
from math import log


class InformationGain(object):
    def __init__(self):
        pass

    # 计算经验熵
    @staticmethod
    def empirical_entropy(y):
        count_y = dict(Counter(y))
        entropy = 0
        for k,v in count_y.items():
            entropy -= (v / len(y)) * math.log2(v / len(y))
        return entropy

    # 计算经验条件熵
    def empirical_conditional_entropy(self, X, y):
        features = X.shape[1]
        conditional_entropy = {}
        for feature in range(features):
            feature_entropy = 0
            X_feature = X[:, feature]
            y_features = {}
            for i in range(len(X_feature)):
                if X_feature[i] not in y_features:
                    y_features[X_feature[i]] = [y[i]]
                else:
                    y_features[X_feature[i]].append(y[i])
            for y_feature in y_features.values():
                entropy = 0
                entropy += self.empirical_entropy(y_feature)
                entropy *= len(y_feature) / len(y)
                feature_entropy += entropy
            if feature not in conditional_entropy:
                conditional_entropy[feature] = feature_entropy
        return conditional_entropy

    # 计算信息增益
    def information_gain(self, X, y, feature):
        entropy = self.empirical_entropy(y)
        conditional_entropy = self.empirical_conditional_entropy(X, y)
        gain_entropy = entropy-conditional_entropy[feature]
        return gain_entropy


# 定义树的结点类
class Node(object):
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {
            'label:': self.label,
            'feature': self.feature,
            'tree': self.tree
        }

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, value, node):
        self.tree[value] = node

    def predict(self, features):
        if self.root is None:
            return self.label
        return self.tree[features[self.feature]].predict(features)


class DecisionTree(object):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    '''
    下面一部分代码就是在计算经验熵和条件熵, 和上面的有一些调整，主要是数据的输入采用DataFrame格式输入
    '''

    # 熵
    @staticmethod
    def calc_ent(datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p / data_length) * log(p / data_length, 2)
                    for p in label_count.values()])
        return ent

    # 经验条件熵
    def cond_ent(self, datasets, axis=0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(p) / data_length) * self.calc_ent(p)
                        for p in feature_sets.values()])
        return cond_ent

        # 信息增益

    @staticmethod
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    def info_gain_train(self, datasets):
        count = len(datasets[0]) - 1
        ent = self.calc_ent(datasets)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(datasets, axis=c))
            best_feature.append((c, c_info_gain))
        # 比较大小
        best_ = max(best_feature, key=lambda x: x[-1])
        return best_

    def train(self, train_data):
        """
        input:数据集D(DataFrame格式)，特征集A，阈值eta
        output:决策树T
        """
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[: -1]
        # 1,若D中实例属于同一类Ck，则T为单节点树，并将类Ck作为结点的类标记，返回T
        if len(y_train.value_counts()) == 1:
            return Node(root=True, label=y_train.iloc[0])

        # 2, 若A为空，则T为单节点树，将D中实例树最大的类Ck作为该节点的类标记，返回T
        if len(features) == 0:
            return Node(
                root=True,
                label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 3,计算最大信息增益 同5.1,Ag为信息增益最大的特征
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]

        # 4,Ag的信息增益小于阈值eta,则置T为单节点树，并将D中是实例数最大的类Ck作为该节点的类标记，返回T
        if max_info_gain < self.epsilon:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # 5,构建Ag子集
        node_tree = Node(
            root=False, feature_name=max_feature_name, feature=max_feature)

        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)

            # 6, 递归生成树
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)

        # pprint.pprint(node_tree.tree)
        return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)

if __name__ == '__main__':
    ig = InformationGain()
    datasets = np.array([['青年', '否', '否', '一般', '否'],
                ['青年', '否', '否', '好', '否'],
                ['青年', '是', '否', '好', '是'],
                ['青年', '是', '是', '一般', '是'],
                ['青年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '一般', '否'],
                ['中年', '否', '否', '好', '否'],
                ['中年', '是', '是', '好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['中年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '非常好', '是'],
                ['老年', '否', '是', '好', '是'],
                ['老年', '是', '否', '好', '是'],
                ['老年', '是', '否', '非常好', '是'],
                ['老年', '否', '否', '一般', '否'],
                ])

    data_df = pd.DataFrame(datasets, columns=['年龄', '有工作', '有自己的房子', '信贷情况', '类别'])
    dt = DecisionTree()
    tree = dt.fit(data_df)
    print(tree)
