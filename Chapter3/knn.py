# coding: utf_8


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

if __name__ == '__main__':
    kt = KdTree()
