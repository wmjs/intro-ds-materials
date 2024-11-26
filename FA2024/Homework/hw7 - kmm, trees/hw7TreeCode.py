import numpy as np

class TreeNode(object): # Do not modify this class
    '''
    A node class for a decision tree.
    '''
    def __init__(self, feature=None, value=None, left=None, right=None, label=None):
        self.feature = feature # feature to split on
        self.value = value # value used for splitting
        self.left = left # left child
        self.right = right # right child
        self.label = label # label for the node

class TreeBase(object): # Do not modify this class
    def __init__(self, data, label, max_depth=5):
        '''
        Constructor
        Parameters:
            data: DataFrame, the data for the tree
            label: str, the label of the target
            max_depth: int, the maximum depth of the tree
        '''
        self.data = data
        self.root = None
        self.max_depth = max_depth
        self.label = label
        self.features = data.columns.drop(label)

    def select_best_feature(self, data, features):
        '''
        Select the feature with the highest information gain
        Parameters:
            data: DataFrame
            features: list of features
        Returns:
            best_feature: str
            best_value: str
        '''
        best_gain = 0

        for feature in features: #Iterate over all features
            values = data[feature].unique() #Get all unique values of the feature
            for value in values: #Iterate over all values
                gain = self.information_gain(data, feature, value) #Calculate the information gain
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = value

        return best_feature, best_value

    def entropy(self, data):
        raise NotImplementedError

    def information_gain(self, data, feature, value):
        raise NotImplementedError

    def predict(self, data_point):
        '''
        Predict the label for a single instance
        Parameters:
            data_point: pandas Series, with keys being the feature names
        Returns:
            label: the label of the instance
        '''
        node = self.root
        label = None
        while True:
            if node.label is not None: #Leaf node
                label = node.label
                break
            if type(node.value) == str: #Categorical feature
                go_left = data_point[node.feature] == node.value
            else: #Numerical feature
                go_left = data_point[node.feature] < node.value

            if go_left:
                node = node.left
            else:
                node = node.right
        return label

    def fit(self):
        self.root = self.build_tree(TreeNode(), self.data, self.features, 0) #Build the tree

    def build_tree(self, node, data, features, depth):
        '''
        Recursively build the decision tree
        Parameters:
            node: TreeNode, current node to be split
            data: DataFrame, the data for the tree
            features: list of features
            depth: int, the current depth of the tree
        Returns:
            node: TreeNode, the root of the built tree
        '''

        #Stop if the entropy is 0, or the depth >= max_depth, or all data points has the same feature value.
        stop = np.isclose(self.entropy(data), 0) or \
                 depth >= self.max_depth or \
                 (data.values[0] == data.values).all()
        if stop:
            node = TreeNode()
            node.label = data[self.label].mode()[0] #Get the most common label, only set label for leaf nodes
            return node

        feature, value = self.select_best_feature(data, features)
        node = TreeNode(feature=feature, value=value)
        if type(value) == str: #Categorical feature
            left_data, right_data = data[data[feature] == value], data[data[feature] != value]
        else: #Numerical feature
            left_data, right_data = data[data[feature] < value], data[data[feature] >= value]

        node.left = self.build_tree(node.left, left_data, features, depth + 1)
        node.right = self.build_tree(node.right, right_data, features, depth + 1)
        return node

    def print_tree(self):
        '''
        Print the tree
        Parameters:
            node: TreeNode, the current node
            depth: int, the current depth of the tree
        '''
        self._print_tree(self.root, 0)

    def _print_tree(self, node, depth):
        '''
        Recursively print the tree
        Parameters:
            node: TreeNode, the current node
            depth: int, the current depth of the tree
        '''
        if node is None:
            return
        if node.label is not None:
            print("  " *  depth, depth, "-", node.label)
        else:
            if type(node.value) == str:
                print("  " * depth, depth, "-", node.feature, "=", node.value)
            else:
                print("  "  *  depth, "-", node.feature, "<", node.value)
            self._print_tree(node.left, depth + 1)
            self._print_tree(node.right, depth + 1)

class DecisionTree(TreeBase):
    '''
    Binary decision tree class, inherits from TreeBase
    '''
    def __init__(self, data, label, max_depth=5):
        super(DecisionTree, self).__init__(data, label, max_depth)

    def entropy(self, data):
        '''
        Calculate the entropy of the data
        Parameters:
            data: DataFrame
        Returns:
            the entropy of the data
        '''

        if len(data) == 0:
          return 0

        entropy = 0
        for label in data[self.label].unique():
          count = len(data[data[self.label] == label])
          prob = count / len(data)
          entropy -= prob * np.log2(prob)

        return entropy

    def information_gain(self, data, feature, value):
        '''
        Calculate the information gain
        Parameters:
            data: DataFrame
            feature: the feature to split
            vavlue: the value of the feature
        Returns:
            the information gain

        '''

        total_entropy = self.entropy(data)

        #From build_tree
        if type(value) == str: #Categorical feature
            left_data, right_data = data[data[feature] == value], data[data[feature] != value]
        else: #Numerical feature
            left_data, right_data = data[data[feature] < value], data[data[feature] >= value]

        left_entropy = self.entropy(left_data)
        right_entropy = self.entropy(right_data)

        left_weight = len(left_data) / len(data)
        right_weight = len(right_data) / len(data)

        quality = total_entropy - (left_weight * left_entropy + right_weight * right_entropy)

        return quality