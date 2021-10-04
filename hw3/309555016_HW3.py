

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data = load_breast_cancer()
feature_names = data['feature_names']
print(feature_names)


class RandomForest():
    def __init__(
            self,
            n_estimators,
            max_features,
            boostrap=True,
            criterion='gini',
            max_depth=None):
        self.forest = []
        self.tree_f_index = []
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.boostrap = boostrap
        self.n_estimators = n_estimators

        for i in range(n_estimators):
            tree = DecisionTree(criterion=self.criterion,
                                max_depth=self.max_depth)
            self.forest.append(tree)

    # 建立森林裡所有的樹
    def build_forest(self, x, y):

        data_count, feature_count = np.shape(x)
        tree_count = len(self.forest)

        for tree_index in range(tree_count):
            all_f = [i for i in range(feature_count)]

            # 抽樣feature
            rand_f = np.random.choice(all_f, int(
                self.max_features), replace=False)
            self.tree_f_index.append(rand_f)

            # 抽樣 使用的樣本,目前設定一棵樹使用2/3的資料去建立。
            if self.boostrap:
                all_data = [i for i in range(data_count)]
                rand_data = np.random.choice(
                    all_data, int(data_count * 0.66), replace=False)
                new_x = x[rand_data, :]
                new_x = new_x[:, rand_f]

                # 用抽樣的feature 與 樣本建立樹
                self.forest[tree_index].build_tree(new_x, y[rand_data, :])
            else:
                # 用抽樣的 feature 建立樹
                self.forest[tree_index].build_tree(x[:, rand_f], y)

    # 使用森林去做預測
    def forest_classify(self, x, y):
        data_count, feature_count = np.shape(x)
        tree_count = len(self.forest)
        all_vote = np.zeros((data_count, 2))

        # 用森林裡的每個樹做Classify，
        for tree_index in range(tree_count):
            index_f = self.tree_f_index[tree_index]
            new_x = x[:, index_f]

            vote = self.forest[tree_index].classfy(new_x, y, for_forest=True)

            for j in range(len(vote)):
                p = int(vote[j])
                all_vote[j][p] += 1
        # 將每棵樹的預測結果用投票的方式表決
        final_result = np.argmax(all_vote, axis=1)

        # 計算錯誤率
        error = 0
        for i in range(data_count):
            if y[i] != final_result[i]:
                error += 1

        error_rate = error / data_count
        print("==============result==============")
        print("forest_size:", self.n_estimators)
        print("type : ", self.criterion)
        print("max_feature : ", int(self.max_features))
        print("max_depth : ", self.max_depth)
        print("accuracy : ", 1 - error_rate)


class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        if criterion == 'gini':
            self.count_pur = gini
        else:
            self.count_pur = entropy
        self.max_depth = max_depth
        self.total_fi = None
        self.root = None
        self.important_count = np.zeros((len(feature_names)))
        return None

    class Node():
        def __init__(self):
            self.using_feature = None
            self.threshold = None
            self.impurity = None
            self.left_node = None
            self.right_node = None
            self.label = None

    # 用遞迴的方式將資料分群，回傳每個子樹的 root node
    def split(self, data, now_depth):
        this_node = self.Node()
        # 如果只剩一種資料，判定label
        # 如果已經到樹的最底層，判定label
        if self.count_pur(data[:, -1]) == 0:
            unique, each_count = np.unique(data[:, -1], return_counts=True)
            result = unique[0]
            this_node.label = result
        elif now_depth == 0:
            unique, each_count = np.unique(data[:, -1], return_counts=True)
            max_label = np.argmax(each_count)
            this_node.label = unique[max_label]

        # find threshold
        # 分群left_data,right_data
        else:
            data_count, feature_count = data.shape
            feature_count = feature_count - 1

            best_cut_threshold = 0
            best_cut_feature = 0
            best_cut_position = 0
            min_impurity = 100000000

            for f in range(feature_count):
                # 先將此feature根據大小順序排好
                after_sort = np.asarray(sorted(data, key=lambda t: t[f]))
                for i in range(1, data_count):
                    # treshold設為 data[i] 與 data[i-1] 的中間值
                    t = (after_sort[i][f] + after_sort[i - 1][f]) / 2
                    left_data = after_sort[:i, :]
                    right_data = after_sort[i:, :]

                    left_impurity = self.count_pur(left_data[:, -1])
                    right_impurity = self.count_pur(right_data[:, -1])

                    total_impurity = i * left_impurity + \
                        (data_count - i) * right_impurity
                    total_impurity /= data_count

                    if total_impurity <= min_impurity:
                        min_impurity = total_impurity
                        best_cut_feature = f
                        best_cut_threshold = t
                        best_cut_position = i

            this_node.using_feature = best_cut_feature
            this_node.impurity = min_impurity
            this_node.threshold = best_cut_threshold

            after_sort = np.asarray(
                sorted(data, key=lambda t: t[best_cut_feature]))
            left_data = after_sort[:best_cut_position, :]
            right_data = after_sort[best_cut_position:, :]

            if now_depth is None:
                this_node.right_node = self.split(right_data, None)
                this_node.left_node = self.split(left_data, None)
            else:
                this_node.right_node = self.split(right_data, now_depth - 1)
                this_node.left_node = self.split(left_data, now_depth - 1)

        return this_node
    # 建立一棵樹

    def build_tree(self, x, y):
        data = np.hstack((x, y))
        self.root = self.split(data, self.max_depth)
    # 用這棵樹進行分類

    def classfy(self, x, y, for_forest=False):

        hight = self.max_depth
        data_count, data_f = np.shape(x)
        error = 0
        predict_label = np.zeros((data_count))

        for i in range(data_count):
            r = self.root
            result = 10000
            # print("DATA",i)
            while True:
                f = r.using_feature
                if r.label is None:
                    if x[i][f] < r.threshold:
                        r = r.left_node
                    else:
                        r = r.right_node
                else:
                    result = r.label
                    break
            # 計算error
            if result != y[i]:
                error += 1
            predict_label[i] = result

        error_rate = error / data_count
        if for_forest is not True:
            print("==============result==============")
            print("type : ", self.criterion)
            print("max_depth : ", self.max_depth)
            print("accuracy : ", 1 - error_rate)
        return predict_label

    def get_node_feature(self, n):
        f = n.using_feature

        if n.label is None:
            self.important_count[f] += 1
            self.get_node_feature(n.left_node)
            self.get_node_feature(n.right_node)

    # 獲得important_feature，return每個feature分別被使用幾次
    def fit(self):
        self.important_count = np.zeros((len(feature_names)))
        n = self.root
        self.get_node_feature(n)

        print(self.important_count)
        return self.important_count


def gini(sequence):

    length = len(sequence)
    unique, each_count = np.unique(sequence, return_counts=True)
    p = each_count / length

    g = 0
    for i in range(len(p)):
        g += p[i]**2
    g = 1 - g
    return g


def entropy(sequence):
    length = len(sequence)
    unique, each_count = np.unique(sequence, return_counts=True)
    p = each_count / length

    e = 0
    for i in range(len(p)):
        e += p[i] * np.log2(p[i])
    e = -1 * e
    return e


if __name__ == '__main__':

    x_train = pd.read_csv("x_train.csv")
    y_train = pd.read_csv("y_train.csv")
    x_test = pd.read_csv("x_test.csv")
    y_test = pd.read_csv("y_test.csv")

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    # Question 1
    # Gini Index or Entropy is often used for measuring the “best” splitting
    # of the data. Please compute the Entropy and Gini Index of provided data.
    # Please use the formula from the course sludes on E3

    data = np.array([1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 2])
    print("Gini of data is ", gini(data))
    print("Entropy of data is ", entropy(data))

    print("\n\n\n\n")
    # ### Question 2.1
    # Using Criterion=‘gini’, showing the accuracy score of test data by
    # Max_depth=3 and Max_depth=10, respectively.

    clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
    clf_depth3.build_tree(x_train, y_train)
    error = clf_depth3.classfy(x_test, y_test)

    clf_depth10 = DecisionTree(criterion='gini', max_depth=10)
    clf_depth10.build_tree(x_train, y_train)
    error = clf_depth10.classfy(x_test, y_test)

    print("\n\n\n\n")

    # ### Question 2.2
    # Using Max_depth=3, showing the accuracy score of test data by
    # Criterion=‘gini’ and Criterion=’entropy’, respectively.

    clf_gini = DecisionTree(criterion='gini', max_depth=3)
    clf_gini.build_tree(x_train, y_train)
    error = clf_gini.classfy(x_test, y_test)

    clf_entropy = DecisionTree(criterion='entropy', max_depth=3)
    clf_entropy.build_tree(x_train, y_train)
    error = clf_entropy.classfy(x_test, y_test)

    print("\n\n\n\n")

    # ## Question 3
    # Plot the [feature importance]
    # (https://sefiks.com/2020/04/06/feature-importance-in-decision-trees/)
    # of your Decision Tree model. You can get the feature importance
    # by counting the feature used for splitting data.
    # - You can simply plot the feature counts for building tree without
    # normalize the importance
    # ![image](https://i2.wp.com/sefiks.com/wp-content/uploads/2020/04/c45-fi-results.jpg?w=481&ssl=1)

    important_feature = clf_depth10.fit()

    x = np.arange(len(feature_names))

    x_axis_num = np.arange(max(important_feature) + 1)

    plt.barh(x, important_feature)
    plt.ylabel('feature')
    plt.xlabel('importance')

    plt.xticks(x_axis_num)
    plt.yticks(x, feature_names)
    plt.tight_layout()
    # plt.savefig('fi.png', dpi=300, transparent=True)
    plt.show()

    print("\n\n\n\n")
    # ## Question 4
    # implement the Random Forest algorithm by using the CART you just
    # implemented from question 2. You should implement
    # three arguments for the Random Forest.
    # 1. **N_estimators**: The number of trees in the forest.
    # 2. **Max_features**: The number of random select features to
    #    consider when looking for the best split
    # 3. **Bootstrap**: Whether bootstrap samples are used when building tree
    # ### Question 4.1
    # Using Criterion=‘gini’, Max_depth=None, Max_features=sqrt(n_features),
    # showing the accuracy score of test data by n_estimators=10 and
    # n_estimators=100, respectively.

    clf_10tree = RandomForest(
        n_estimators=10, max_features=np.sqrt(x_train.shape[1]))

    clf_10tree.build_forest(x_train, y_train)
    clf_10tree.forest_classify(x_test, y_test)

    clf_100tree = RandomForest(
        n_estimators=100, max_features=np.sqrt(x_train.shape[1]))

    clf_100tree.build_forest(x_train, y_train)
    clf_100tree.forest_classify(x_test, y_test)

    # ### Question 4.2
    # Using Criterion=‘gini’, Max_depth=None, N_estimators=10,
    # showing the accuracy score of test data by Max_features=sqrt(n_features)
    # and Max_features=n_features, respectively.

    # In[13]:
    print("\n\n\n\n")

    clf_random_features = RandomForest(
        n_estimators=10, max_features=np.sqrt(x_train.shape[1]))

    clf_random_features.build_forest(x_train, y_train)
    clf_random_features.forest_classify(x_test, y_test)

    clf_all_features = RandomForest(
        n_estimators=10, max_features=x_train.shape[1])

    clf_all_features.build_forest(x_train, y_train)
    clf_all_features.forest_classify(x_test, y_test)
