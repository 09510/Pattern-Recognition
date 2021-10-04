import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# ## Load data
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
y_test = np.load("y_test.npy")

# 550 data with 300 features
print(y_train.shape)

# It's a binary classification problem
print(np.unique(y_train))


# for HW1
learn_curve = []
learn_curve_vild = []
# for HW1


def import_data(path):
    tmp = np.loadtxt(path, dtype=np.str, delimiter=",")
    data = tmp[1:, :].astype(np.float)
    return data
# for HW1


def MSE(A, A_vild, b, b_vild, x):
    # train data 的 error
    data_acount = np.size(A, 0)
    ax = np.dot(A, x)
    ax_b = ax - b
    ax_b = ax_b * ax_b
    ax_b = ax_b / data_acount
    total_error = ax_b.sum()
    learn_curve.append(total_error)
    # test data 的 error
    data_acount = np.size(A_vild, 0)
    ax = np.dot(A_vild, x)
    ax_b = ax - b_vild
    ax_b = ax_b * ax_b
    ax_b = ax_b / data_acount
    total_error = ax_b.sum()
    learn_curve_vild.append(total_error)
# for HW1


def MAE(A, A_vild, b, b_vild, x):
    total_error = np.average(np.abs((A @ x) - b))
    learn_curve.append(total_error)
    total_error = np.average(np.abs((A_vild @ x) - b_vild))
    learn_curve_vild.append(total_error)
# for HW1


def g_d(data, data_vild, learning_rate, error_type, x):
    # 設定初始值
    data_acount = np.size(data, 0)
    vild_acount = np.size(data_vild, 0)
    w_amount = np.size(data, 1)
    n = 2
    A = np.ndarray([data_acount, n])
    A_vild = np.ndarray([vild_acount, n])
    b = np.ndarray([data_acount, 1])
    b_vild = np.ndarray([vild_acount, 1])
    for j in range(n):
        for i in range(data_acount):
            A[i, j] = pow(data[i, 0], (n - j - 1))
    for j in range(n):
        for i in range(vild_acount):
            A_vild[i, j] = pow(data_vild[i, 0], (n - j - 1))
    for i in range(data_acount):
        b[i, 0] = (data[i, 1])
    for i in range(vild_acount):
        b_vild[i, 0] = (data_vild[i, 1])
    # 判斷使用MAE或是MSE
    if error_type == "MSE":
        # gradent=2(A.TA - Atx) * LR / N
        x_0 = x
        f_one = np.dot(A.T, A)
        f_one = f_one * 2
        f_one = np.dot(f_one, x_0)
        aa = np.dot(A.T, b)
        aa = aa * 2
        f_one = f_one - aa
        f_one = f_one / data_acount
        f_one = f_one * learning_rate
        # 梯度下降  x1 = x0 - gradient * LR
        x_1 = x_0 - f_one
        # 計算\紀錄 error
        MSE(A, A_vild, b, b_vild, x_1)
        return x_1
    elif error_type == "MAE":
        x_0 = x
        A_gd = np.zeros((1, w_amount), dtype=np.float)
        # 計算本次梯度的loss function 以及 他的gradient
        for row in range(data_acount):
            now_data = A[row, :]
            ax = np.dot(now_data, x_0)
            ax = ax - b[row]
            if ax > 0:
                A_gd = A_gd + now_data
            else:
                A_gd = A_gd - now_data
        c = A_gd.T * learning_rate / data_acount
        # 梯度下降  x1 = x0 - gradient * LR
        x_1 = x_0 - c
        # 計算\紀錄 error
        MAE(A, A_vild, b, b_vild, x_1)
        return x_1
# for HW1


def pow(num, exp):
    x = 1
    for i in range(0, exp):
        x = x * num
    return x


def draw_image(acc, c, g):

    # x軸為gamma , y軸為C
    x = g
    y = c

    x_len = len(x)
    y_len = len(y)

    plt.imshow(acc, interpolation='nearest', cmap=plt.cm.hot)
    plt.ylabel('C parameter')
    plt.xlabel('Gamma Parameter')
    plt.title('Hyperparameter Gridsearch')

    plt.colorbar()
    plt.xticks(np.arange(x_len), x)
    plt.yticks(np.arange(y_len), y)
    for i in range(x_len):
        for j in range(y_len):
            plt.text(
                i,
                j,
                '%.3f' %
                acc[j][i],
                ha='center',
                va='center',
                color='blue')
    plt.show()


def train(data, data_vild, dim, iterate, learning_rate, error_type):
    # 起始值隨機
    x = np.random.rand(dim, 1)
    # 梯度下降執行 iteraion 次
    for i in range(iterate):
        x_new = g_d(data, data_vild, learning_rate, error_type, x)
        x = x_new

    return x


def cross_validation(x_train, y_train, k=5):
    data_num, data_dim = np.shape(x_train)

    # 每個folder有幾筆data
    how_many_data_each_folder = data_num // k

    # 前 data_num % k 個folder,data數量多一
    how_many_folder_plus_one = data_num % k

    # 洗牌
    data_index = [i for i in range(data_num)]
    np.random.shuffle(data_index)

    # 切成 k folder
    each_folder = []

    # 決定哪個index屬於哪個folder
    now_index = 0
    for i in range(k):
        # 多一筆資料
        if i < how_many_folder_plus_one:
            destination_index = now_index + how_many_data_each_folder + 1
            this_folder_index = data_index[now_index:destination_index]
            this_folder_index = list(this_folder_index)
            each_folder.append(this_folder_index)

            now_index = destination_index
        else:
            destination_index = now_index + how_many_data_each_folder
            this_folder_index = data_index[now_index:destination_index]
            this_folder_index = list(this_folder_index)
            each_folder.append(this_folder_index)

            now_index = destination_index

    each_dataset = []

    # 整理成題目規定的形式
    for i in range(k):
        this_dataset = []
        train_set = []
        test_set = []
        for j in range(k):
            if j != i:
                train_set = train_set + each_folder[j]
            else:
                test_set = each_folder[j]
        train_set = np.array(train_set)
        test_set = np.array(test_set)

        # 排序,方便report觀看
        train_set = np.sort(train_set)
        test_set = np.sort(test_set)

        this_dataset.append(train_set)
        this_dataset.append(test_set)
        each_dataset.append(this_dataset)

    return each_dataset


def training(x, y, c, g, model_type="SVC"):

    if model_type == "SVC":
        clf = SVC(C=c, kernel='rbf', gamma=g)
        clf.fit(x, y)
        return clf
    elif model_type == "SVR":
        clf = SVR(C=c, kernel='rbf', gamma=g)
        clf.fit(x, y)
        return clf


def use_model(clf, x, y=None, model_type="SVC"):
    result = clf.predict(x)

    try:
        num = len(y)
    except BaseException:
        print("predict")
        return result

    if model_type == "SVC":
        error = 0
        for i in range(num):
            if y[i] != result[i]:
                error += 1
        accuracy = 1 - error / num
        return accuracy
    # SVR, square error
    else:
        square_error = 0
        for i in range(num):
            square_error += ((y[i] - result[i]) ** 2)
        square_error /= num
        return square_error


if __name__ == "__main__":
    # ## Question 1
    # K-fold data partition: Implement the K-fold cross-validation function.
    # Your function should take K as an argument and return a list of lists
    # (len(list) should equal to K), which contains K elements. Each element
    # is a list contains two parts, the first part contains the index of all
    # training folds, e.g. Fold 2 to Fold 5 in split 1. The second part
    # contains the index of validation fold, e.g. Fold 1 in  split 1

    kfold_data = cross_validation(x_train, y_train, k=10)
    assert len(kfold_data) == 10  # should contain 10 fold of data
    # each element should contain train fold and validation fold
    assert len(kfold_data[0]) == 2
    # The number of data in each validation fold should equal to training data
    # divieded by K
    assert kfold_data[0][1].shape[0] == 55

    k_folder = 5
    kfold_data = cross_validation(x_train, y_train, k=k_folder)

    print(kfold_data)
    # ## Question 2
    # Using sklearn.svm.SVC to train a classifier on the provided train set
    # and conduct the grid search of “C”, “kernel” and “gamma” to find the
    # best parameters by cross-validation.

    g = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    c = [0.01, 0.01, 0.1, 1, 10, 100, 10000]

    acc = np.zeros((len(c), len(g)))
    best_result = 0
    best_c_idx = 0
    best_g_idx = 0
    for i in range(len(c)):
        for j in range(len(g)):
            this_acc = 0
            for k in range(len(kfold_data)):
                # 把本folder中的test/train data 分別代入train
                x = x_train[kfold_data[k][0], :]
                y = y_train[kfold_data[k][0]]
                x_validation = x_train[kfold_data[k][1], :]
                y_validation = y_train[kfold_data[k][1]]

                clf = training(x=x, y=y, c=c[i], g=g[j])

                # 使用 validation data 來紀錄此 model 的好壞
                this_acc += use_model(x=x_validation, clf=clf, y=y_validation)

            # 平均
            this_acc = this_acc / len(kfold_data)

            # 尋找最佳解, 並且記錄 c 與 g
            if this_acc > best_result:
                best_result = this_acc
                best_c_idx = i
                best_g_idx = j
            acc[i][j] = this_acc

    best_c = c[best_c_idx]
    best_g = g[best_g_idx]
    print(best_c, best_g)

    # ## Question 3
    # Plot the grid search results of your SVM. The x, y represents the
    # hyperparameters of “gamma” and “C”, respectively. And the color
    # represents the average score of validation folds
    # You reults should be look like the reference image
    # ![image](https://miro.medium.com/max/1296/1*wGWTup9r4cVytB5MOnsjdQ.png)

    # 畫圖
    draw_image(acc, c, g)

    # ## Question 4
    # Train your SVM model by the best parameters you found from question 2 on
    # the whole training set and evaluate the performance on the test set.
    # **You accuracy should over 0.85**

    # 用全部的 train data 進去 train
    best_model = training(x=x_train, y=y_train, c=best_c, g=best_g)
    y_pred = best_model.predict(x_test)

    num = len(y_test)
    error = 0
    for i in range(num):
        if y_pred[i] != y_test[i]:
            error += 1

    accuracy = 1 - (error / num)
    print("Accuracy score: ", accuracy)

    # ## Question 5
    # Compare the performance of each model you have implemented from HW1

    # ### HW1
    train_df = pd.read_csv("./train_data.csv")
    x_train = train_df['x_train'].to_numpy().reshape(-1, 1)
    y_train = train_df['y_train'].to_numpy().reshape(-1, 1)
    y_train = np.ravel(y_train)

    test_df = pd.read_csv("./test_data.csv")
    x_test = test_df['x_test'].to_numpy().reshape(-1, 1)
    y_test = test_df['y_test'].to_numpy().reshape(-1, 1)
    y_test = np.ravel(y_test)

    c = [0.1, 1, 100, 10000, 1000000]
    g = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    k_folder = 5
    kfold_data = cross_validation(x_train, y_train, k=k_folder)

    acc = np.zeros((len(c), len(g)))
    best_result = 10000000
    best_c_idx = 0
    best_g_idx = 0

    for i in range(len(c)):
        for j in range(len(g)):
            this_acc = 0
            for k in range(len(kfold_data)):
                x = x_train[kfold_data[k][0], :]
                y = y_train[kfold_data[k][0]]
                x_validation = x_train[kfold_data[k][1], :]
                y_validation = y_train[kfold_data[k][1]]

                clf = training(x=x, y=y, c=c[i], g=g[j], model_type="SVR")
                this_acc += use_model(x=x_validation, clf=clf,
                                      y=y_validation, model_type="SVR")
            this_acc /= len(kfold_data)
            print(this_acc)

            if this_acc < best_result:
                best_result = this_acc
                best_c_idx = i
                best_g_idx = j

            acc[i][j] = this_acc

    best_c = c[best_c_idx]
    best_g = g[best_g_idx]

    print(best_c, best_g)
    draw_image(acc, c, g)

    best_model = training(
        x=x_train,
        y=y_train,
        c=best_c,
        g=best_g,
        model_type="SVR")
    y_pred = best_model.predict(x_test)

    num = len(y_test)
    square_error = 0

    for i in range(num):
        square_error += ((y_test[i] - y_pred[i]) ** 2)
    SVM_mean_square_error = square_error / num

    # 下面是HW1的計算
    iterate = 1000
    learning_rate = 0.01
    dim = 2
    error_type = "MSE"
    data = import_data("train_data.csv")
    data_vild = import_data("test_data.csv")
    weight = train(data, data_vild, dim, iterate, learning_rate, error_type)
    data_num = len(data_vild)
    A = np.zeros((data_num, 2))
    for j in range(data_num):
        A[j][0] = pow(x_test[j], 1)
    for j in range(data_num):
        A[j][1] = pow(x_test[j], 0)
    result = A @ weight
    square_error = 0
    for i in range(data_num):
        square_error += (y_test[i] - result[i])**2
    square_error /= data_num

    print("Square error of SVM regresssion model: ", SVM_mean_square_error)
    print("Square error of Linear regression: ", square_error)
