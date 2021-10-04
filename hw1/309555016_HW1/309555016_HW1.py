import numpy as np
import matplotlib.pyplot as plt

# 用來記錄每一輪的error
learn_curve = []
learn_curve_vild = []


def import_data(path):
    tmp = np.loadtxt(path, dtype=np.str, delimiter=",")
    data = tmp[1:, :].astype(np.float)
    return data


def MSE(A, A_vild, b, b_vild, x):

    # train data 的 error
    data_acount = np.size(A, 0)
    ax = np.dot(A, x)
    ax_b = ax - b
    ax_b = ax_b * ax_b
    ax_b = ax_b / data_acount
    total_error = ax_b.sum()
    learn_curve.append(total_error)
    print("=========================error : ", total_error.round(5))

    # test data 的 error
    data_acount = np.size(A_vild, 0)
    ax = np.dot(A_vild, x)
    ax_b = ax - b_vild
    ax_b = ax_b * ax_b
    ax_b = ax_b / data_acount
    total_error = ax_b.sum()

    learn_curve_vild.append(total_error)
    print("=========================error_vild : ", total_error.round(5))


def MAE(A, A_vild, b, b_vild, x):

    # train data 的 error
    total_error = np.average(np.abs((A @ x) - b))
    learn_curve.append(total_error)
    print("=========================error : ", total_error.round(5))

    # test data 的 error
    total_error = np.average(np.abs((A_vild @ x) - b_vild))
    learn_curve_vild.append(total_error)
    print("=========================error_vild : ", total_error.round(5))


def g_d(data, data_vild, learning_rate, error_type, x):
    print("training")

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


# 指數
def pow(num, exp):
    x = 1
    for i in range(0, exp):
        x = x * num
    return x


def train(data, data_vild, dim, iterate, learning_rate, error_type):

    # 起始值隨機
    x = np.random.rand(dim, 1)
    # 梯度下降執行 iteraion 次
    for i in range(iterate):
        x_new = g_d(data, data_vild, learning_rate, error_type, x)
        x = x_new

    return x


def draw_l_curve():
    # x=[0,1,2,3........,n]
    # y=每一輪的error
    x = np.arange(len(learn_curve))
    y = learn_curve
    plt.plot(x, y)

    x = np.arange(len(learn_curve_vild))
    y = learn_curve_vild
    plt.plot(x, y, c='red')

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Learning curve")
    plt.show()


if __name__ == "__main__":

    # 設定參數
    iterate = 1000
    learning_rate = 0.01
    dim = 2
    error_type = "MAE"
    # 匯入資料
    data = import_data("train_data.csv")
    data_vild = import_data("test_data.csv")
    # 開始training
    weight = train(data, data_vild, dim, iterate, learning_rate, error_type)

    # 結束
    print("============Done====================")
    print("X:", weight)

    # 畫圖
    draw_l_curve()
