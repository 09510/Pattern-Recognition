
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')

    # ## 1. Compute the mean vectors mi, (i=1,2) of each 2 classes

    x0 = []
    x1 = []
    n = len(x_train)

    for i in range(len(x_train)):
        if y_train[i] == 0:
            x0.append(x_train[i])
        elif y_train[i] == 1:
            x1.append(x_train[i])

    x0_mean = 0
    x1_mean = 0
    for i in range(len(x0)):
        x0_mean += x0[i]
    for i in range(len(x1)):
        x1_mean += x1[i]

    x0_mean = x0_mean / len(x0)
    x1_mean = x1_mean / len(x1)

    print("mean vector of class 1:", x0_mean,
          " mean vector of class 2:", x1_mean)

    x0_mean = np.reshape(x0_mean, (2, 1))
    x1_mean = np.reshape(x1_mean, (2, 1))

    x0 = np.array(x0)
    x1 = np.array(x1)
    x0 = np.reshape(x0, (len(x0), 2, 1))
    x1 = np.reshape(x1, (len(x1), 2, 1))

    x0_within = np.zeros((2, 2))
    x1_within = np.zeros((2, 2))

    # ## 2. Compute the Within-class scatter matrix SW
    for i in range(len(x0)):
        x0_within += (x0[i] - x0_mean) @ (x0[i] - x0_mean).T
    for i in range(len(x1)):
        x1_within += (x1[i] - x1_mean) @ (x1[i] - x1_mean).T
    within = x0_within + x1_within

    print("Within-class scatter matrix SW:\n", within, "\n")
    # ## 3.  Compute the Between-class scatter matrix SB

    between = (x0_mean - x1_mean) @ (x0_mean - x1_mean).T

    print("Between-class scatter matrix SB:\n", between, "\n")
    # ## 4. Compute the Fisher’s linear discriminant

    w = np.linalg.inv(within) @ (x1_mean - x0_mean)
    print("Fisher's linear discriminant:\n", w, "\n")
    # ## 5. Project the test data by linear discriminant
    #  to get the class prediction by nearest-neighbor
    # rule and calculate the accuracy score you can use
    # accuracy_score function from sklearn.metric.accuracy_score

    intercept = 0
    m = w[0] / w[1]
    m = (1) / m
    x = np.linspace(-4, 4, 50)
    y = m * x + intercept

    all_project_point = []

    for i in range(len(x_train)):

        t = (m * x_train[i][0] + (-1 * x_train[i][1]) + intercept) / (m**2 + 1)
        x_point = x_train[i][0] - m * t
        y_point = x_train[i][1] - (-1 * t)
        all_project_point.append((x_point[0], y_point[0], y_train[i]))

    all_project_point.sort(key=lambda s: s[0])

    error = 0
    for i in range(len(x_test)):

        t = (m * x_test[i][0] + (-1 * x_test[i][1]) + intercept) / (m**2 + 1)
        x_point = x_test[i][0] - m * t
        y_point = x_test[i][1] - (-1 * t)

        distance = 100000
        for j in range(len(all_project_point)):
            this_dis = abs(all_project_point[j][0] - x_point)
            if this_dis > distance:
                pred = all_project_point[j - 1][2]
                if pred != y_test[i]:
                    error += 1
                    # print("wrong")
                break
            distance = this_dis

    print("Accuracy of test-set", 1 - error / len(y_test))
    # ## 6. Plot the 1) best projection line on the
    # training data and show the slope and
    # intercept on the title (you can choose any value
    # of intercept for better visualization)
    # 2) colorize the data with each class 3) project
    # all data points on your projection line.
    # Your result should look like
    intercept = 3

    for i in range(len(x_train)):
        if y_train[i] == 0:
            plt.scatter(x_train[i][0], x_train[i][1], c='r')
        else:
            plt.scatter(x_train[i][0], x_train[i][1], c='b')

    m = w[0] / w[1]
    m = (1) / m
    x = np.linspace(-4, 8, 500)
    y = m * x + intercept
    plt.plot(x, y, color='cyan')

    for i in range(len(x_train)):
        if y_train[i] == 1:
            t = (m * x_train[i][0] + (-1 * x_train[i][1]) +
                 intercept) / (m**2 + 1)

            x_point = x_train[i][0] - m * t
            y_point = x_train[i][1] - (-1 * t)
            plt.scatter(x_point, y_point, c='b')

            plt.plot([x_point, x_train[i][0]], [
                     y_point, x_train[i][1]], c='cyan', alpha=0.2)
        if y_train[i] == 0:
            t = (m * x_train[i][0] + (-1 * x_train[i][1]) +
                 intercept) / (m**2 + 1)

            x_point = x_train[i][0] - m * t
            y_point = x_train[i][1] - (-1 * t)
            plt.scatter(x_point, y_point, c='r')

            plt.plot([x_point, x_train[i][0]], [
                     y_point, x_train[i][1]], c='cyan', alpha=0.2)

    title = "Project Line: w=" + str(m[0]) + " b=" + str(intercept)
    plt.title(title)
    plt.axis('equal')

    plt.show()
