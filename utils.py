import numpy as np
import matplotlib.pyplot as plt


# 一次方程式
def line(x):
    return 1.0e-2 * x


# 二次関数
def quadratic(x):
    return x * x


# 双曲線
def hyperbola(x):
    return np.sqrt(x)


# expd
def my_exp(x):
    return np.exp(x ** 50 - 1) - np.exp(-1)


# プロット用関数
def plot_feature(features, c_nodes):
    # labels = ["No. "+str(num) for num in range(1, len(features[0]) + 1)]
    fig = plt.figure(figsize=(10, 10))

    for it, feature in enumerate(features):
        sub_plot = fig.add_subplot(4, 5, it + 1)
        dif = [0.5 for i in range(c_nodes[0])]
        # sub_plot.set_title("t = "+str(it))
        sub_plot.scatter(feature[: c_nodes[0], 0], feature[:c_nodes[0], 1],
                         color="blue")
        sub_plot.scatter(feature[c_nodes[0]:, 0], feature[c_nodes[0]:, 1],
                         color="red", marker='x')
        sub_plot.set_xlim(-0.15, 0.15)
        sub_plot.set_ylim(-0.15, 0.15)
        # sub_plot.set_aspect("equal")

        # for i, label in enumerate(labels):
        #     sub_plot.text(feature[:, 0][i], feature[:, 1][i], label)

    plt.show()


# xのプロット
def plot_x(features):
    xs = []
    for it, feature in enumerate(features):
        xs.extend(feature[:, 0])

    times = []
    for i in range(len(features)):
        for j in range(len(features[0])):
            times.append(i)

    plt.plot(times, xs)
    plt.show()


# yのプロット
def plot_y(features):
    ys = []
    for it, feature in enumerate(features):
        ys.extend(feature[:, 1])

    times = []
    for i in range(len(features)):
        for j in range(len(features[0])):
            times.append(i)

    plt.plot(times, ys)
    plt.show()


# xyを同じfigにプロット
def plot_xy(features, c_nodes):
    xs_1 = []
    ys_1 = []
    xs_2 = []
    ys_2 = []

    for feature in features:
        ys_1.extend(feature[:c_nodes[0], 1])

    for feature in features:
        xs_1.extend(feature[:c_nodes[0], 0])

    for feature in features:
        ys_2.extend(feature[c_nodes[0]:, 1])

    for feature in features:
        xs_2.extend(feature[c_nodes[0]:, 0])
    times_1 = []
    times_2 = []
    for i in range(len(features)):
        for j in range(c_nodes[0]):
            times_1.append(i)

    for i in range(len(features)):
        for j in range(c_nodes[1]):
            times_2.append(i)

    fig = plt.figure()
    sub_plot_x = fig.add_subplot(1, 2, 1)
    sub_plot_y = fig.add_subplot(1, 2, 2)
    sub_plot_x.set_title("x")
    sub_plot_y.set_title("y")
    sub_plot_x.set_ylim(-1, 1)
    sub_plot_y.set_ylim(-1, 1)
    sub_plot_x.plot(times_1, xs_1, c="blue")
    sub_plot_y.plot(times_1, ys_1, c="blue")
    sub_plot_x.plot(times_2, xs_2, c="red", marker='x')
    sub_plot_y.plot(times_2, ys_2, c="red", marker='x')

    plt.show()


# xyを同じfigにプロット
def plot_mean_xy(features, c_nodes):
    mean_1 = []
    mean_2 = []

    for feature in features:
        mean_1.append(np.mean(feature[:c_nodes[0]], 0))

    for feature in features:
        mean_2.append(np.mean(feature[c_nodes[0]:], 0))

    times = []

    for i in range(len(features)):
        times.append(i)

    fig = plt.figure()
    sub_plot_x = fig.add_subplot(1, 2, 1)
    sub_plot_y = fig.add_subplot(1, 2, 2)
    sub_plot_x.set_title("x")
    sub_plot_y.set_title("y")
    sub_plot_x.set_ylim(-1, 1)
    sub_plot_y.set_ylim(-1, 1)
    sub_plot_x.plot(times, np.array(mean_1)[:, 0], c="blue")
    sub_plot_y.plot(times, np.array(mean_1)[:, 1], c="blue")
    sub_plot_x.plot(times, np.array(mean_2)[:, 0], c="red")
    sub_plot_y.plot(times, np.array(mean_2)[:, 1], c="red")

    plt.show()


# 座標をプロット
def plot_pos(node_locs):
    fig = plt.figure(figsize=(10, 10))
    for it, node_loc in enumerate(node_locs):
        subplot = fig.add_subplot(4, 5, it+1)
        subplot.set_xlim(-6, 6)
        subplot.set_ylim(-6, 6)
        subplot.scatter(node_loc[:, 0], node_loc[:, 1])
    plt.show()
