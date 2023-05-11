import networkx as nx
import numpy as np
import numpy.linalg as LA
import copy
import utils
# import matplotlib.pyplot as plt

from scipy.spatial import distance


# グラフを作成する関数
def make_graph(edge_file):
    G = nx.Graph()
    G = nx.read_edgelist(edge_file, nodetype=int)

    return G


# ランダムにグラフを作成する関数。引数はクラスタの数と一つのクラスタに属するノードの数
def make_random_graph(num_cluster, num_node):
    graphs = []
    p = 0.3
    seed = 42
    for i in range(num_cluster):
        random_graph = nx.gnp_random_graph(num_node, p, seed + i)
        random_graph.to_undirected()
        graphs.append(random_graph)
    join_graph = nx.disjoint_union_all(graphs)

    return join_graph


# グラフの孤立したノード同士を引っ付ける関数
def connect_nodes(graph, nodes_list, weight_list):
    new_graph = copy.deepcopy(graph)
    for it, nodes in enumerate(nodes_list):
        new_graph.add_edge(nodes[0], nodes[1], weight=weight_list[it])

    return new_graph


# numpy配列で隣接行列をもらい、そこからラプラシアン行列を計算する関数
def make_laplacian(A):
    D = np.diag(np.sum(A, axis=0))
    L = D - A

    return L


# 正規化ラプラシアン行列を返す関数
def make_normed_laplacian(A):
    D = np.diag(np.sum(A, axis=0))
    L = D - A
    normed_L = np.dot((np.sqrt(LA.inv(D))), np.dot(L, np.sqrt(LA.inv(D))))

    return normed_L


# numpy配列で隣接行列をもらいそこから特徴を計算し返す関数
def make_feature(A):
    # L = make_laplacian(A)
    L = make_normed_laplacian(A)
    lam, v = LA.eigh(L)
    vec1 = []
    vec2 = []

    for i in range(len(v)):
        vec1.append(v[i, 0])
        vec2.append(v[i, 1])
    vec = [vec1, vec2]
    print(lam[0])
    print(lam[1])
    print("\n")
    feature = np.array(vec)
    feature = feature.T

    return feature


# ランダムにグラフを作成する関数。引数はクラスタの数と一つのクラスタに属するノードの数
def make_circle(radius, c_num, amp, omega, time, init):
    mu_x = []
    mu_y = []
    for i in range(0, c_num):
        x = np.cos(2 * np.pi * i / c_num)
        y = np.sin(2 * np.pi * i / c_num)
        mu_x.append(x * (amp * np.sin(omega * time + init) + radius))
        mu_y.append(y * (amp * np.sin(omega * time + init) + radius))

    loc = np.array([mu_x, mu_y])
    loc = loc.T
    print(LA.norm(loc[0] - loc[1]))

    return loc


# 指定した場所を平均とする分散varのnodeをnode_num個出力する関数
def make_nodes_normal(loc, var, node_num):
    var = np.diag([var, var])
    nodes_loc = np.random.multivariate_normal(loc[0], var, node_num[0])
    for c in range(1, len(node_num)):
        ramdom_node = np.random.multivariate_normal(loc[c], var, node_num[c])
        nodes_loc = np.append(nodes_loc, copy.deepcopy(ramdom_node), axis=0)

    return nodes_loc


# k近傍に重み付き辺を張る
def make_edge(nodes_loc, k):
    graph = nx.Graph()
    for node in range(len(nodes_loc)):
        graph.add_node(node)
    dist = distance.squareform(distance.pdist(nodes_loc))
    arg_dist = np.argsort(dist)
    arg_dist = arg_dist[:, 1:k+1]

    for i in range(len(arg_dist)):
        for j in range(k):
            target = arg_dist[i][j]
            graph.add_edge(i, target,
                           weight=np.exp(-(dist[i][target]**2) / 2))

    graph.to_undirected()

    return graph


# 円形にどんどん進化していくグラフを生成する
def make_circle_graph(radius, c_num, amp, omega, times, ini, var, node_num, k):
    graphs = []
    node_locs = []
    for time in range(times):
        loc = make_circle(radius, c_num, amp, omega, time, ini)
        nodes_loc = make_nodes_normal(loc, var, node_num)
        graphs.append(make_edge(nodes_loc, k))
        node_locs.append(nodes_loc)
    utils.plot_pos(node_locs)

    return graphs


# main文
def main():
    # 指定されたグラフを読み込んで隣接行列Aを計算する
    # G = make_graph("./conf/graph.txt")
    # A = nx.adjacency_matrix(G).todense().astype(float)
    # print(nx.adjacency_matrix(G).todense().astype(float))

    # ランダムなグラフの特徴を見る場合の処理
    # G = make_random_graph(2, 500)
    # A = nx.adjacency_matrix(G).todense().astype(float)

    T = 20
    node_num = [200, 300]
    k = 10

    graphs = make_circle_graph(radius=3, c_num=2, amp=1.0, omega=np.pi / 4,
                               times=T, ini=np.pi, var=1.0,
                               node_num=node_num, k=k)

#    for t in range(len(graphs)):
#        pos = nx.spring_layout(graphs[t], seed=0)
#        plt.figure(figsize=(10, 10))
#        nx.draw_networkx(graphs[t], pos)
#        plt.show()

    # print(nx.adjacency_matrix(graphs[0]).todense().astype(float))
    # print(graphs[0])


# 指定されたノード同士をじっくりくっつける
    A_list = []
    for i in range(0, T):
        A_list.append(nx.adjacency_matrix(graphs[i]).todense().astype(float))


# 特徴の計算
    features = []
    for A in A_list:
        features.append(copy.deepcopy(make_feature(A)))

#    for feature in features:
#        print(np.cov(feature[:200].T))
#        print("\n")

# 特徴のプロット
    utils.plot_feature(features, node_num)
    # utils.plot_xy(features, node_num)
    # utils.plot_mean_xy(features, node_num)


if __name__ == "__main__":
    main()
