import torch
import copy


# グラフスペクトルを正規化する
# 固有値の内積の正負によって-1をかけるかを決める
def normalize_spectol(ses):
    normed_spectol = []
    normed_spectol.append(ses[0])

    # 行列積を使って各ノードにおける時刻間のグラフスペクトルの内積を計算する
    # 行列積の対角成分の平均を計算
    for t in range(len(ses)-1):
        dp = torch.mm(torch.t(normed_spectol[-1]), ses[t+1])
        dp = torch.diagonal(dp, 0)
        tmp_spectol = []

        for i, d in enumerate(dp):
            if d < 0:
                tmp_spectol.append(-1 * ses[t+1][:, i])
            else:
                tmp_spectol.append(ses[t+1][:, i])

        normed_spectol.append(torch.stack(tmp_spectol, dim=1))
        tmp_spectol.clear()
    normed_spectol = torch.stack(normed_spectol)

    return normed_spectol


# グラフスペクトルを正規化する
# グラフスペクトルの平均で正負を決める
def normalize_spectol_v2(cfg, ses):
    avg_vectors = []
    normed_spectol = []
    g_spectols = copy.deepcopy(ses)
    # 各時刻の平均ベクトルを求める
    for g_spectol in ses:
        avg_vectors.append(torch.mean(g_spectol, dim=0))

    # 各時刻の平均ベクトルの内積を使って-1をかけるかを決める
    normed_spectol.append(g_spectols[0])
    for t, avg in enumerate(avg_vectors[:-1]):
        product = avg_vectors[t] * avg_vectors[t+1]
        tmp = []
        for idx in range(len(avg)):
            if product[idx] < 0:
                tmp.append(-1*g_spectols[t+1][:, idx])
            else:
                tmp.append(g_spectols[t+1][:, idx])
        normed_spectol.append(torch.stack(tmp, dim=1))
        tmp.clear()

    normed_spectol = torch.stack(normed_spectol)

    return normed_spectol


if __name__ == "__main__":
    hi = torch.tensor([1.0, 2.0])
    p1 = torch.tensor([2.0, 4.0])
    p2 = torch.tensor([100.0, 200.0])
    label = torch.tensor([0, 1, 0])
    pred = torch.tensor([0.4, 0.6, 0.8])

    # pos_features = [p1, p2]
    # print(smote(hi, pos_features, 3))
