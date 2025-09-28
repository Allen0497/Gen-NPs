import os
from importlib.machinery import SourceFileLoader
import math
import torch

def gen_load_func(parser, func):
    def load(args, cmdline):
        sub_args, cmdline = parser.parse_known_args(cmdline)
        for k, v in sub_args.__dict__.items():
            args.__dict__[k] = v
        return func(**sub_args.__dict__), cmdline
    return load


def load_module(filename):

    # 动态加载指定路径的 Python 模块
    module_name = os.path.splitext(os.path.basename(filename))[0]
    return SourceFileLoader(module_name, filename).load_module()
    # <module "module_name" from "filename">
    #
    # ex.
    # <module "cnp" from "models/cnp.py">


def logmeanexp(x, dim=0):
    # 计算给定维度上张量元素的指数平均的对数
    return x.logsumexp(dim) - math.log(x.shape[dim])


def stack(x, num_samples=None, dim=0):
    # 在指定维度上重复堆叠张量
    return x if num_samples is None \
            else torch.stack([x]*num_samples, dim=dim)


def hrminsec(duration):
    hours, left = duration // 3600, duration % 3600
    mins, secs = left // 60, left % 60
    return f"{hours}hrs {mins}mins {secs}secs"


def calculate_batch_lof(X, k):
    """
    计算给定批次数据的局部离群因子（LOF）分数。

    参数:
    X - 输入数据的张量，形状应为 [B, N, D]，其中
        B 是批次大小，
        N 是每批中的数据点数，
        D 是每个数据点的特征数量。
    k - 用于计算LOF的邻居数量。

    返回:
    一个形状为 [B, N, 1] 的张量，包含每批中每个数据点的LOF分数。
    """

    def calculate_lof(X, k):
        if X.ndim == 1:
            X = X.view(-1, 1)  # 将一维数组转换为二维 [N, 1]

        def calculate_distance_matrix(X):
            return torch.cdist(X, X, p=2)

        def k_nearest_neighbors(distances, k):
            knn_distances, knn_indices = torch.topk(distances, k+1, largest=False, sorted=True)
            return knn_distances[:, 1:], knn_indices[:, 1:]

        def local_density(knn_distances):
            return 1.0 / knn_distances.mean(dim=1)

        def lof_score(densities, knn_indices):
            extended_densities = densities.unsqueeze(0).repeat(knn_indices.size(0), 1)
            reach_density = torch.gather(extended_densities, 1, knn_indices)
            lof_scores = reach_density.mean(dim=1) / densities
            return lof_scores

        distances = calculate_distance_matrix(X)
        knn_distances, knn_indices = k_nearest_neighbors(distances, k)
        densities = local_density(knn_distances)
        lof_scores = lof_score(densities, knn_indices)

        return lof_scores  # 确保输出格式为 [N, 1]

    # 初始化列表来存储每批次的LOF结果
    lof_results = [calculate_lof(batch, k) for batch in X]

    # 使用 torch.stack 将所有结果堆叠成一个新的三维张量
    return torch.stack(lof_results)