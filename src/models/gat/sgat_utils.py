import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import scatter

sig = nn.Sigmoid()
gamma = -0.1
zeta = 1.1
beta = 0.22
eps = 1e-20
const1 = beta*np.log(-gamma/zeta + eps)


def l0_train(logAlpha, min, max):
    U = torch.rand(logAlpha.size()).type_as(logAlpha) + eps
    s = sig((torch.log(U / (1 - U)) + logAlpha) / beta)
    s_bar = s * (zeta - gamma) + gamma
    mask = F.hardtanh(s_bar, min, max)
    return mask


def l0_test(logAlpha, min, max):
    s = sig(logAlpha/beta)
    s_bar = s * (zeta - gamma) + gamma
    mask = F.hardtanh(s_bar, min, max)

    return mask


def get_loss2(logAlpha):
    return sig(logAlpha - const1)


def standardize_per_node(src: Tensor,
                         index: Tensor,
                         num_nodes: int = None,
                         eps: float = 1e-6) -> Tensor:
    """
    Zero-mean / unit-std standardisation of src [E, H] over the edges of each destination
    node (per head), rather than over all edges in the graph. Result for a node is then
    independent of how many other destination nodes share the graph, so stacking several
    target cells in one graph gives the same values as one graph per cell.
    Population std (unbiased=False) and the same eps as the former global version.
    """
    N = maybe_num_nodes(index, num_nodes)

    count = scatter(torch.ones_like(src[:, :1]), index, dim=0, dim_size=N, reduce='sum').clamp(min=1)  # [N, 1]
    mean = scatter(src, index, dim=0, dim_size=N, reduce='sum') / count  # [N, H]
    centered = src - mean.index_select(0, index)
    var = scatter(centered ** 2, index, dim=0, dim_size=N, reduce='sum') / count  # [N, H]

    return centered / (var.sqrt().index_select(0, index) + eps)


def masked_normalize_multihead(src: Tensor,
                               index: Tensor,
                               num_nodes: int = None,
                               eps: float = 1e-16) -> Tensor:
    """
    src: [E, H]
    index: [E]
    mask: [E, 1] or [E, H]
    """

    N = maybe_num_nodes(index, num_nodes)

    # Sum per destination node *per head*
    # Move dim 0 to be edges; scatter over dim 0:
    group_sum = scatter(src, index, dim=0, dim_size=N, reduce='sum')  # [N, H]

    # Broadcast sums back to edge space:
    denom = group_sum.index_select(0, index)  # [E, H]

    out = torch.zeros_like(src)
    nonzero = denom > eps
    out[nonzero] = src[nonzero] / denom[nonzero]

    return out
