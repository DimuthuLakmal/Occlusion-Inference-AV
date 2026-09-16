import typing
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import (
    Adj,
    NoneType,
    OptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.utils.sparse import set_sparse_value

from src.models.gat.message_passing import MessagePassing
from src.models.gat.sgat_utils import l0_train, l0_test, get_loss2, masked_normalize_multihead

if typing.TYPE_CHECKING:
    from typing import overload
else:
    from torch.jit import _overload_method as overload


class GATv2Conv(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            share_weights: bool = False,
            residual: bool = False,
            **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.residual = residual
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(
                in_channels,
                heads * out_channels,
                bias=bias,
                weight_initializer='glorot'
            )
            self.lin_l_z = Linear(
                in_channels,
                heads * out_channels,
                bias=bias,
                weight_initializer='glorot'
            )

            if share_weights:
                self.lin_r = self.lin_l
                self.lin_r_z = self.lin_l_z
            else:
                self.lin_r = Linear(
                    in_channels,
                    heads * out_channels,
                    bias=bias,
                    weight_initializer='glorot'
                )
                self.lin_r_z = Linear(
                    in_channels,
                    heads * out_channels,
                    bias=bias,
                    weight_initializer='glorot'
                )

        else:
            self.lin_l = Linear(
                in_channels[0],
                heads * out_channels,
                bias=bias,
                weight_initializer='glorot'
            )
            self.lin_l_z = Linear(
                in_channels[0],
                heads * out_channels,
                bias=bias,
                weight_initializer='glorot'
            )

            if share_weights:
                self.lin_r = self.lin_l
                self.lin_r_z = self.lin_l_z
            else:
                self.lin_r = Linear(
                    in_channels[1],
                    heads * out_channels,
                    bias=bias,
                    weight_initializer='glorot'
                )
                self.lin_r_z = Linear(
                    in_channels[1],
                    heads * out_channels,
                    bias=bias,
                    weight_initializer='glorot'
                )

        self.att = Parameter(torch.empty(1, heads, out_channels))

        # Head-specific mask attention parameters.
        self.att_z_l = Parameter(torch.empty(1, heads, out_channels))
        self.att_z_r = Parameter(torch.empty(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(
                edge_dim,
                heads * out_channels,
                bias=False,
                weight_initializer='glorot'
            )
            self.lin_edge_z = Linear(
                edge_dim,
                heads * out_channels,
                bias=False,
                weight_initializer='glorot'
            )
        else:
            self.lin_edge = None
            self.lin_edge_z = None

        total_out_channels = out_channels * (heads if concat else 1)

        if residual:
            self.res = Linear(
                in_channels if isinstance(in_channels, int) else in_channels[1],
                total_out_channels,
                bias=False,
                weight_initializer='glorot',
            )
        else:
            self.register_parameter('res', None)

        if bias:
            self.bias = Parameter(torch.empty(total_out_channels))
        else:
            self.register_parameter('bias', None)

        self.gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * out_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, 1),
            torch.nn.Sigmoid()
        )

        # SGAT / L0 mask parameters.
        self.bias_l0 = nn.Parameter(torch.FloatTensor([0]))

        # Changed from scalar [1] to per-head [1, H].
        # This allows each head to learn a different sparsity threshold.
        self.bias_l0_z = nn.Parameter(torch.zeros(1, heads))

        self.loss = 0

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

        # Main attention branch
        nn.init.uniform_(self.lin_l.weight, a=-1.0, b=1.0)
        nn.init.uniform_(self.lin_r.weight, a=-1.0, b=1.0)

        # Sparse-mask branch
        nn.init.uniform_(self.lin_l_z.weight, a=-1.0, b=1.0)
        nn.init.uniform_(self.lin_r_z.weight, a=-1.0, b=1.0)

        if self.lin_l.bias is not None:
            zeros(self.lin_l.bias)
        if self.lin_r.bias is not None:
            zeros(self.lin_r.bias)
        if self.lin_l_z.bias is not None:
            zeros(self.lin_l_z.bias)
        if self.lin_r_z.bias is not None:
            zeros(self.lin_r_z.bias)

        # Edge feature branches
        if self.lin_edge is not None:
            nn.init.uniform_(self.lin_edge.weight, a=-1.0, b=1.0)

        if self.lin_edge_z is not None:
            nn.init.uniform_(self.lin_edge_z.weight, a=-1.0, b=1.0)

        # Residual branch
        if self.res is not None:
            nn.init.uniform_(self.res.weight, a=-1.0, b=1.0)

        # Attention vectors
        glorot(self.att)
        glorot(self.att_z_l)
        glorot(self.att_z_r)

        # Output bias
        if self.bias is not None:
            zeros(self.bias)

        # L0 gate biases
        with torch.no_grad():
            self.bias_l0.zero_()
            self.bias_l0_z.zero_()

    @overload
    def forward(
            self,
            x: Union[Tensor, PairTensor],
            edge_index: Adj,
            edge_attr: OptTensor = None,
            return_attention_weights: NoneType = None,
    ) -> Tensor:
        pass

    @overload
    def forward(  # noqa: F811
            self,
            x: Union[Tensor, PairTensor],
            edge_index: Tensor,
            edge_attr: OptTensor = None,
            return_attention_weights: bool = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @overload
    def forward(  # noqa: F811
            self,
            x: Union[Tensor, PairTensor],
            edge_index: SparseTensor,
            edge_attr: OptTensor = None,
            return_attention_weights: bool = None,
    ) -> Tuple[Tensor, SparseTensor]:
        pass

    def forward(
            self,
            x: Union[Tensor, PairTensor],
            edge_index: Adj,
            edge_attr: OptTensor = None,
            return_attention_weights: Optional[bool] = None,
    ) -> Union[
        Tensor,
        Tuple[Tensor, Tuple[Tensor, Tensor]],
        Tuple[Tensor, SparseTensor],
        Tuple[Tensor, Tensor]
    ]:
        H, C = self.heads, self.out_channels

        res: Optional[Tensor] = None

        x_l: OptTensor = None
        x_r: OptTensor = None
        x_l_z: OptTensor = None
        x_r_z: OptTensor = None

        if isinstance(x, Tensor):
            assert x.dim() == 2

            if self.res is not None:
                res = self.res(x)

            x_l = self.lin_l(x).view(-1, H, C)
            x_l_z = self.lin_l_z(x).view(-1, H, C)

            if self.share_weights:
                x_r = x_l
                x_r_z = x_l_z
            else:
                x_r = self.lin_r(x).view(-1, H, C)
                x_r_z = self.lin_r_z(x).view(-1, H, C)

        else:
            x_l_input, x_z_input, x_r_input, x_r_z_input = x[0], x[1], x[2], x[3]
            assert x_l_input.dim() == 2

            if x_r_input is not None and self.res is not None:
                res = self.res(x_r_input)

            x_l = self.lin_l(x_l_input).view(-1, H, C)
            x_l_z = self.lin_l_z(x_z_input).view(-1, H, C)

            if x_r_input is not None:
                x_r = self.lin_r(x_r_input).view(-1, H, C)
                x_r_z = self.lin_r_z(x_r_z_input).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None
        assert x_l_z is not None
        assert x_r_z is not None

        self.loss = 0

        # Compute sparse masks first.
        self.z = self.edge_updater(
            edge_index,
            x=(x_l_z, x_r_z),
            edge_attr=edge_attr,
            func='edge_update_z'
        )

        # Compute attention and apply mask.
        alpha = self.edge_updater(
            edge_index,
            x=(x_l, x_r),
            edge_attr=edge_attr,
            z=self.z,
            func='edge_update_alpha'
        )

        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if res is not None:
            out = out + res

        if self.bias is not None:
            out = out + self.bias

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                if is_torch_sparse_tensor(edge_index):
                    adj = set_sparse_value(edge_index, alpha)
                    return out, (adj, alpha)
                else:
                    return out, (edge_index, alpha)

            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')

        else:
            return out, self.loss, self.z

    def edge_update_z(
            self,
            x_j: Tensor,
            x_i: Tensor,
            edge_attr: OptTensor,
            index: Tensor,
            ptr: OptTensor,
            dim_size: Optional[int],
    ) -> Tensor:
        """
        Compute L0 sparse mask per edge and per head.

        x_i, x_j: [E, H, C]
        output z_raw: [E, H]
        """

        # Use the mask-specific attention parameters.
        # This was missing in the current version, although att_z_l/att_z_r exist.
        logits_l = (x_i * self.att_z_l).sum(dim=-1)  # [E, H]
        logits_r = (x_j * self.att_z_r).sum(dim=-1)  # [E, H]

        logits = logits_l + logits_r + self.bias_l0_z  # [E, H]

        # Add edge features in a head-specific way.
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)

            assert self.lin_edge_z is not None

            edge_attr_z = self.lin_edge_z(edge_attr)  # [E, H * C]
            edge_attr_z = edge_attr_z.view(
                -1,
                self.heads,
                self.out_channels
            )  # [E, H, C]

            logits = logits + edge_attr_z.sum(dim=-1)  # [E, H]

        # Important change:
        # Normalise each head over all edges, not each edge over all heads.
        std = logits.std(dim=0, keepdim=True, unbiased=False)
        logits = (logits - logits.mean(dim=0, keepdim=True)) / (std + 1e-6)

        if self.training:
            # Keep this simple first.
            # You can add multi-sample averaging later if the gates are too noisy.
            z_raw = l0_train(logits, 0.0, 1.0)  # [E, H]
        else:
            z_raw = l0_test(logits, 0.0, 1.0)  # [E, H]

        # L0 regularisation term.
        self.loss = get_loss2(logits).sum()

        return z_raw

    def edge_update_alpha(
            self,
            x_j: Tensor,
            x_i: Tensor,
            edge_attr: OptTensor,
            index: Tensor,
            ptr: OptTensor,
            dim_size: Optional[int],
            z: Tensor,
    ) -> Tensor:
        """
        Compute attention alpha, apply sparse mask, then renormalise.

        x_i, x_j: [E, H, C]
        z:        [E, H]
        alpha:    [E, H]
        """

        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)

            assert self.lin_edge is not None

            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(
                -1,
                self.heads,
                self.out_channels
            )

            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)

        alpha = (x * self.att).sum(dim=-1)  # [E, H]

        # Standard GAT softmax over incoming edges per target node.
        alpha = softmax(alpha, index, ptr, dim_size)

        # Apply learned sparse mask.
        alpha = alpha * z

        # Important change:
        # Re-normalise after masking so sparse heads do not automatically
        # produce lower-magnitude messages than the dense head.
        alpha = masked_normalize_multihead(alpha, index, dim_size)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
