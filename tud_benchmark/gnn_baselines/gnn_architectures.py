import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, JumpingKnowledge
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset

import torch.nn as nn
from torch_scatter import scatter


# TODO: try having a learnable mixing weight for message passing
def propagate_messages(x, reduction='none'):
    # x.shape == (2, E, C)
    assert x.dim() == 3
    assert x.shape[0] == 2
    if reduction == 'none':
        return torch.cat((x, x.flip(0)), dim=2)
    elif reduction == 'mean':
        return (x + x.flip(0)) / 2
    elif reduction == 'max':
        return torch.maximum(x, x.flip(0))
    elif reduction == 'min':
        return torch.minimum(x, x.flip(0))
    elif reduction == 'sum':
        return x + x.flip(0)
    else:
        raise NotImplementedError


# NOTE: isolated node features will be lost without self-loops
def expand_features(x, edge_index):
    # node features to edge features
    assert x.dim() == 2
    assert edge_index.dim() == 2
    assert edge_index.shape[0] == 2
    return x[edge_index]


def contract_features(x, edge_index):
    # edge features to node features
    assert x.dim() == 3
    assert x.shape[0] == 2
    assert edge_index.dim() == 2
    assert edge_index.shape[0] == 2
    return scatter(src=x.view(-1, x.shape[2]), index=edge_index.view(-1), dim=0, reduce='mean')


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, attention=False, key_dim=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attention = attention
        self.key_dim = key_dim

        if attention:
            assert key_dim > 0
            self.lin_q = nn.Linear(in_channels, key_dim)
            self.lin_k = nn.Linear(in_channels, key_dim)
            self.lin_v = nn.Linear(in_channels, out_channels)
            raise NotImplementedError
        else:
            self.lin_on = nn.Linear(in_channels, out_channels)
            self.lin_off = nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        self.lin_on.reset_parameters()
        self.lin_off.reset_parameters()

    # TODO: self-loops, edge types, isolated nodes
    def forward(self, x, edge_index):
        n = torch.max(edge_index) + 1
        e = edge_index.shape[1]

        assert x.shape == (2, e, self.in_channels)
        assert edge_index.shape == (2, e)

        x_on = self.lin_on(x)
        x_off = self.lin_off(x)

        assert x_on.shape == (2, e, self.out_channels)
        assert x_off.shape == (2, e, self.out_channels)

        # TODO: the layer can also be (non-linear) self-attention where every edge-end generates [key, value, query]. This would make the GNN anisotropic.
        # NOTE: currently, the layer is a linear set-equivariant layer, which is isotropic.
        # TODO: try adding a learnable/tunable temperature parameter to the softmax of attention

        # NOTE: scatter is highly parallelizable and suitable for GPUs
        z = scatter(
            src=x_off.view(-1, self.out_channels),
            index=edge_index.view(-1),
            dim=0, reduce='sum')
        deg = scatter(
            src=torch.ones_like(x)[..., 0].view(-1),
            index=edge_index.view(-1),
            dim=0, reduce='sum')

        assert z.shape == (n, self.out_channels)
        assert deg.shape == (n,)

        # NOTE: the following is a gather operation which, similar to the scatter operation above, is highly parallelizable and suitable for GPUs
        x_off = z[edge_index]
        deg = deg[edge_index]
        x_out = (x_on + x_off) / deg[..., None]

        assert x_off.shape == (2, e, self.out_channels)
        assert deg.shape == (2, e)
        assert x_out.shape == (2, e, self.out_channels)

        return x_out


class GraphConvNet(nn.Module):
    def __init__(self, dataset, num_layers, hidden, act_fn=F.relu, conv_repeat=1, skip_connection=False, reduction='mean'):
        super().__init__()
        self.dataset = dataset
        self.num_layers = num_layers
        self.hidden = hidden
        self.act_fn = act_fn
        self.conv_repeat = conv_repeat
        self.skip_connection = skip_connection
        self.reduction = reduction
        self.conv1 = GraphConv(dataset.num_features + dataset.num_edge_features, hidden)
        self.convs = nn.ModuleList()
        for i in range(num_layers - 1):
            for _ in range(conv_repeat):
                self.convs.append(GraphConv(hidden * (2 if reduction == 'none' else 1), hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        x = expand_features(x, edge_index)
        # x.shape == (2, E, dataset.num_features)

        if edge_attr is not None:
            print(edge_attr.shape)
            assert False
            # TODO: implement this part when the shape is known
            # possible implementation for edge_attr.shape == (E, d):
            # edge_attr = torch.stack((edge_attr, edge_attr), dim=0)
            # x = torch.cat((x, edge_attr), dim=2)

        # x = propagate_messages(x, reduction='none')
        x = self.conv1(x, edge_index)
        x = self.act_fn(x)
        x = propagate_messages(x, reduction=self.reduction)
        for conv in self.convs:
            x_old = x
            x = conv(x, edge_index)
            x = self.act_fn(x)
            # TODO: batchnorm or some kind of layer normalization
            x = propagate_messages(x, reduction=self.reduction)
            if self.skip_connection:
                x[..., :x_old.shape[2]] += x_old

        x = contract_features(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.act_fn(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


# Taken from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py.
class GIN(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GIN, self).__init__()
        self.conv1 = GINConv(Sequential(
            Linear(dataset.num_features, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            BN(hidden),
        ),
            train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    BN(hidden),
                ),
                    train_eps=True))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


# Taken from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py.
class GIN0(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GIN0, self).__init__()
        self.conv1 = GINConv(Sequential(
            Linear(dataset.num_features, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            BN(hidden),
        ),
            train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    BN(hidden),
                ),
                    train_eps=False))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


# Taken from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py.
class GINWithJK(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, mode='cat'):
        super(GINWithJK, self).__init__()
        self.conv1 = GINConv(Sequential(
            Linear(dataset.num_features, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            BN(hidden),
        ),
            train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden),
                    ReLU(),
                    BN(hidden),
                ),
                    train_eps=True))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GINE0Conv(MessagePassing):
    def __init__(self, edge_dim, dim_init, dim):
        super(GINE0Conv, self).__init__(aggr="add")

        self.edge_encoder = Sequential(Linear(edge_dim, dim_init), ReLU(), Linear(dim_init, dim_init), ReLU(),
                                       BN(dim_init))
        self.mlp = Sequential(Linear(dim_init, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp(x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        reset(self.edge_encoder)
        reset(self.mlp)


class GINE0(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GINE0, self).__init__()
        self.conv1 = GINE0Conv(dataset.num_edge_features, dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GINE0Conv(dataset.num_edge_features, hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GINEConv(MessagePassing):
    def __init__(self, edge_dim, dim_init, dim):
        super(GINEConv, self).__init__(aggr="add")

        self.edge_encoder = Sequential(Linear(edge_dim, dim_init), ReLU(), Linear(dim_init, dim_init), ReLU(),
                                       BN(dim_init))
        self.mlp = Sequential(Linear(dim_init, dim), ReLU(), Linear(dim, dim), ReLU(), BN(dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))
        self.initial_eps = 0

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.edge_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

    def reset_parameters(self):
        reset(self.edge_encoder)
        reset(self.mlp)
        self.eps.data.fill_(self.initial_eps)


class GINE(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden):
        super(GINE, self).__init__()
        self.conv1 = GINEConv(dataset.num_edge_features, dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GINEConv(dataset.num_edge_features, hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GINEWithJK(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, mode='cat'):
        super(GINEWithJK, self).__init__()
        self.conv1 = GINEConv(dataset.num_edge_features, dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GINEConv(dataset.num_edge_features, hidden, hidden))

        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
