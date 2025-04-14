import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# PyTorch version of the weight initializer
def weight_variable_random_uniform(input_dim, output_dim=None):
    if output_dim is not None:
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = torch.empty(input_dim, output_dim).uniform_(-init_range, init_range)
    else:
        init_range = np.sqrt(6.0 / input_dim)
        initial = torch.empty(input_dim).uniform_(-init_range, init_range)
    return nn.Parameter(initial)

# Dropout for sparse tensors (approximate version)
def dropout_sparse(x, keep_prob):
    # Randomly zero some values in a sparse tensor
    mask = ((torch.rand(x._values().size()) + keep_prob).floor()).bool()
    rc = x._indices()
    val = x._values()[mask] * (1.0 / keep_prob)
    idx = rc[:, mask]
    return torch.sparse_coo_tensor(idx, val, x.shape).coalesce()

# Dot product wrapper
def dot(x, y, sparse=False):
    if sparse:
        return torch.sparse.mm(x, y)
    else:
        return torch.matmul(x, y)

class StackGCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_support, dropout=0.,
                 share_user_item_weights=True, sparse_inputs=False):
        super(StackGCN, self).__init__()

        assert output_dim % num_support == 0, 'output_dim must be multiple of num_support for stackGC layer'

        self.weights_u = nn.ParameterList()
        self.weights_v = nn.ParameterList()

        for i in range(num_support):
            self.weights_u.append(weight_variable_random_uniform(input_dim, output_dim // num_support))

            if share_user_item_weights:
                self.weights_v.append(self.weights_u[i])
            else:
                self.weights_v.append(weight_variable_random_uniform(input_dim, output_dim // num_support))

        self.num_support = num_support
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs

    def forward(self, x_u, x_v, supports, supports_t):
        if self.sparse_inputs:
            x_u = dropout_sparse(x_u, 1 - self.dropout)
            x_v = dropout_sparse(x_v, 1 - self.dropout)
        else:
            x_u = F.dropout(x_u, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        supports_u = []
        supports_v = []

        for i in range(self.num_support):
            tmp_u = dot(x_u, self.weights_u[i], sparse=self.sparse_inputs)
            tmp_v = dot(x_v, self.weights_v[i], sparse=self.sparse_inputs)

            support = supports[i]
            support_t = supports_t[i]

            supports_u.append(torch.sparse.mm(support, tmp_v))
            supports_v.append(torch.sparse.mm(support_t, tmp_u))

        z_u = torch.cat(supports_u, dim=1)
        z_v = torch.cat(supports_v, dim=1)

        u_outputs = F.relu(z_u)
        v_outputs = F.relu(z_v)

        return u_outputs, v_outputs


class OrdinalMixtureGCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_support, dropout=0.0,
                 sparse_inputs=False, bias=False, share_user_item_weights=False,
                 self_connections=False):
        super().__init__()

        self.dropout = dropout
        self.sparse_inputs = sparse_inputs
        self.self_connections = self_connections
        self.bias_enabled = bias

        self.weights_u = nn.ParameterList([
            nn.Parameter(torch.empty(input_dim, output_dim)) for _ in range(num_support)
        ])

        if not share_user_item_weights:
            self.weights_v = nn.ParameterList([
                nn.Parameter(torch.empty(input_dim, output_dim)) for _ in range(num_support)
            ])
        else:
            self.weights_v = self.weights_u

        if self_connections:
            self.weights_u_self_conn = nn.Parameter(torch.empty(input_dim, output_dim))
            self.weights_v_self_conn = nn.Parameter(torch.empty(input_dim, output_dim))

        if bias:
            self.bias_u = nn.Parameter(torch.zeros(output_dim))
            self.bias_v = nn.Parameter(torch.zeros(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weights_u:
            nn.init.xavier_uniform_(w)
        for w in self.weights_v:
            nn.init.xavier_uniform_(w)
        if self.self_connections:
            nn.init.xavier_uniform_(self.weights_u_self_conn)
            nn.init.xavier_uniform_(self.weights_v_self_conn)

    def forward(self, x_u, x_v, support_list, support_t_list,
                u_self_support=None, v_self_support=None):

        if self.sparse_inputs:
            # 구현 필요시 dropout_sparse 등 정의 가능
            raise NotImplementedError("sparse dropout is not implemented in PyTorch.")
        else:
            x_u = F.dropout(x_u, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        supports_u, supports_v = [], []

        # Self-connections
        if self.self_connections:
            u_out = torch.sparse.mm(u_self_support, torch.matmul(x_u, self.weights_u_self_conn))
            v_out = torch.sparse.mm(v_self_support, torch.matmul(x_v, self.weights_v_self_conn))
            supports_u.append(u_out)
            supports_v.append(v_out)

        # Normal supports
        for i in range(len(support_list)):
            w_u = self.weights_u[i]
            w_v = self.weights_v[i]
            tmp_u = torch.matmul(x_u, w_u)
            tmp_v = torch.matmul(x_v, w_v)

            su = torch.sparse.mm(support_list[i], tmp_v)
            sv = torch.sparse.mm(support_t_list[i], tmp_u)

            supports_u.append(su)
            supports_v.append(sv)

        z_u = torch.stack(supports_u, dim=0).sum(dim=0)
        z_v = torch.stack(supports_v, dim=0).sum(dim=0)

        if self.bias_enabled:
            z_u += self.bias_u
            z_v += self.bias_v

        return F.relu(z_u), F.relu(z_v)

class BilinearMixture(nn.Module):
    def __init__(self, num_classes, input_dim, num_weights=3,
                 diagonal=True, user_item_bias=False, num_users=None, num_items=None, dropout=0.0):
        super().__init__()

        self.num_classes = num_classes
        self.num_weights = num_weights
        self.diagonal = diagonal
        self.user_item_bias = user_item_bias
        self.dropout = dropout

        if diagonal:
            self.weights = nn.ParameterList([
                nn.Parameter(torch.empty(input_dim)) for _ in range(num_weights)
            ])
        else:
            self.weights = nn.ParameterList([
                nn.Parameter(torch.empty(input_dim, input_dim)) for _ in range(num_weights)
            ])

        self.weights_scalar = nn.Parameter(torch.empty(num_weights, num_classes))

        if user_item_bias:
            self.user_bias = nn.Parameter(torch.zeros(num_users, num_classes))
            self.item_bias = nn.Parameter(torch.zeros(num_items, num_classes))

        self.reset_parameters()

    def reset_parameters(self):
        for w in self.weights:
            if self.diagonal:
                nn.init.uniform_(w, -0.1, 0.1)
            else:
                nn.init.xavier_uniform_(w)
        nn.init.xavier_uniform_(self.weights_scalar)

    def forward(self, u_emb, v_emb, u_indices, v_indices):
        u_emb = F.dropout(u_emb, p=self.dropout, training=self.training)
        v_emb = F.dropout(v_emb, p=self.dropout, training=self.training)

        u = u_emb[u_indices]
        v = v_emb[v_indices]

        basis_outputs = []
        for i in range(self.num_weights):
            if self.diagonal:
                ui = u * self.weights[i]
            else:
                ui = torch.matmul(u, self.weights[i])
            x = torch.sum(ui * v, dim=1)  # dot product
            basis_outputs.append(x)

        basis_stack = torch.stack(basis_outputs, dim=1)
        out = torch.matmul(basis_stack, self.weights_scalar)

        if self.user_item_bias:
            out += self.user_bias[u_indices]
            out += self.item_bias[v_indices]

        return F.log_softmax(out, dim=1)

class RecommenderSideInfoGAE(nn.Module):
    def __init__(self, input_dim, feat_hidden_dim, hidden_dims, num_classes,
                 num_basis_functions, num_users, num_items, num_side_features,
                 accum='sum', self_connections=False, dropout=0.5):
        super().__init__()

        # 1. GCN layer (custom 구현 필요: StackGCN or OrdinalMixtureGCN)
        if accum == 'sum':
            self.gcn = OrdinalMixtureGCN(input_dim, hidden_dims[0], self_connections)
        elif accum == 'stack':
            self.gcn = StackGCN(input_dim, hidden_dims[0])
        else:
            raise ValueError("accum must be 'sum' or 'stack'")

        # 2. Dense for side features
        self.side_dense = nn.Linear(num_side_features, feat_hidden_dim)

        # 3. Projection layer after concat
        self.concat_dense = nn.Linear(hidden_dims[0] + feat_hidden_dim, hidden_dims[1])

        # 4. Bilinear decoder
        self.decoder = BilinearMixture(
            input_dim=hidden_dims[1],
            num_classes=num_classes,
            num_users=num_users,
            num_items=num_items,
            num_weights=num_basis_functions
        )

        self.dropout = dropout

    def forward(self, u_feat, v_feat, u_feat_side, v_feat_side,
            support, support_t, u_indices, v_indices):
    
        # GCN
        gcn_u, gcn_v = self.gcn(u_feat, v_feat, support, support_t)

        gcn_u = F.dropout(gcn_u, p=self.dropout, training=self.training)
        gcn_v = F.dropout(gcn_v, p=self.dropout, training=self.training)
        
        # Side info
        feat_u = F.relu(self.side_dense(u_feat_side))
        feat_v = F.relu(self.side_dense(v_feat_side))
                    
        feat_u = F.dropout(feat_u, p=self.dropout, training=self.training)
        feat_v = F.dropout(feat_v, p=self.dropout, training=self.training)
        
        # Concat + projection
        concat_u = torch.cat([gcn_u, feat_u], dim=1)
        concat_v = torch.cat([gcn_v, feat_v], dim=1)

        proj_u = F.dropout(self.concat_dense(concat_u), p=self.dropout, training=self.training)
        proj_v = F.dropout(self.concat_dense(concat_v), p=self.dropout, training=self.training)

        # Decoder: 선택된 user-item 쌍에 대해만 예측
        out = self.decoder(proj_u, proj_v, u_indices, v_indices)

        return out
