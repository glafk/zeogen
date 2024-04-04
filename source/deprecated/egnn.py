from torch import nn
import torch
from torch_scatter import scatter_mean, scatter_add

# DEPRECATED
class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    adjusted for periodicity
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(),
                 residual=True, attention=False, normalize=False, coords_agg='mean', 
                 tanh=False, l=None, z_size=0):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1
        self.l = nn.Parameter(l, requires_grad=False)
        self.z_size = z_size

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + z_size, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr, num_nodes, z=None):
        row, col = edge_index
        
        agg = scatter_add(edge_attr, row, 0, dim_size=num_nodes)
        # agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        
        if z is not None:
            agg = torch.cat([agg, z], dim=1)

        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, num_nodes):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = scatter_add(trans, row, 0, dim_size=num_nodes)
        elif self.coords_agg == 'mean':
            agg = scatter_mean(trans, row, 0, dim_size=num_nodes)
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        
            
        coord = torch.remainder(coord, self.l) # map coordinates back into the unit cell
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        diff = coord[row] - coord[col]
        
        # adjust for pbc
        diff = torch.where(diff > 0.5 * self.l, diff - self.l, diff)
        diff = torch.where(diff < -0.5 * self.l, diff + self.l, diff)
        coord_diff = diff
        
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff =         coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None, num_nodes=None, z=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, num_nodes)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, num_nodes, z)

        return h, coord, edge_attr, edge_feat


class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False,
                 normalize=False, tanh=False, l=None, predict_force=False, z_size=0):
        '''

        :param in_node_nf: Number of features for 'h' at the input
        :param hidden_nf: Number of hidden features
        :param out_node_nf: Number of features for 'h' at the output
        :param in_edge_nf: Number of features for the edge features
        :param device: Device (e.g. 'cpu', 'cuda:0',...)
        :param act_fn: Non-linearity
        :param n_layers: Number of layer for the EGNN
        :param residual: Use residual connections, we recommend not changing this one
        :param attention: Whether using attention or not
        :param normalize: Normalizes the coordinates messages such that:
                    instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
                    we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
                    We noticed it may help in the stability or generalization in some future works.
                    We didn't use it in our paper.
        :param tanh: Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds the output of
                        phi_x(m_ij) which definitely improves in stability but it may decrease in accuracy.
                        We didn't use it in our paper.
        '''

        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        self.predict_force = predict_force
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh, l=l, z_size=z_size))
        if self.predict_force: 
            self.force_mlp = nn.Linear(hidden_nf, 1)

        self.to(self.device)

    def forward(self, h, x, edges, edge_attr, num_nodes, z=None):
        # TODO: Check the order in which parameters are passed to this function as it seems to be incorrect
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, edge_attr, edge_feat = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr, num_nodes=num_nodes, z=z)
        h = self.embedding_out(h)
            

        return h, x