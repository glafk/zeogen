import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
from torch_geometric.nn import aggr
from egnn import EGNN

# DEPRECATED
def unpack_batch(databatch, device='cuda'):
        x = databatch.x.float().to(device)
        pos = databatch.pos.float().to(device)
        edge_index = databatch.edge_index.to(device)
        batch = databatch.batch.to(device)
        num_nodes = databatch.num_nodes

        return x, pos, edge_index, batch, num_nodes

class Encoder(nn.Module):

    def __init__(self, l_uc, hidden_size=32, latent_size=32):
        super().__init__()

        self.l = nn.Parameter(l_uc, requires_grad=False)
        self.gnn = EGNN(1, hidden_size, latent_size, l=l_uc, device='cuda', normalize=True, tanh=True)
        self.aggr = aggr.MeanAggregation()
        
        self.z_mu = nn.Linear(latent_size, latent_size)
        self.z_sigma = nn.Linear(latent_size, latent_size)


    def forward(self, x, pos, edge_index, batch, num_nodes):
        h, _ = self.gnn(x, pos, edge_index, None, num_nodes)
        h = self.aggr(h, batch)
        mu = self.z_mu(h)
        sigma = F.softplus(self.z_sigma(h))

        return mu, sigma
    
class Decoder(nn.Module):

    def __init__(self, l_uc, hidden_size=32, latent_size=32):
        super().__init__()
        
        self.l = nn.Parameter(l_uc, requires_grad=False)
        self.gnn = EGNN(hidden_size, hidden_size, 1, l=l_uc, device='cuda', predict_force=True, normalize=True, tanh=True, z_size=latent_size)

        self.emb = nn.Linear(latent_size+1, hidden_size)

    
    def forward(self, x, z, pos, edge_index, batch, num_nodes):

        z = torch.index_select(z, 0, batch)
        h = torch.cat([x.view(x.size(0), 1), z], -1)
        h = self.emb(h)
        
        _, coords = self.gnn(h, pos, edge_index, None, num_nodes, z)
        
        diff = coords - pos
        diff = torch.where(diff > 0.5*self.l, diff-self.l, diff)
        diff = torch.where(diff < -0.5*self.l, diff+self.l, diff)
        
        return diff

class BasicVAE(nn.Module):

    def __init__(self, l_uc, hidden_size=32, latent_size=32):
        super(BasicVAE, self).__init__()

        self.l = nn.Parameter(l_uc, requires_grad=False)
        self.encoder = Encoder(l_uc, hidden_size, latent_size)
        self.decoder = Decoder(l_uc, hidden_size, latent_size)

        self.z = latent_size

    def forward(self, x, pos, edge_index, batch, num_nodes):

        mu, sigma = self.encoder(x.view(x.size(0), 1), pos, edge_index, batch, num_nodes)
        distribution = Normal(mu, sigma+1e-6)
        z = distribution.rsample()

        coord_diff = self.decoder(x, z, pos, edge_index, batch, num_nodes)

        return coord_diff, z, distribution

    def rec_loss(self, coord_diff):

            coord_diff = torch.where(coord_diff > 0.5*self.l, coord_diff-self.l, coord_diff) 
            coord_diff = torch.where(coord_diff < -0.5*self.l, coord_diff+self.l, coord_diff)
            loss = torch.sum(coord_diff).mean()
            
            # Loss - predicted coord diff vs actual coord diff from adding noise
            return loss

    def kld_loss(self, posterior, z):

            mu = torch.zeros_like(z)
            sigma = torch.ones_like(z)
            prior = Normal(mu, sigma)


            kld = (posterior.log_prob(z)-prior.log_prob(z)).sum(1).mean()

            return kld

    def training_step(self, databatch, beta, opt):
            opt.zero_grad()
            data = unpack_batch(databatch)
            coord_diff, z, dist = self.forward(*data)

            rec_loss = self.rec_loss(coord_diff)
            kld_loss = self.kld_loss(dist, z)


            loss = rec_loss + beta * kld_loss
            log_dict = {'kl':kld_loss.item(), 'rec':rec_loss.item(), 'loss':loss.item()}
            loss.backward()
            opt.step()

            return log_dict