import torch.nn as nn

from gemnet.gemnet import GemNetT

class GemNetTEncoderExt(nn.Module):
    """Wrapper for GemNetT."""

    def __init__(
        self,
        num_targets,
        hidden_size,
        otf_graph=False,
        radius=6.0,
        max_neighbors=20,
        scale_file=None,
    ):
        super(GemNetTEncoderExt, self).__init__()
        self.num_targets = num_targets
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors
        self.otf_graph = otf_graph

        self.gemnet = GemNetT(
            num_targets=num_targets,
            latent_dim=0,
            emb_size_atom=hidden_size,
            emb_size_edge=hidden_size,
            regress_forces=False,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            otf_graph=self.otf_graph,
            scale_file=scale_file,
            # Setting the activation to tanh to prevent range of latent representation being too large
            activation="tanh"
        )

        self.fc_mu = nn.Linear(num_targets, num_targets)
        self.fc_var = nn.Sequential(nn.Linear(num_targets, num_targets), nn.Softplus())

    def forward(self, data):
        # (num_crysts, num_targets)
        hidden = self.gemnet(
            z=None,
            frac_coords=data.frac_coords,
            atom_types=data.atom_types,
            num_atoms=data.num_atoms,
            lengths=data.lengths,
            angles=data.angles,
            edge_index=data.edge_index,
            to_jimages=data.to_jimages,
            num_bonds=data.num_bonds
        )

        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden) + 1e-5

        # Here hidden is returned for debugging purposes
        return mu, log_var, hidden