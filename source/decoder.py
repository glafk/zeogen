import torch.nn as nn

MAX_ATOMIC_NUM = 100
from gemnet.gemnet import GemNetT


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim, final_activation=None):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]

    if final_activation == "sigmoid":
        mods.append(nn.Sigmoid())
    elif final_activation == "relu":
        mods.append(nn.ReLU())
    elif final_activation == "hard_sigmoid":
        mods.append(nn.Hardsigmoid())
    elif final_activation == "softmax":
        mods.append(nn.Softmax(dim=-1))  # softmax often needs a dimension argument
    elif final_activation == "selu":
        mods.append(nn.SELU())

    return nn.Sequential(*mods)


class GemNetTDecoder(nn.Module):
    """Decoder with GemNetT."""

    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        max_neighbors=20,
        radius=6.,
        scale_file=None,
    ):
        """
        Initializes the GemNetTDecoder class.

        Args:
            hidden_dim (int, optional): The hidden dimension of the model. Defaults to 128.
            latent_dim (int, optional): The latent dimension of the model. Defaults to 256.
            max_neighbors (int, optional): The maximum number of neighbors. Defaults to 20.
            radius (float, optional): The cutoff radius. Defaults to 6.0.
            scale_file (str, optional): The file path to the scale file. Defaults to None.

        Returns:
            None
        """
        super(GemNetTDecoder, self).__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors

        print("Instantiating gemnet model")
        self.gemnet = GemNetT(
            num_targets=1,
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            regress_forces=True,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            otf_graph=True,
            scale_file=scale_file,
        )
        self.fc_atom = build_mlp(hidden_dim, hidden_dim, 3, 2)
        # self.fc_atom = nn.Sequential(nn.Linear(hidden_dim, 2), nn.Sigmoid())

    def forward(self, z, pred_frac_coords, pred_atom_types, num_atoms,
                lengths, angles):
        """
        args:
            z: (N_cryst, num_latent)
            pred_frac_coords: (N_atoms, 3)
            pred_atom_types: (N_atoms, ), need to use atomic number e.g. H = 1
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3)
            angles: (N_cryst, 3)
        returns:
            atom_frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """
        # (num_atoms, hidden_dim) (num_crysts, 3)
        # Here the returns are from GemNet, the targets + the predictions from regress forces
        h, pred_cart_coord_diff = self.gemnet(
            z=z,
            frac_coords=pred_frac_coords,
            atom_types=pred_atom_types,
            num_atoms=num_atoms,
            lengths=lengths,
            angles=angles,
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
        )
        pred_atom_types = self.fc_atom(h)

        return pred_cart_coord_diff, pred_atom_types