from pathlib import Path
from typing import Any, Dict
import os


import env
import hydra
import numpy as np
import omegaconf
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm
import pickle
from utils import add_object
import torch.distributions as dist

from data_utils.crystal_utils import frac_to_cart_coords, cart_to_frac_coords, min_distance_sqr_pbc, mard, lengths_angles_to_volume

# Load environment variables
env.load_envs()

MAX_ATOMIC_NUM = 20
PROJECT_ROOT = Path(env.get_env("PROJECT_ROOT"))

ZEOLITE_CODES_MAPPING = {'DDRch1': 0, 'DDRch2': 1, 'FAU': 2, 
                         'FAUch': 3, 'ITW': 4, 'MEL': 5, 
                         'MELch': 6, 'MFI': 7, 'MOR': 8, 
                         'RHO': 9, 'TON': 10, 'TON2': 11, 
                         'TON3': 12, 'TON4': 13, 'TONch': 14, 
                         'BEC': 15, 'CHA': 16, 'ERI': 17, 
                         'FER': 18, 'HEU': 19, 'LTA': 20, 
                         'LTL': 21, 'MER': 22, 'MTW': 23, 
                         'NAT': 24, 'YFI': 25, "DDR": 26}

ZEOLITE_CODES_MAPPING_SMALL = {'DDR': 1, 'FAU': 2, 'ITW': 3, 'MEL': 4, 'MFI': 5, 'MOR': 6, 
                               'RHO': 7, 'TON': 8, 'BEC': 9, 'CHA': 10, 'ERI': 11, 'FER': 12, 
                               'HEU': 13, 'LTA': 14, 'LTL': 15, 'MER': 16, 'MTW': 17, 'NAT': 18, 'YFI': 19}

ZEOLITE_CODES_MAPPING_ALL_CODES = {'ACO': 1,
 'AEI': 2,
 'AEL': 3,
 'AEN': 4,
 'AET': 5,
 'AFG': 6,
 'AFI': 7,
 'AFN': 8,
 'AFO': 9,
 'AFR': 10,
 'AFS': 11,
 'AFT': 12,
 'AFV': 13,
 'AFX': 14,
 'AFY': 15,
 'AHT': 16,
 'ANO': 17,
 'APC': 18,
 'APD': 19,
 'AST': 20,
 'ASV': 21,
 'ATN': 22,
 'ATO': 23,
 'ATS': 24,
 'ATT': 25,
 'ATV': 26,
 'AVE': 27,
 'AVL': 28,
 'AWO': 29,
 'AWW': 30,
 'BCT': 31,
 'BIK': 32,
 'BOF': 33,
 'BOG': 34,
 'BOZ': 35,
 'BPH': 36,
 'BRE': 37,
 'BSV': 38,
 'CAN': 39,
 'CAS': 40,
 'CDO': 41,
 'CFI': 42,
 'CGF': 43,
 'CGS': 44,
 'CON': 45,
 'CSV': 46,
 'CZP': 47,
 'DAC': 48,
 'DFO': 49,
 'DFT': 50,
 'DOH': 51,
 'DON': 52,
 'EAB': 53,
 'EEI': 54,
 'EMT': 55,
 'EON': 56,
 'EOS': 57,
 'EPI': 58,
 'ESV': 59,
 'ETL': 60,
 'ETR': 61,
 'ETV': 62,
 'EUO': 63,
 'EWF': 64,
 'EWO': 65,
 'EWS': 66,
 'EZT': 67,
 'FAR': 68,
 'FRA': 69,
 'GIS': 70,
 'GIU': 71,
 'GME': 72,
 'GON': 73,
 'GOO': 74,
 'IFO': 75,
 'IFR': 76,
 'IFW': 77,
 'IFY': 78,
 'IHW': 79,
 'IRN': 80,
 'IRR': 81,
 'ISV': 82,
 'ITE': 83,
 'ITG': 84,
 'ITH': 85,
 'ITR': 86,
 'ITT': 87,
 'ITW': 88,
 'IWR': 89,
 'IWS': 90,
 'IWV': 91,
 'IWW': 92,
 'JNT': 93,
 'JOZ': 94,
 'JRY': 95,
 'JSN': 96,
 'JSR': 97,
 'JST': 98,
 'JSW': 99,
 'JSY': 100,
 'JZT': 101,
 'KFI': 102,
 'LAU': 103,
 'LEV': 104,
 'LIO': 105,
 'LOS': 106,
 'LOV': 107,
 'LTF': 108,
 'LTJ': 109,
 'MAR': 110,
 'MAZ': 111,
 'MEI': 112,
 'MEP': 113,
 'MFS': 114,
 'MON': 115,
 'MOZ': 116,
 'MRT': 117,
 'MSE': 118,
 'MSO': 119,
 'MTF': 120,
 'MTN': 121,
 'MTT': 122,
 'MVY': 123,
 'MWW': 124,
 'NAB': 125,
 'NES': 126,
 'NON': 127,
 'NPT': 128,
 'NSI': 129,
 'OBW': 130,
 'OFF': 131,
 'OKO': 132,
 'OSI': 133,
 'OSO': 134,
 'OWE': 135,
 'PCR': 136,
 'PHI': 137,
 'PON': 138,
 'POR': 139,
 'POS': 140,
 'PSI': 141,
 'PTF': 142,
 'PTO': 143,
 'PTT': 144,
 'PTY': 145,
 'PWW': 146,
 'RFE': 147,
 'RRO': 148,
 'RTE': 149,
 'RTH': 150,
 'RUT': 151,
 'RWR': 152,
 'RWY': 153,
 'SAF': 154,
 'SAO': 155,
 'SAS': 156,
 'SAT': 157,
 'SAV': 158,
 'SBE': 159,
 'SBN': 160,
 'SBS': 161,
 'SBT': 162,
 'SEW': 163,
 'SFE': 164,
 'SFF': 165,
 'SFG': 166,
 'SFH': 167,
 'SFN': 168,
 'SFO': 169,
 'SFS': 170,
 'SFW': 171,
 'SGT': 172,
 'SIV': 173,
 'SOD': 174,
 'SOF': 175,
 'SOR': 176,
 'SOS': 177,
 'SOV': 178,
 'SSF': 179,
 'SSY': 180,
 'STF': 181,
 'STI': 182,
 'STT': 183,
 'STW': 184,
 'SVV': 185,
 'SWY': 186,
 'SZR': 187,
 'TER': 188,
 'THO': 189,
 'TOL': 190,
 'TUN': 191,
 'UEI': 192,
 'UFI': 193,
 'UOS': 194,
 'UOV': 195,
 'UOZ': 196,
 'USI': 197,
 'UTL': 198,
 'UWY': 199,
 'VET': 200,
 'VFI': 201,
 'VNI': 202,
 'VSV': 203,
 'WEI': 204,
 'YUG': 205,
 'ZON': 206,
 'FER': 207,
 'BEC': 208,
 'FAU': 209,
 'DDR': 210,
 'MTW': 211,
 'MFI': 212,
 'TON': 213,
 'RHO': 214,
 'MEL': 215,
 'MOR': 216,
 'HEU': 217,
 'ERI': 218,
 'LTA': 219,
 'YFI': 220,
 'CHA': 221,
 'LTL': 222,
 'NAT': 223,
 'MER': 224}

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class CondPrior(nn.Module):
    def __init__(self, cond_dim, z_dim, embed=True, hoa_conditional=False):
        super(CondPrior, self).__init__()
        self.emb = nn.Embedding(len(ZEOLITE_CODES_MAPPING_ALL_CODES.keys()) + 1, 128)
        if embed:
            # The +1 is to account for the normalized HOA
            if hoa_conditional:
                self.fc1 = nn.Sequential(nn.Linear(128 + 1, z_dim, bias=False), nn.BatchNorm1d(z_dim), nn.ReLU())
            else:
                self.fc1 = nn.Sequential(nn.Linear(128, z_dim, bias=False), nn.BatchNorm1d(z_dim), nn.ReLU())
        else:
            if hoa_conditional:
                self.fc1 = nn.Sequential(nn.Linear(cond_dim + 1, z_dim, bias=False), nn.BatchNorm1d(z_dim), nn.ReLU())
            else:
                self.fc1 = nn.Sequential(nn.Linear(cond_dim, z_dim, bias=False), nn.BatchNorm1d(z_dim), nn.ReLU())

        self.fc21 = nn.Sequential(nn.Linear(z_dim, z_dim))
        self.fc22 = nn.Sequential(nn.Linear(z_dim, z_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, condition_frame, condition_hoa=None, embed=True, hoa_conditional=False):
        if embed:
            if hoa_conditional:
                condition = self.emb(condition_frame)
                #print(condition)
                #print(condition_hoa)
                #print(condition.shape)
                #print(condition_hoa.shape)
                condition = torch.cat([condition, condition_hoa], dim=1)
            else:
                condition = self.emb(condition_frame)
        
        hidden = self.fc1(condition)
        z_loc = self.fc21(hidden)
        z_log_var = self.fc22(hidden) + 1e-7

        return z_loc, z_log_var


# This class code is repeated in the GEMNet file. TODO: Remove repetition
class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}



class DiffusionModel(BaseModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.encoder = hydra.utils.instantiate(
            self.hparams.encoder, num_targets=self.hparams.latent_dim)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder)

        self.fc_mu = nn.Linear(self.hparams.latent_dim,
                               self.hparams.latent_dim)
        self.fc_var = nn.Linear(self.hparams.latent_dim,
                                self.hparams.latent_dim)

        self.fc_num_atoms = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                      self.hparams.fc_num_layers, self.hparams.max_atoms+1)
        self.fc_lattice = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                    self.hparams.fc_num_layers, 6)
        self.fc_composition = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                        self.hparams.fc_num_layers, MAX_ATOMIC_NUM)

        if self.hparams.predict_property:
            self.fc_property = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                         self.hparams.fc_num_layers, 1)

        sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_begin),
            np.log(self.hparams.sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        type_sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.type_sigma_begin),
            np.log(self.hparams.type_sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

        self.conditional = False
        self.hoa_conditional = False
        if self.hparams.conditional:
            if not self.hparams.hoa_conditional:
                self.pz = CondPrior(1, self.hparams.latent_dim, embed=True, hoa_conditional=False)
                self.conditional = True
                self.hoa_conditional = False
            else:
                self.pz = CondPrior(1, self.hparams.latent_dim, embed=True, hoa_conditional=True)
                self.conditional = True
                self.hoa_conditional = True
    

        # These are passed from the datamodule after both it and the model have been initialized
        self.lattice_scaler = None
        self.scaler = None

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch):
        """
        encode crystal structures to latents.
        """
        hidden = self.encoder(batch)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        log_var = torch.clamp(log_var, min=-5, max=5)
        z = self.reparameterize(mu, log_var)
        return mu, log_var, z, hidden

    def decode_stats(self, z, gt_num_atoms=None, gt_lengths=None, gt_angles=None,
                     teacher_forcing=False):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing.
        """
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(z)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, gt_num_atoms))
            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.predict_num_atoms(z).argmax(dim=-1)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, num_atoms))
            composition_per_atom = self.predict_composition(z, num_atoms)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom

    def forward(self, batch, teacher_forcing=False, training=False):
        # hacky way to resolve the NaN issue. Will need more careful debugging later.
        mu, log_var, z, hidden = self.encode(batch)

        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
         pred_composition_per_atom) = self.decode_stats(
            z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing)

        # sample noise levels.
        noise_level = torch.randint(0, self.sigmas.size(0),
                                    (batch.num_atoms.size(0),),
                                    device=self.device)
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            batch.num_atoms, dim=0)

        type_noise_level = torch.randint(0, self.type_sigmas.size(0),
                                         (batch.num_atoms.size(0),),
                                         device=self.device)
        used_type_sigmas_per_atom = (
            self.type_sigmas[type_noise_level].repeat_interleave(
                batch.num_atoms, dim=0))

        # THIS BIT IS IMPORTANT. WILL NEED THIS FOR MY INITIAL DIFFUSION PROCESS DEVELOPMENT
        # add noise to atom types and sample atom types.
        pred_composition_probs = F.softmax(
            pred_composition_per_atom.detach(), dim=-1)
        atom_type_probs = (
            F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM) +
            pred_composition_probs * used_type_sigmas_per_atom[:, None])
        try:
            rand_atom_types = torch.multinomial(
                atom_type_probs, num_samples=1).squeeze(1) + 1
        except Exception as e:
            error_obj = {
                "batch": batch,
                "hidden": hidden,
                "z": z,
                "mu": mu,
                "log_var": log_var,
                "atom_type_probs": atom_type_probs
            }

            with open("instable_types.pickle", "wb") as f:
                pickle.dump(error_obj)

        # add noise to the cart coords
        # TODO: Investigate: is it needed to add noise to the cart coords?
        # Can we just add noise to the frac coords?
        # Then we wouldn't need the predictions for angles and lengths
        cart_noises_per_atom = (
            torch.randn_like(batch.frac_coords) *
            used_sigmas_per_atom[:, None])
        cart_coords = frac_to_cart_coords(
            batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms)
        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(
            cart_coords, pred_lengths, pred_angles, batch.num_atoms)

        # THIS IS WHERE THE DECODER IS CALLED AS PART OF THE FORWARD PASS
        # THE pred_cart_coord_diff is the prediction for the difference in the atom coords based on the noise that is added
        # SO HERE I NEED TO SETUP A NETWORK WITH THIS MODEL'S DECODER SO THAT I CAN REPLICATE THE DIFFUSION PROCESS, BUT WITHOUT ANYTHING ELSE FROM THIS MODEL FOR NOW
        # TODO: Ask at the meeting: Can we pass the angles and lengths from the ground truth to the decoder?
        # Would that make the training better?
        try:
            pred_cart_coord_diff, pred_atom_types = self.decoder(
                z, noisy_frac_coords, rand_atom_types, batch.num_atoms, pred_lengths, pred_angles)
        except Exception as e:
            # Handle the case where atoms have 0 neighors in the computational graph 
            # and the forward pass fails
            # Pass the ground truths to the decoder
            pred_cart_coord_diff, pred_atom_types = self.decoder(z, noisy_frac_coords, rand_atom_types, batch.num_atoms, batch.lengths, batch.angles)

        if self.conditional:
            # print(batch["zeolite_code_enc"])
            p_mu, p_log_var = self.pz(batch['zeolite_code_enc'], batch["norm_hoa"], embed=True, hoa_conditional=self.hoa_conditional)

        # compute loss.
        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types, batch)
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)
        type_loss = self.type_loss(pred_atom_types, batch.atom_types,
                                   used_type_sigmas_per_atom, batch)

        if self.conditional:
            kld_loss = self.kld_loss(mu, log_var, p_mu, p_log_var)
        else:
            kld_loss = self.kld_loss(mu, log_var)

        if self.hparams.predict_property:
            property_loss = self.property_loss(z, batch)
        else:
            property_loss = 0.

        return {
            'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            'composition_loss': composition_loss,
            'coord_loss': coord_loss,
            'type_loss': type_loss,
            'kld_loss': kld_loss,
            'property_loss': property_loss,
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'pred_atom_types': pred_atom_types,
            'pred_composition_per_atom': pred_composition_per_atom,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            'rand_atom_types': rand_atom_types,
            'z': z,
        }

    def generate_rand_init(self, pred_composition_per_atom, pred_lengths,
                           pred_angles, num_atoms, batch):
        rand_frac_coords = torch.rand(num_atoms.sum(), 3,
                                      device=num_atoms.device)
        pred_composition_per_atom = F.softmax(pred_composition_per_atom,
                                              dim=-1)
        rand_atom_types = self.sample_composition(
            pred_composition_per_atom, num_atoms)
        return rand_frac_coords, rand_atom_types

    def sample_composition(self, composition_prob, num_atoms):
        """
        Samples composition such that it exactly satisfies composition_prob
        """
        batch = torch.arange(
            len(num_atoms), device=num_atoms.device).repeat_interleave(num_atoms)
        assert composition_prob.size(0) == num_atoms.sum() == batch.size(0)
        composition_prob = scatter(
            composition_prob, index=batch, dim=0, reduce='mean')

        all_sampled_comp = []

        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            comp_num = torch.round(comp_prob * num_atom)
            atom_type = torch.nonzero(comp_num, as_tuple=True)[0] + 1
            atom_num = comp_num[atom_type - 1].long()

            sampled_comp = atom_type.repeat_interleave(atom_num, dim=0)

            # if the rounded composition gives less atoms, sample the rest
            if sampled_comp.size(0) < num_atom:
                left_atom_num = num_atom - sampled_comp.size(0)

                left_comp_prob = comp_prob - comp_num.float() / num_atom

                left_comp_prob[left_comp_prob < 0.] = 0.
                left_comp = torch.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True)
                # convert to atomic number
                left_comp = left_comp + 1
                sampled_comp = torch.cat([sampled_comp, left_comp], dim=0)

            sampled_comp = sampled_comp[torch.randperm(sampled_comp.size(0))]
            sampled_comp = sampled_comp[:num_atom]
            all_sampled_comp.append(sampled_comp)

        all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_property(self, z):
        self.scaler.match_device(z)
        return self.scaler.inverse_transform(self.fc_property(z))

    def predict_lattice(self, z, num_atoms):
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        # TODO: Reverser the changes so that lattice_scale_method is an attribute and not an indexer
        if self.hparams.data["lattice_scale_method"] == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float()**(1/3)
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom


    @torch.no_grad()
    def langevin_dynamics(self, z, ld_kwargs, gt_num_atoms=None, gt_atom_types=None):
        """
        decode crystral structure from latent embeddings.
        ld_kwargs: args for doing annealed langevin dynamics sampling:
            n_step_each:  number of steps for each sigma level.
            step_lr:      step size param.
            min_sigma:    minimum sigma to use in annealed langevin dynamics.
            save_traj:    if <True>, save the entire LD trajectory.
            disable_bar:  disable the progress bar of langevin dynamics.
        gt_num_atoms: if not <None>, use the ground truth number of atoms.
        gt_atom_types: if not <None>, use the ground truth atom types.
        """
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        # obtain key stats.
        num_atoms, _, lengths, angles, composition_per_atom = self.decode_stats(
            z, gt_num_atoms)
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # obtain atom types.
        composition_per_atom = F.softmax(composition_per_atom, dim=-1)
        if gt_atom_types is None:
            cur_atom_types = self.sample_composition(
                composition_per_atom, num_atoms)
        else:
            cur_atom_types = gt_atom_types

        # init coords.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)

        # annealed langevin dynamics.
        print(f"Langevin dynamics sigmas...", self.sigmas)
        for sigma in tqdm(self.sigmas, total=self.sigmas.size(0), disable=ld_kwargs.disable_bar):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):
                noise_cart = torch.randn_like(
                    cur_frac_coords) * torch.sqrt(step_size * 2)
                pred_cart_coord_diff, pred_atom_types = self.decoder(
                    z, cur_frac_coords, cur_atom_types, num_atoms, lengths, angles) # lines 8,9 of pseudocode
                cur_cart_coords = frac_to_cart_coords(
                    cur_frac_coords, lengths, angles, num_atoms)
                pred_cart_coord_diff = pred_cart_coord_diff / sigma
                cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart # line 11 in psedocode
                cur_frac_coords = cart_to_frac_coords(
                    cur_cart_coords, lengths, angles, num_atoms)

                if gt_atom_types is None:
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 1

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_pred_cart_coord_diff.append(
                        step_size * pred_cart_coord_diff)
                    all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        output_dict = {'z': z.cpu().numpy(),
                       'num_atoms': num_atoms.cpu().numpy(), 
                       'lengths': lengths.cpu().numpy(), 
                       'angles': angles.cpu().numpy(),
                       'frac_coords': cur_frac_coords.cpu().numpy(),
                       'atom_types': cur_atom_types.cpu().numpy(),
                       'is_traj': False}

        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=torch.stack(all_frac_coords, dim=0).cpu().numpy(),
                all_atom_types=torch.stack(all_atom_types, dim=0).cpu().numpy(),
                # all_pred_cart_coord_diff=torch.stack(
                #     all_pred_cart_coord_diff, dim=0),
                # all_noise_cart=torch.stack(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict

    def sample(self, num_samples, ld_kwargs, save_samples=False, samples_file="samples.pickle", domains=None, hoas=None):
        # Here in the sampling part I will need to figure out how to force the model to sample from the part of the distribution where the representations of the "high-capacity" crystals lie
        if self.conditional:
            if self.hoa_conditional:
                assert num_samples == len(hoas)
                zs = []
                for domain in domains:
                    for hoa in hoas:
                        z_mu, z_log_var = self.pz(torch.tensor([ZEOLITE_CODES_MAPPING_ALL_CODES[domain]], device=self.device), 
                                                torch.tensor([hoa], device=self.device), 
                                                embed=True)
                        pz = dist.Normal(z_mu.squeeze(), z_log_var.exp().squeeze())
                        sample_n = pz.sample((1,))
                        zs.append(sample_n)

                z = torch.cat(zs)
            else:
                zs = []
                for domain in domains:
                    z_mu, z_log_var = self.pz(torch.tensor([ZEOLITE_CODES_MAPPING_ALL_CODES[domain]], device=self.device), 
                                            torch.tensor([hoa], device=self.device), 
                                            embed=True, hoa_conditional=False)
                    pz = dist.Normal(z_mu.squeeze(), z_log_var.exp().squeeze())
                    sample_n = pz.sample((1,))
                    zs.append(sample_n)

                z = torch.cat(zs)
        else:
            print(f"Saving sampled crystals - {save_samples}.")
            z = torch.randn(num_samples, self.hparams.latent_dim,
                            device=self.device)
        
        samples = self.langevin_dynamics(z, ld_kwargs)

        # if self.conditional:
        #     domains_list = [domain for domain in domains for _ in range(num_samples)]
        #     for i in range(len(domains_list)):
        #         samples[i]["domain"] = domains_list[i]

        if save_samples:
            print(f"Saving samples to {samples_file}.")
            with open(os.path.join(f"{PROJECT_ROOT}/samples", samples_file), "wb") as f:
                pickle.dump(samples, f)

        return samples

    def reconstruct(self, batch, ld_kwargs, reconstructions_file="reconstructions.pickle"):
        # Reconstruct materials from dataset sample
        mu, log_var, z, hidden = self.encode(batch)

        reconstruction = self.langevin_dynamics(z, ld_kwargs)

        print(f"Saving reconstructions to {reconstructions_file}.")
        reconstructions_path = os.path.join(f"{PROJECT_ROOT}/reconstructions", reconstructions_file)
        gt_path = os.path.join(f"{PROJECT_ROOT}/reconstructions", reconstructions_file.split('.')[0] + "_gt.pickle")

        add_object(reconstruction, reconstructions_path)
        add_object(batch, gt_path)

    def num_atom_loss(self, pred_num_atoms, batch):
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    def property_loss(self, z, batch):
        return F.l1_loss(self.fc_property(z), batch.y)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        # TODO: Here as well
        if self.hparams.data["lattice_scale_method"] == 'scale_length':
            target_lengths = batch.lengths / \
                batch.num_atoms.view(-1, 1).float()**(1/3)
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        # print(pred_composition_per_atom.shape)
        # print(f"Target atom types shape: {target_atom_types.shape}")
        # print(f"Batch.batch dimensions: {batch.batch.shape}")
        batch_cpu = batch.batch.cpu()
        # print(f"Min batch.batch {batch_cpu.min()}")
        # print(f"Max batch.batch {batch_cpu.max()}")
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        # print(f"Loss shape {loss.shape}")
        return scatter(loss, batch.batch, reduce='mean').mean()

    def coord_loss(self, pred_cart_coord_diff, noisy_frac_coords,
                   used_sigmas_per_atom, batch):
        noisy_cart_coords = frac_to_cart_coords(
            noisy_frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, noisy_cart_coords, batch.lengths, batch.angles,
            batch.num_atoms, self.device, return_vector=True)

        target_cart_coord_diff = target_cart_coord_diff / \
            used_sigmas_per_atom[:, None]**2
        pred_cart_coord_diff = pred_cart_coord_diff / \
            used_sigmas_per_atom[:, None]

        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def type_loss(self, pred_atom_types, target_atom_types,
                  used_type_sigmas_per_atom, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, reduction='none')
        # rescale loss according to noise
        loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.batch, reduce='mean').mean()

    def kld_loss(self, mu1, log_var1, mu2=None, log_var2=None):
        if mu2 is not None and log_var2 is not None:
            var1 = log_var1.exp()  # Variance of q1
            var2 = log_var2.exp()  # Variance of q2

            kld = 0.5 * torch.sum(
                log_var2 - log_var1
                - 1
                + var1 / var2
                + (mu2 - mu1).pow(2) / var2,
                dim=1  # Sum over dimensions of the latent space
            )
            
            return kld.mean()  # Mean over the batch
        else:
            kld_loss = torch.mean(
                -0.5 * torch.sum(1 + log_var1 - mu1**2 - log_var1.exp(), dim=1), dim=0)
            
            return kld_loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch)
        outputs = self(batch, teacher_forcing, training=True)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, batch, outputs, prefix):
        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        composition_loss = outputs['composition_loss']
        property_loss = outputs['property_loss']

        loss = (
            self.hparams.cost_natom * num_atom_loss +
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_coord * coord_loss +
            self.hparams.cost_type * type_loss +
            self.hparams.beta * kld_loss +
            self.hparams.cost_composition * composition_loss +
            self.hparams.cost_property * property_loss)

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
            f'{prefix}_composition_loss': composition_loss,
        }

        if prefix != 'train':
            # validation/test loss only has coord and type
            loss = (
                self.hparams.cost_coord * coord_loss +
                self.hparams.cost_type * type_loss)

            # evaluate num_atom prediction.
            pred_num_atoms = outputs['pred_num_atoms'].argmax(dim=-1)
            num_atom_accuracy = (
                pred_num_atoms == batch.num_atoms).sum() / batch.num_graphs

            # evalute lattice prediction.
            pred_lengths_and_angles = outputs['pred_lengths_and_angles']
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles)
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.data.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * \
                    batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)

            # evaluate atom type prediction.
            pred_atom_types = outputs['pred_atom_types']
            target_atom_types = outputs['target_atom_types']
            type_accuracy = pred_atom_types.argmax(
                dim=-1) == (target_atom_types - 1)
            type_accuracy = scatter(type_accuracy.float(
            ), batch.batch, dim=0, reduce='mean').mean()

            log_dict.update({
                f'{prefix}_loss': loss,
                f'{prefix}_property_loss': property_loss,
                f'{prefix}_natom_accuracy': num_atom_accuracy,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_volumes_mard': volumes_mard,
                f'{prefix}_type_accuracy': type_accuracy,
            })

        return log_dict, loss