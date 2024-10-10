import gc
from pathlib import Path
from typing import Any, Dict
from pytorch_lightning.utilities.exceptions import MisconfigurationException

import env
import hydra
import numpy as np
import omegaconf
import pickle
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
from torch_scatter import scatter
from tqdm import tqdm
import torch.distributions as dist

from utils import add_object, log_config_to_wandb, masked_bce_loss
from data_utils.crystal_utils import frac_to_cart_coords, cart_to_frac_coords, min_distance_sqr_pbc, mard, lengths_angles_to_volume

# Load environment variables
env.load_envs()

PROJECT_ROOT = Path(env.get_env("PROJECT_ROOT"))

ZEOLITE_CODES_MAPPING = {'DDRch1': 0, 'DDRch2': 1, 'FAU': 2, 'FAUch': 3, 'ITW': 4, 'MEL': 5, 'MELch': 6, 'MFI': 7, 'MOR': 8, 'RHO': 9, 'TON': 10, 'TON2': 11, 'TON3': 12, 'TON4': 13, 'TONch': 14, 'BEC': 15, 'CHA': 16, 'ERI': 17, 'FER': 18, 'HEU': 19, 'LTA': 20, 'LTL': 21, 'MER': 22, 'MTW': 23, 'NAT': 24, 'YFI': 25, "DDR": 26}

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


class CondPrior(nn.Module):
    def __init__(self, cond_dim, z_dim):
        super(CondPrior, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(cond_dim, z_dim, bias=False), nn.BatchNorm1d(z_dim), nn.ReLU())
        self.fc21 = nn.Sequential(nn.Linear(z_dim, z_dim))
        self.fc22 = nn.Sequential(nn.Linear(z_dim, z_dim), nn.Softplus())

        torch.nn.init.xavier_uniform_(self.fc1[0].weight)
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, condition):
        hidden = self.fc1(condition)
        z_loc = self.fc21(hidden)
        z_scale = self.fc22(hidden) + 1e-7

        return z_loc, z_scale


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



class CDiVAE_v3(BaseModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # print(self.hparams)
        self.zd_encoder = hydra.utils.instantiate(
            self.hparams.cdivae_v3["encoders"]["domain_encoder"], num_targets=self.hparams.domain_latent_dim) # DIVA -> self.qzd
        
        self.zy_encoder = hydra.utils.instantiate(
            self.hparams.cdivae_v3["encoders"]["class_encoder"], num_targets=self.hparams.class_latent_dim) # DIVA -> self.qzy

        self.pzd = CondPrior(1, self.hparams.domain_latent_dim)

        # Hard code one as y in this case would be the HOA which is just a scalar
        self.pzy = CondPrior(1, self.hparams.class_latent_dim)
        
        self.positions_decoder = hydra.utils.instantiate(self.hparams.cdivae_v3["decoders"]["positions_decoder"])
        self.types_decoder = hydra.utils.instantiate(self.hparams.cdivae_v3["decoders"]["types_decoder"])

        # DIVA - > self.fc_d -> self.qd. Predictor for the zeolite type from zd
        # TODO: Test this domain predictor without the final softmax activation
        # as the loss is calculated by cross_entropy which automatically applies softmax
        self.domain_predictor = build_mlp(self.hparams.domain_latent_dim, self.hparams.hidden_dim,
                                          self.hparams.fc_num_layers, self.hparams.num_zeolite_types)

        # Used to predict the mean of the HOA from the zd latent vector
        # Needed because the distribution of the HOA is different for different zeolite types
        # Actually I'm starting to question the need for this.
        # I may not need to predict neither the mu not the std
        # I can keep the generation of the zy latent space from the scaled hoa
        # separate from the zd latent space, while also generating the final
        # HOA prediction from the zd latent space and the zy latent space but not backpropagating through it.
        # so as to keep the learning on the scaled HOA which I will later need for the sampling
        self.hoa_mu_predictor = build_mlp(self.hparams.domain_latent_dim, self.hparams.hidden_dim,
                                self.hparams.fc_num_layers, 1)

        self.hoa_std_predictor = build_mlp(self.hparams.domain_latent_dim, self.hparams.hidden_dim,
                                self.hparams.fc_num_layers, 1)        

        self.norm_hoa_predictor = build_mlp(self.hparams.class_latent_dim, self.hparams.hidden_dim,
                                            self.hparams.fc_num_layers, 1)

        self.fc_num_atoms = build_mlp(self.hparams.domain_latent_dim, self.hparams.hidden_dim,
                                      self.hparams.fc_num_layers, self.hparams.max_atoms+1)
        self.fc_lengths = build_mlp(self.hparams.domain_latent_dim, self.hparams.hidden_dim,
                                    self.hparams.fc_num_layers, 3, final_activation="selu")
        self.fc_angles = build_mlp(self.hparams.domain_latent_dim, self.hparams.hidden_dim,
                                   self.hparams.fc_num_layers, 3, final_activation='sigmoid')
        self.fc_composition = build_mlp(self.hparams.class_latent_dim, self.hparams.hidden_dim,
                                        self.hparams.fc_num_layers, 1, final_activation="hard_sigmoid")

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

        # These are passed from the datamodule after both it and the model have been initialized
        self.lengths_scaler = None
        self.prop_scaler = None
        self.prop_mu_scaler = None
        self.prop_std_scaler = None

    # region ENCODE HELPERS
    def reparameterize(self, loc, scale):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        d = dist.Normal(loc, scale)

        return d

    def encode(self, batch):
        """
        encode crystal structures to latents.
        """
        # Comments providing mapping between the variables in this implementation
        # and the original DIVA implementation
        # DIVA - mu_d -> zd_q_loc, log_var_d -> zd_q_scale 
        zd_q_loc, zd_q_scale, hidden_d = self.zd_encoder(batch, uniform_types=True)
        qzd = self.reparameterize(zd_q_loc, zd_q_scale)
        zd = qzd.rsample()

        # DIVA - mu_y -> zy_q_loc, log_var_y -> zy_q_scale
        zy_q_loc, zy_q_scale, hidden_y = self.zy_encoder(batch)
        qzy = self.reparameterize(zy_q_loc, zy_q_scale)
        zy = qzy.rsample()

        z = torch.cat([zd, zy], dim=-1)

        # Temporarily save the hidden layer for debugging
        return zd_q_loc, zd_q_scale, zy_q_loc, zy_q_scale, hidden_d, hidden_y, zd, zy, z

    def decode_stats(self, zd, zy, gt_num_atoms=None, gt_lengths=None, gt_angles=None,
                     teacher_forcing=False, num_atoms_forcing=False):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing.
        """
        if gt_num_atoms is not None and teacher_forcing:
            num_atoms = self.predict_num_atoms(zd)
            lengths = self.predict_lenghts(zd, gt_num_atoms)
            angles = self.predict_angles(zd)
            lengths_and_angles = torch.cat([lengths, angles], dim=-1)
            # The new composition prediction would predict a tensor of size
            # [batch_size, max_atoms] so that for each crystal in the batch
            # there will be a prediction for each individual atom_num
            # To calculate the loss I will use a mask to take into account
            # only the number of atoms in each crystal, either ground truth 
            # or predicted 
            composition_per_crystal = self.predict_composition(zy, gt_num_atoms)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        # elif gt_num_atoms is not None and num_atoms_forcing:
        #     num_atoms = self.predict_num_atoms(zd)
        #     lengths = self.predict_lenghts(zd, gt_num_atoms)
        #     angles = self.predict_angles(zd)
        #     lengths_and_angles = torch.cat([lengths, angles], dim=-1)
        #     composition_per_crystal = self.predict_composition(zy, gt_num_atoms)
        else:
            num_atoms = self.predict_num_atoms(zd)
            lengths = self.predict_lenghts(zd, num_atoms.argmax(dim=-1))
            angles = self.predict_angles(zd)
            lengths_and_angles = torch.cat([lengths, angles], dim=-1)
            composition_per_crystal = self.predict_composition(zy, num_atoms.argmax(dim=-1))

        return num_atoms, lengths_and_angles, lengths, angles, composition_per_crystal

    # endregion

    def forward(self, batch, teacher_forcing=False, training=False):
        # region ENCODE
        (zd_q_loc, 
         zd_q_scale, 
         zy_q_loc, 
         zy_q_scale, 
         hidden_d, 
         hidden_y,
         zd, 
         zy,
         z) = self.encode(batch)

        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
         pred_si_ratio_per_crystal) = self.decode_stats(
            zd, zy , batch.num_atoms, batch.lengths, batch.angles, teacher_forcing)
        # endregion

        # region PREDICT

        # sample noise levels.
        noise_level = torch.randint(0, self.sigmas.size(0),
                                    (batch.num_atoms.size(0),),
                                    device=self.device)
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            batch.num_atoms, dim=0)

        # add noise to the cart coords
        cart_noises_per_atom = (
            torch.randn_like(batch.frac_coords) *
            used_sigmas_per_atom[:, None])
        cart_coords = frac_to_cart_coords(
            batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms)
        noisy_cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(
            noisy_cart_coords, pred_lengths, pred_angles, batch.num_atoms)

        # select random atoms according to the predicted composition
        # We do that since at inference time this is where the decoder
        # will have to start working from
        si_ratio_per_atom = torch.repeat_interleave(pred_si_ratio_per_crystal, batch.num_atoms, dim=0).squeeze(1)

        rand_atom_types = torch.multinomial(torch.stack((1 - si_ratio_per_atom, si_ratio_per_atom), dim=1), num_samples=1).squeeze(1) + 13
        try:
            # pred_cart_coord_diff is the prediction for the difference in the atom coords based on the noise that is added
            pred_cart_coord_diff, _ = self.positions_decoder(
                zd, noisy_frac_coords, rand_atom_types, batch.num_atoms, pred_lengths, pred_angles)
        except Exception as e:
            # Handle the case where atoms have 0 neighors in the computational graph 
            # and the forward pass fails
            # Pass the ground truths to the decoder
            # pred_cart_coord_diff, pred_atom_types = self.decoder(z, noisy_frac_coords, rand_atom_types, batch.num_atoms, batch.lengths, batch.angles)
            print("positions_exception", e)
            raise e

        # Before going to the second decoder, calucate the predicted positions of atoms
        # by adding the predicted cartesian coord diff to the original coords
        pred_cart_coords = cart_coords + pred_cart_coord_diff
        pred_frac_coords = cart_to_frac_coords(
            pred_cart_coords, pred_lengths, pred_angles, batch.num_atoms
        )

        try:
            # This variable will determine whether to sample noise according to the 
            # ground truth or the predicted number of atoms
            # num_atoms_for_noise = batch.num_atoms if num_atoms_forcing else pred_num_atoms
            # Select a random noise level for the atom types per atom in the batch
            type_noise_level = torch.randint(0, self.type_sigmas.size(0),
                                            (batch.num_atoms.size(0),),
                                            device=self.device)
            # used_type_sigmas_per_atom = (
            #     self.type_sigmas[type_noise_level].repeat_interleave(
            #         batch.num_atoms, dim=0))

            type_noise = self.type_sigmas[type_noise_level]
            type_noise = type_noise.repeat_interleave(batch.num_atoms, dim=0)

            # Generate random noise signs to ensure noise does not only increase the probabilit of Si atoms
            # random_signs = (torch.rand(batch.num_atoms.size(0), device=self.device) - 0.5).sign()
            # random_signs = random_signs.repeat_interleave(batch.num_atoms, dim=0)

            # Here we add noise to the original atom types, so that the decoder can learn to
            # push them back to the original atom types. Since the pred_compositon probs
            # have a shape of [batch_size, max_atoms] we need to slice only the for which we have 
            # ground truth atoms
            atom_type_probs = (F.one_hot(batch.atom_types - 13, num_classes=2) 
                + (torch.rand_like(type_noise.float()) * type_noise).unsqueeze(dim=1))
            
            # Clamp the atom_type_probs to ensure no probability going into the torch.multinomial sampling is negative
            # atom_type_probs = torch.clamp(atom_type_probs, max=1)
            # used_type_sigmas_per_atom = used_type_sigmas_per_atom * random_signs

            # Add noise to predicted ratios and clamp to [0, 1]
            # pred_composition_per_atom = pred_composition_per_atom * used_type_sigmas_per_atom[:, None]
            # Expand the predicted composition ratio to match the number of atoms per crystal in the batch
            # pred_composition_per_atom = pred_composition_ratio.squeeze()[batch.batch.squeeze()]
            # noisy_composition_per_atom = noisy_ratios[batch.batch.squeeze()]
            # pred_composition_per_atom = F.softmax(pred_composition_per_atom, dim=-1)

            # Adjust with 13 to end up with only Al and Si atoms
            noisy_atom_types = torch.multinomial(atom_type_probs, num_samples=1).squeeze(1) + 13
        except Exception as e:
            print("atoms_error", e)
            batch = {
                "original_batch": batch,
                "teacher_forcing": teacher_forcing,
                "hidden_d": hidden_d,
                "hidden_y": hidden_y,
                "zd": zd,
                "zy": zy,
                "type_noise_level": type_noise_level,
                "type_noise": type_noise,
                "pred_si_ratio_per_crystal": pred_si_ratio_per_crystal,
                "pred_lengths": pred_lengths,
                "pred_angles": pred_angles,
                "zeolite_code": batch.zeolite_code
            }
            # Save the batch
            with open(f"/home/dglavinkov/ondemand/zeogen/source/problem_batch_mu_sig_std.pkl", "wb") as f:
                pickle.dump(batch, f)

            # Save the model weights
            torch.save(self.state_dict(), f"/home/dglavinkov/ondemand/zeogen/source/model_weights_mu_sig_std.pth")

            raise e

        _, pred_atom_types = self.types_decoder(
                zy, pred_frac_coords, noisy_atom_types, batch.num_atoms, pred_lengths, pred_angles)

        # Predict domain and HOA
        domain_pred = self.domain_predictor(zd)
        hoa_mu_pred = self.hoa_mu_predictor(zd)
        hoa_std_pred = self.hoa_std_predictor(zd)
        norm_hoa_pred = self.norm_hoa_predictor(zy)       

        # Predict parameters of conditional distributions
        zd_p_loc, zd_p_scale = self.pzd(batch['zeolite_code_enc'].float().view(-1, 1))
        zy_p_loc, zy_p_scale = self.pzy(batch['norm_hoa'].view(-1, 1))
        # endregion

        return {
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'pred_atom_types': pred_atom_types,
            'pred_si_ratio_per_crystal': pred_si_ratio_per_crystal,
            # 'pred_composition_ratio': pred_composition_ratio,
            'used_sigmas_per_atom': used_sigmas_per_atom,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            'rand_atom_types': rand_atom_types,
            'type_noise': type_noise,
            'noisy_frac_coords': noisy_frac_coords,
            'zd_q_loc': zd_q_loc,
            'zd_q_scale': zd_q_scale,
            'zy_q_loc': zy_q_loc,
            'zy_q_scale': zy_q_scale,
            'z': z,
            'zd': zd,
            'zy': zy,
            'zd_p_loc': zd_p_loc,
            'zd_p_scale': zd_p_scale,
            'zy_p_loc': zy_p_loc,
            'zy_p_scale': zy_p_scale,
            'domain_pred': domain_pred,
            'hoa_mu_pred': hoa_mu_pred,
            'hoa_std_pred': hoa_std_pred,
            'norm_hoa_pred': norm_hoa_pred
        }

    # region PREDICT HELPERS
    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_lenghts(self, z, num_atoms):
        self.lengths_scaler.match_device(z)
        pred_lengths = self.fc_lengths(z)
        # Perform inverse transform to get the original lengths
        # This scaling operation causes the gradients to now flow
        # through the predictions, so instead I'll scale the targets
        # Initially it looked like this wasn't the issue as when we cat 
        # the lengths and angles, the angles still require gradients
        # so it looks like the entire tensor requires a gradient
        pred_lengths = self.lengths_scaler.inverse_transform_backprob_compat(pred_lengths)
        if self.hparams["lattice_scale_method"] == 'scale_length':
            # Scale according to number of atoms
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float()**(1/3)

        return pred_lengths
        
    def predict_angles(self, z):
        pred_angles = self.fc_angles(z)
        # Multiply each angles by 180
        pred_angles = pred_angles * 180
        return pred_angles

    def predict_composition(self, z, num_atoms):
        pred_composition_per_atom = self.fc_composition(z)
        return pred_composition_per_atom
    # endregion

    # region SAMPLING
    @torch.no_grad()
    def langevin_dynamics(self, zd, zy, ld_kwargs, domain, norm_hoas, pred_hoas, gt_num_atoms=None, gt_atom_types=None):
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

        # For saving the trajectory I'll skip logging the noise
        # and prediction cart coord diffs
        if ld_kwargs.save_traj:
            all_frac_coords = []
            # all_pred_cart_coord_diff = []
            # all_noise_cart = []
            all_atom_types = []

        # obtain key stats.
        num_atoms, _, lengths, angles, si_ratio_per_crystal = self.decode_stats(
            zd, zy, gt_num_atoms)
        
        num_atoms = num_atoms.argmax(dim=-1)    
        if gt_num_atoms is not None:
            num_atoms = gt_num_atoms

        # obtain atom types.

        si_ratio_per_atom = torch.repeat_interleave(si_ratio_per_crystal, num_atoms, dim=0).squeeze(1)

        cur_atom_types = torch.multinomial(torch.stack((1 - si_ratio_per_atom, si_ratio_per_atom), dim=1), num_samples=1).squeeze(1) + 13

        # init coords.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=zd.device)

        # annealed langevin dynamics.
        # print(f"Langevin dynamics sigmas...", self.sigmas)
        for sigma in tqdm(self.sigmas, total=self.sigmas.size(0), disable=ld_kwargs.disable_bar):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            for step in range(ld_kwargs.n_step_each):

                # Add noise to coords and predict the new coords
                noise_cart = torch.randn_like(
                    cur_frac_coords) * torch.sqrt(step_size * 2)
                pred_cart_coord_diff, _ = self.positions_decoder(
                    zd, cur_frac_coords, cur_atom_types, num_atoms, lengths, angles) # lines 8,9 of pseudocode
                cur_cart_coords = frac_to_cart_coords(
                    cur_frac_coords, lengths, angles, num_atoms)
                pred_cart_coord_diff = pred_cart_coord_diff / sigma
                # Noise added to cartesian coordinates
                cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart # line 11 in psedocode
                cur_frac_coords = cart_to_frac_coords(
                    cur_cart_coords, lengths, angles, num_atoms)

                # After predicting the coords, predict the atom types
                if gt_atom_types is None:
                    _, pred_atom_types = self.types_decoder(
                        zy, cur_frac_coords, cur_atom_types, num_atoms, lengths, angles)
                    cur_atom_types = torch.argmax(pred_atom_types, dim=1) + 13

                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    # all_pred_cart_coord_diff.append(
                    #     step_size * pred_cart_coord_diff)
                    # all_noise_cart.append(noise_cart)
                    all_atom_types.append(cur_atom_types)

        output_dict = {'zd': zd.cpu().numpy(), 'zy': zy.cpu().numpy(),
                       'num_atoms': num_atoms.cpu().numpy(), 'lengths': lengths.cpu().numpy(), 'angles': angles.cpu().numpy(),
                       'frac_coords': cur_frac_coords.cpu().numpy(), 'atom_types': cur_atom_types.cpu().numpy(),
                       'domains': [domain] * len(norm_hoas), 'norm_hoas': norm_hoas,
                       'pred_hoas': pred_hoas.cpu().numpy(),
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

    def sample(self, num_samples_per_domain, ld_kwargs, kwargs_conf_name, domains, norm_hoas):
        """
        Samples crystals and optionally saves them.

        Args:
            num_samples_per_domain (int): Number of samples to generate per domain.
            ld_kwargs (dict): Keyword arguments for the Langevin dynamics method.
            kwargs_conf_name (str, optional): Name for the WandB artifact for the config.
            domains (list): Domains for which to generate samples
            norm_hoas (list): Conditional normalized HOA for each samples 
                to be taken for each domain. Length should be equal to num_samples  

        Returns:
            samples (Tensor): The generated samples.
        """
        # Log the LD configuration
        log_config_to_wandb(ld_kwargs, kwargs_conf_name, auxiliary_config=True)

        # Make sure norm_hoas has the same length as num_samples
        # This way for each generated sample per zeolite type we can have different HOAs
        assert len(norm_hoas) == num_samples_per_domain

        # Here in the sampling part I will need to figure out how to force the model to sample from the part of the distribution where the representations of the "high-capacity" crystals lie
        
        print(f"Sampling {num_samples_per_domain} crystals per domain with the following HOAs: {norm_hoas}.")
        print(f"Domains: {domains}")

        all_samples = [] 
        for domain in domains:
            if len(domain.split('/')) == 1:
                print(f"Sampling domain: {domain}")
                # Here we are in the case where we condition on a single domain which we have seen
                zd_p_mu, zd_p_log_var = self.pzd(torch.tensor([ZEOLITE_CODES_MAPPING[domain]], device=self.device).float().view(-1, 1))
                zy_p_mu, zy_p_log_var = self.pzy(torch.tensor([norm_hoas], device=self.device).view(-1, 1)) 
                
                zd_p_std = torch.exp(0.5 * zd_p_log_var)
                pzd = dist.Normal(zd_p_mu, zd_p_std)
                zd = pzd.sample()
                zd_per_hoa = zd.repeat(num_samples_per_domain, 1)

                zdp_y_std = torch.exp(0.5 * zy_p_log_var)
                pzy = dist.Normal(zy_p_mu, zdp_y_std)
                zy = pzy.sample()
                
                hoa_mu_pred = self.hoa_mu_predictor(zd_per_hoa)
                hoa_mu_pred = self.prop_mu_scaler.inverse_transform(hoa_mu_pred)
                hoa_std_pred = self.hoa_std_predictor(zd_per_hoa)
                hoa_std_pred = self.prop_std_scaler.inverse_transform(hoa_std_pred)
                norm_hoa_pred = self.norm_hoa_predictor(zy)  
                pred_hoas = norm_hoa_pred * hoa_std_pred + hoa_mu_pred
                samples = self.langevin_dynamics(zd_per_hoa, zy, ld_kwargs, domain, norm_hoas, pred_hoas)
                all_samples.append(samples)
            else:
                domain = domain.split('/')
                # Here we are in the case where we condition on multiple domains and interpolate between them
                zd_p_mu_1, zd_p_log_var_1 = self.pzd(torch.tensor([ZEOLITE_CODES_MAPPING[domain[0]]], device=self.device).float().view(-1, 1))
                zd_p_mu_2, zd_p_log_var_2 = self.pzd(torch.tensor([ZEOLITE_CODES_MAPPING[domain[1]]], device=self.device).float().view(-1, 1))
                # For now only interpolate with a weight of 0.5. We could do something more sophisticated later
                # like interpolating with a weight of 0.1, 0.3, 0.6, 0.9

                zd_p_std_1 = torch.exp(0.5 * zd_p_log_var_1)
                pzd = dist.Normal(zd_p_mu_1, zd_p_std_1)
                zd_1 = pzd.sample()

                zd_p_std_2 = torch.exp(0.5 * zd_p_log_var_2)
                pzd = dist.Normal(zd_p_mu_2, zd_p_std_2)
                zd_2 = pzd.sample()

                zd_interpolated = torch.lerp(zd_1, zd_2, 0.5)
                zd_per_hoa = zd_interpolated.repeat(num_samples_per_domain, 1)

                zdp_y_std = torch.exp(0.5 * zy_p_log_var)
                pzy = dist.Normal(zy_p_mu, zdp_y_std)
                zy = pzy.sample()

                hoa_mu_pred = self.hoa_mu_predictor(zd_per_hoa)
                hoa_mu_pred = self.prop_mu_scaler.inverse_transform(hoa_mu_pred)
                hoa_std_pred = self.hoa_std_predictor(zd_per_hoa)
                hoa_std_pred = self.prop_std_scaler.inverse_transform(hoa_std_pred)
                norm_hoa_pred = self.norm_hoa_predictor(zy)
                pred_hoas = norm_hoa_pred * hoa_std_pred + hoa_mu_pred

                samples = self.langevin_dynamics(zd_per_hoa, zy, ld_kwargs, domain, norm_hoas, pred_hoas)
                all_samples.append(samples)

        return all_samples   

    # TODO: Refactor recosntruction 
    def reconstruct(self, batch, ld_kwargs, reconstructions_path, ground_truth_path, kwargs_conf_name):
        """
        Reconstructs materials from a dataset sample using the Langevin dynamics method.

        Args:
            batch (torch.Tensor): The input batch of data.
            ld_kwargs (dict): The keyword arguments for the langeevin dynamics method.
            reconstructions_file (str, optional): The file name to save the reconstructions. Defaults to "reconstructions.pickle".
            save_to_wandb (bool, optional): Whether to save the reconstructions to wandb. Defaults to False.

        Returns:
            None
        """
        # Reconstruct materials from dataset sample
        mu, log_var, z = self.encode(batch)

        reconstruction = self.langevin_dynamics(z, ld_kwargs)

        add_object(reconstruction, reconstructions_path)
        add_object(batch, ground_truth_path)
    # endregion
    
    # region LOSSES
    def num_atom_loss(self, pred_num_atoms, batch):
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    # def property_loss(self, z, batch):
    #     return F.mse_loss(self.fc_property(z), batch.y)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lengths_scaler.match_device(pred_lengths_and_angles)
        
        # Scaling the target instead of the predictions to avoid
        # the problem with the gradients not flowing
        # if self.hparams.data["lattice_scale_method"] == 'scale_length':
        #     target_lengths = batch.lengths / \
        #         batch.num_atoms.view(-1, 1).float()**(1/3)
        # scaled_target_lengths = self.lengths_scaler.transform(target_lengths)
        target_lengths_and_angles = torch.cat(
             [batch.lengths, batch.angles], dim=-1)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_si_ratio_per_crystal, batch):
        # Cast target atom types to float
        targets = batch.atom_types.float() - 13
        targets = scatter(targets, batch.batch, reduce="mean")
    
        loss = F.l1_loss(pred_si_ratio_per_crystal.squeeze(), targets, reduction="mean")

        return loss

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

        # Average the error in each dimension of 3D space
        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def type_loss(self, pred_atom_types, target_atom_types,
                  type_noise, batch):
        target_atom_types = target_atom_types - 13
        # Define weights for the weighted BCE loss
        # Needed because the Al atoms are much more common in the dataset
        atom_type_weights = torch.tensor([0.75, 0.25]).to(self.device) 
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, weight=atom_type_weights, reduction='none')
        # TODO: try to train without rescaling to see if it improves training
        # Leaving it like this rn to see if the other changes I made would have an effect 
        # on this loss
        loss = loss / type_noise
        # rescale loss according to noise
        # loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.batch, reduce='mean').mean()

    def kld_loss(self, q_loc, q_scale, p_loc, p_scale, z):
        q = dist.Normal(q_loc, q_scale)
        p = dist.Normal(p_loc, p_scale)

        kld = -torch.sum(p.log_prob(z) - q.log_prob(z), dim=-1)

        return torch.mean(kld, dim=0)

    def domain_pred_loss(self, pred_domain_logits, batch):
        return F.cross_entropy(pred_domain_logits, batch.zeolite_code_enc)

    def norm_hoa_pred_loss(self, pred_norm_hoa, batch):
        return F.l1_loss(pred_norm_hoa, batch.norm_hoa)

    def hoa_mu_pred_loss(self, pred_hoa_mu, batch):
        pred_hoa_mu = self.prop_mu_scaler.inverse_transform_backprob_compat(pred_hoa_mu)
        return F.l1_loss(pred_hoa_mu, batch.hoa_mu)
    
    def hoa_std_pred_loss(self, pred_hoa_std, batch):
        pred_hoa_std = self.prop_std_scaler.inverse_transform_backprob_compat(pred_hoa_std)
        return F.l1_loss(pred_hoa_std, batch.hoa_std)

    @torch.no_grad()
    def final_hoa_loss(self, pred_hoa, batch):
        pred_hoa = self.prop_scaler.inverse_transform_backprob_compat(pred_hoa)
        return F.l1_loss(pred_hoa, batch.hoa)

    def compute_loss(self, batch, outputs, prefix):
        pred_num_atoms = outputs['pred_num_atoms']
        pred_lengths_and_angles = outputs['pred_lengths_and_angles']
        pred_si_ratio_per_crystal = outputs['pred_si_ratio_per_crystal']
        pred_cart_coord_diff = outputs['pred_cart_coord_diff']
        pred_atom_types = outputs['pred_atom_types']
        noisy_frac_coords = outputs['noisy_frac_coords']
        used_sigmas_per_atom = outputs['used_sigmas_per_atom']
        type_noise = outputs['type_noise']
        zd_q_loc = outputs['zd_q_loc']
        zd_q_scale = outputs['zd_q_scale']
        zy_q_loc = outputs['zy_q_loc']
        zy_q_scale = outputs['zy_q_scale']
        zd_p_loc = outputs['zd_p_loc']
        zd_p_scale = outputs['zd_p_scale']
        zy_p_loc = outputs['zy_p_loc']
        zy_p_scale = outputs['zy_p_scale']
        zd = outputs['zd']
        zy = outputs['zy']
        domain_pred = outputs['domain_pred']
        hoa_mu_pred = outputs['hoa_mu_pred']
        hoa_std_pred = outputs['hoa_std_pred']
        norm_hoa_pred = outputs['norm_hoa_pred']

        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(
            pred_si_ratio_per_crystal, batch)
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)
        type_loss = self.type_loss(pred_atom_types, batch.atom_types,
                                   type_noise, batch)

        kld_loss_d = self.kld_loss(zd_q_loc, zd_q_scale, zd_p_loc, zd_p_scale, zd)
        kld_loss_y = self.kld_loss(zy_q_loc, zy_q_scale, zy_p_loc, zy_p_scale, zy)
        kld_loss = kld_loss_d + kld_loss_y

        domain_pred_loss = self.domain_pred_loss(domain_pred, batch)
        norm_hoa_pred_loss = self.norm_hoa_pred_loss(norm_hoa_pred, batch)
        hoa_mu_pred_loss = self.hoa_mu_pred_loss(hoa_mu_pred, batch)
        hoa_std_pred_loss = self.hoa_std_pred_loss(hoa_std_pred, batch)

        loss = (
            self.hparams.cost_natom * num_atom_loss +
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_coord * coord_loss +
            self.hparams.cost_type * type_loss +
            self.hparams.beta * kld_loss +
            self.hparams.cost_composition * composition_loss +
            self.hparams.cost_domain * domain_pred_loss +
            self.hparams.cost_norm_hoa * norm_hoa_pred_loss +
            self.hparams.cost_hoa_mu * hoa_mu_pred_loss +
            self.hparams.cost_hoa_std * hoa_std_pred_loss
        )

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_composition_loss': composition_loss,
            f'{prefix}_kld_loss': kld_loss,
            f'{prefix}_kld_loss_d': kld_loss_d,
            f'{prefix}_kld_loss_y': kld_loss_y,
            f'{prefix}_norm_hoa_pred_loss': norm_hoa_pred_loss,
            f'{prefix}_domain_pred_loss': domain_pred_loss,
            f'{prefix}_hoa_mu_pred_loss': hoa_mu_pred_loss,
            f'{prefix}_hoa_std_pred_loss': hoa_std_pred_loss
        }

        if prefix != 'train':
            # validation/test loss only has coord and type
            loss = (
                self.hparams.cost_coord * coord_loss +
                self.hparams.cost_type * type_loss)

            # evaluate num_atom prediction.
            # TODO: CHECK IF SOFTMAX IS NEEDED BEFORE ALL ARGMAX FUNCTIONS
            pred_num_atoms = outputs['pred_num_atoms'].argmax(dim=-1)
            num_atom_accuracy = (
                pred_num_atoms == batch.num_atoms).sum() / batch.num_graphs

            # scaled_lengths = self.lengths_scaler.inverse_transform(pred_lengths_and_angles[:, :3])
            pred_lengths = pred_lengths_and_angles[:, :3]
            pred_angles = pred_lengths_and_angles[:, 3:]

            # The lengths are already scaled during the prediction step
            # if self.hparams.data.lattice_scale_method == 'scale_length':
            #     pred_lengths = pred_lengths * \
            #         batch.num_atoms.view(-1, 1).float()**(1/3)
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
                dim=-1) == (target_atom_types - 13)
            type_accuracy = scatter(type_accuracy.float(
            ), batch.batch, dim=0, reduce='mean').mean()

            # Evaluate aluminum atom predictions
            al_mask = target_atom_types == 13

            al_type_accuracy = pred_atom_types.argmax(dim=-1)[al_mask] == (target_atom_types[al_mask] - 13)

            # Calculate the mean accuracy over the selected atoms and scatter to the batch level
            al_type_accuracy = scatter(al_type_accuracy.float(), batch.batch[al_mask], dim=0, reduce='mean').mean()

            # Evaluate predicted domains
            domain_accuracy = domain_pred.argmax(dim=-1) == batch.zeolite_code_enc
            domain_accuracy = domain_accuracy.float().mean()

            
            # Evaluate final HOA prediction loss
            hoa_pred = norm_hoa_pred * hoa_std_pred + hoa_mu_pred
            final_hoa_loss = self.final_hoa_loss(hoa_pred, batch)

            log_dict.update({
                f'{prefix}_norm_hoa_pred_loss': norm_hoa_pred_loss,
                f'{prefix}_domain_pred_loss': domain_pred_loss,
                f'{prefix}_hoa_mu_pred_loss': hoa_mu_pred_loss,
                f'{prefix}_hoa_std_pred_loss': hoa_std_pred_loss,
                f'{prefix}_final_hoa_loss': final_hoa_loss,
                f'{prefix}_loss': loss,
                f'{prefix}_natom_accuracy': num_atom_accuracy,
                f'{prefix}_domain_accuracy': domain_accuracy,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_volumes_mard': volumes_mard,
                f'{prefix}_type_accuracy': type_accuracy,
                f'{prefix}_al_type_accuracy': al_type_accuracy,
            })

        return log_dict, loss
    # endregion

    # region PYTORCH HOOKS
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch)

        try:
            outputs = self(batch, teacher_forcing, training=True)
        except Exception as e:
            outputs = self(batch, teacher_forcing=True, training=True)
        finally:
            log_dict, loss = self.compute_loss(batch, outputs, prefix='train')
            self.log_dict(
                log_dict,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        try:
            outputs = self(batch, teacher_forcing=False, training=False)
        except Exception as e:
            print("Error", e)
            # Perform a forward pass without gradient computation
            with torch.no_grad():
                dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            return dummy_loss

        log_dict, loss = self.compute_loss(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # num_atoms_forcing = (
        #     self.current_epoch <= self.hparams.teacher_forcing_max_epoch + 5
        # )
        try:
            outputs = self(batch, teacher_forcing=False, training=False)
        except Exception as e:
            print("Error", e)
            # Perform a forward pass without gradient computation
            with torch.no_grad():
                dummy_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            return None

        log_dict, loss = self.compute_loss(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
        )
        return loss

    # def on_before_optimizer_step(self, optimizer):
    #     print("Checking gradients before optimizer step:")
    #     for name, param in self.named_parameters():
    #         if param.grad is not None:
    #             print(f"{name} | Gradients: {param.grad}")
    #         else:
    #             print(f"{name} | Gradients: None")
    # endregion

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        gc.collect()