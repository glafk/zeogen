from pathlib import Path
from typing import Any, Dict

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

from utils import add_object, log_config_to_wandb
from data_utils.crystal_utils import frac_to_cart_coords, cart_to_frac_coords, min_distance_sqr_pbc, mard, lengths_angles_to_volume

# Load environment variables
env.load_envs()

PROJECT_ROOT = Path(env.get_env("PROJECT_ROOT"))

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
        mods.append(nn.Softmax())

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



class CDiVAE(BaseModule):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.zd_encoder = hydra.utils.instantiate(
            self.hparams.domain_encoder, num_targets=self.hparams.domain_latent_dim) # DIVA -> self.qzd
        
        self.zy_encoder = hydra.utils.instantiate(
            self.hparams.class_encoder, num_targets=self.hparams.class_latent_dim) # DIVA -> self.qzy
        
        self.zx_encoder = hydra.utils.instantiate(
            self.hparams.residual_encoder, num_targets=self.hparams.residual_latent_dim) # DIVA -> self.qzx

        self.pzd = CondPrior(1, self.hparams.domain_latent_dim)

        # Hard code one as y in this case would be the HOA which is just a scalar
        self.pzy = CondPrior(1, self.hparams.class_latent_dim)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder)

        # DIVA - > self.fc_d -> self.qd. Predictor for the zeolite type from zd
        self.domain_predictor = build_mlp(self.hparams.domain_latent_dim, self.hparams.hidden_dim,
                                          self.hparams.fc_num_layers, self.hparams.num_zeolite_types,
                                          final_activation="softmax")

        # Used to predict the mean of the HOA from the zd latent vector
        # Needed because the distribution of the HOA is different for different zeolite types
        # Actually I'm starting to question the need for this.
        # I may not need to predict neither the mu not the std
        # I can keep the generation of the zy latent space from the scaled hoa
        # separate from the zd latent space, while also generating the final
        # HOA prediction from the zd latent space and the zy latent space but not backpropagating through it.
        # so as to keep the learning on the scaled HOA which I will later need for the sampling
        self.hoa_mu_predictor = build_mlp(self.hparams.domain_latent_dim, self.hparams.hidden_dim,
                                self.hparams.fc_num_layers, 1, final_activation="relu")

        self.hoa_std_predictor = build_mlp(self.hparams.domain_latent_dim, self.hparams.hidden_dim,
                                self.hparams.fc_num_layers, 1, final_activation="relu")        

        self.norm_hoa_predictor = build_mlp(self.hparams.class_latent_dim, self.hparams.hidden_dim,
                                            self.hparams.fc_num_layers, 1)

        self.fc_num_atoms = build_mlp(self.hparams.total_latent_dim, self.hparams.hidden_dim,
                                      self.hparams.fc_num_layers, self.hparams.max_atoms+1)
        self.fc_lengths = build_mlp(self.hparams.total_latent_dim, self.hparams.hidden_dim,
                                    self.hparams.fc_num_layers, 3, final_activation="relu")
        self.fc_angles = build_mlp(self.hparams.total_latent_dim, self.hparams.hidden_dim,
                                   self.hparams.fc_num_layers, 3, final_activation='sigmoid')
        self.fc_composition = build_mlp(self.hparams.total_latent_dim, self.hparams.hidden_dim,
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
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        q = dist.Normal(mu, log_var)

        return q

    def encode(self, batch):
        """
        encode crystal structures to latents.
        """
        # Comments providing mapping between the variables in this implementation
        # and the original DIVA implementation
        # DIVA - mu_d -> zd_q_loc, log_var_d -> zd_q_scale 
        mu_d, log_var_d, hidden_d = self.zd_encoder(batch)
        qzd = self.reparameterize(mu_d, log_var_d)
        zd = qzd.rsample() # DIVA -> zd_q

        # DIVA - mu_y -> zy_q_loc, log_var_y -> zy_q_scale
        mu_y, log_var_y, hidden_y = self.zy_encoder(batch)
        qzy = self.reparameterize(mu_y, log_var_y)
        zy = qzy.rsample() # DIVA -> zy_q

        # DIVA - mu_x -> zx_q_loc, log_var_x -> zx_q_scale
        mu_x, log_var_x, hidden_x = self.zx_encoder(batch)
        qzx = self.reparameterize(mu_x, log_var_x)
        zx = qzx.rsample() # DIVA -> zx_q

        z = torch.cat([zd, zy, zx], dim=-1)

        # Temporarily save the hidden layer for debugging
        return mu_d, log_var_d, mu_y, log_var_y, mu_x, log_var_x, hidden_d, hidden_y, hidden_x, zd, zy, zx, z

    def decode_stats(self, z, gt_num_atoms=None, gt_lengths=None, gt_angles=None,
                     teacher_forcing=False):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing.
        """
        if gt_num_atoms is not None and teacher_forcing:
            num_atoms = self.predict_num_atoms(z)
            lengths = self.predict_lenghts(z, gt_num_atoms)
            angles = self.predict_angles(z)
            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.predict_num_atoms(z)
            lengths = self.predict_lenghts(z, num_atoms)
            angles = self.predict_angles(z)
            composition_per_atom = self.predict_composition(z, num_atoms)

        lengths_and_angles = torch.cat([lengths, angles], dim=-1)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom

    # endregion

    def forward(self, batch, teacher_forcing=False, training=False):
        # region ENCODE
        (mu_d, 
         log_var_d, 
         mu_y, 
         log_var_y, 
         mu_x, 
         log_var_x, 
         hidden_d, 
         hidden_y, 
         hidden_x, 
         zd, 
         zy, 
         zx, 
         z) = self.encode(batch)

        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
         pred_composition_ratio) = self.decode_stats(
            z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing)
        # endregion

        # region PREDICT
        # sample noise levels.
        noise_level = torch.randint(0, self.sigmas.size(0),
                                    (batch.num_atoms.size(0),),
                                    device=self.device)
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            batch.num_atoms, dim=0)
        try:
            # Select a random noise level for the atom types per crystal in the batch
            type_noise_level = torch.randint(0, self.type_sigmas.size(0),
                                            (batch.num_atoms.size(0),),
                                            device=self.device)
            type_noise = self.type_sigmas[type_noise_level]
            # Generate random noise signs to ensure noise does not only increase the probabilit of Al atoms
            random_signs = (torch.rand(batch.num_atoms.size(0), device=self.device) - 0.5).sign()

            type_noise = type_noise * random_signs

            # Add noise to predicted ratios and clamp to [0, 1]
            noisy_ratios = torch.clamp(pred_composition_ratio.squeeze() + type_noise, 0.0, 1.0)
            # Expand the predicted composition ratio to match the number of atoms per crystal in the batch
            pred_composition_per_atom = pred_composition_ratio.squeeze()[batch.batch.squeeze()]
            noisy_composition_per_atom = noisy_ratios[batch.batch.squeeze()]

            # Adjust with 13 to end up with only Al and Si atoms
            rand_atom_types = torch.multinomial(torch.stack(
                (noisy_composition_per_atom, 1-pred_composition_per_atom), dim=1), num_samples=1).squeeze(1) + 13
        except Exception as e:
            batch = {
                "original_batch": batch,
                "teacher_forcing": teacher_forcing,
                "hidden_d": hidden_d,
                "hidden_y": hidden_y,
                "hidden_x": hidden_x,
                "zd": zd,
                "zy": zy,
                "zx": zx,
                "mu_d": mu_d,
                "log_var_d": log_var_d,
                "mu_y": mu_y,
                "log_var_y": log_var_y,
                "mu_x": mu_x,
                "log_var_x": log_var_x,
                "type_noise_level": type_noise_level,
                "type_noise": type_noise,
                "noisy_ratios": noisy_ratios,
                "pred_composition_per_crystal": pred_composition_per_atom,
                "pred_composition_ratio": pred_composition_ratio,
                "pred_lengths": pred_lengths,
                "pred_angles": pred_angles,
                "zeolite_code": batch.zeolite_code
            }

            # Save the batch
            with open("/home/TUE/20220787/zeogen/problem_batch_mu_sig_std.pkl", "wb") as f:
                pickle.dump(batch, f)

            # Save the model weights
            torch.save(self.state_dict(), "/home/TUE/20220787/zeogen/model_weights_mu_sig_std.pth")

            raise e


        # add noise to the cart coords
        cart_noises_per_atom = (
            torch.randn_like(batch.frac_coords) *
            used_sigmas_per_atom[:, None])
        cart_coords = frac_to_cart_coords(
            batch.frac_coords, pred_lengths, pred_angles, batch.num_atoms)
        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(
            cart_coords, pred_lengths, pred_angles, batch.num_atoms)

        # pred_cart_coord_diff is the prediction for the difference in the atom coords based on the noise that is added
        pred_cart_coord_diff, pred_atom_types = self.decoder(
            z, noisy_frac_coords, rand_atom_types, batch.num_atoms, pred_lengths, pred_angles)

        # Predict domain and HOA
        domain_pred = self.domain_predictor(zd)
        hoa_mu_pred = self.hoa_mu_predictor(zd)
        hoa_std_pred = self.hoa_std_predictor(zd)
        norm_hoa_pred = self.norm_hoa_predictor(zy)       
        hoa_pred = norm_hoa_pred * hoa_std_pred + hoa_mu_pred

        zd_p_mu, zd_p_var = self.pzd(batch['zeolite_code_enc'].float().view(-1, 1))
        zy_p_mu, zy_p_var = self.pzy(batch['norm_hoa'].view(-1, 1))
        zx_p_mu, zx_p_var = torch.zeros(zd_p_mu.size()[0], self.hparams.residual_latent_dim).cuda(),\
                                torch.ones(zd_p_mu.size()[0], self.hparams.residual_latent_dim).cuda()
        # endregion

        return {
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'pred_atom_types': pred_atom_types,
            'pred_composition_per_atom': pred_composition_per_atom,
            'used_sigmas_per_atom': used_sigmas_per_atom,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            'rand_atom_types': rand_atom_types,
            'type_noise': type_noise,
            'noisy_frac_coords': noisy_frac_coords,
            'mu_d': mu_d,
            'log_var_d': log_var_d,
            'mu_y': mu_y,
            'log_var_y': log_var_y,
            'mu_x': mu_x,
            'log_var_x': log_var_x,
            'z': z,
            'zd': zd,
            'zy': zy,
            'zx': zx,
            'zd_p_mu': zd_p_mu,
            'zd_p_var': zd_p_var,
            'zy_p_mu': zy_p_mu,
            'zy_p_var': zy_p_var,
            'zx_p_mu': zx_p_mu,
            'zx_p_var': zx_p_var,
            'domain_pred': domain_pred,
            'hoa_pred': hoa_pred,
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
        pred_lengths = self.lengths_scaler.inverse_transform(pred_lengths)
        if self.hparams.data["lattice_scale_method"] == 'scale_length':
            # Scale according to number of atoms
            pred_lengths = pred_lengths * num_atoms.argmax(dim=-1).view(-1, 1).float()**(1/3)

        return pred_lengths
        
    def predict_angles(self, z):
        pred_angles = self.fc_angles(z)
        # Multiply each angles by 180
        pred_angles = pred_angles * 180
        return pred_angles

    def predict_composition(self, z, num_atoms):
        pred_composition_ratio = self.fc_composition(z)
        return pred_composition_ratio
    # endregion

    # region SAMPLING
    # TODO: Rework sampling to work as intended for the CDiVAE
    def generate_rand_init(self, pred_composition_per_atom, num_atoms):
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
                # Noise added to cartesian coordinates
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

        output_dict = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                       'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types,
                       'is_traj': False}

        if ld_kwargs.save_traj:
            output_dict.update(dict(
                z=z,
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_pred_cart_coord_diff=torch.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_noise_cart=torch.stack(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict

    def sample(self, num_samples, ld_kwargs, kwargs_conf_name, domains, norm_hoas):
        """
        Samples crystals and optionally saves them.

        Args:
            num_samples (int): Number of samples to generate.
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
        assert len(norm_hoas) == num_samples

        # Here in the sampling part I will need to figure out how to force the model to sample from the part of the distribution where the representations of the "high-capacity" crystals lie
        print(f"Sampling {num_samples} crystals.")
        z = torch.randn(num_samples, self.hparams.hidden_dim, device=self.device)
        samples = self.langevin_dynamics(z, ld_kwargs)

        return samples   

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

    def property_loss(self, z, batch):
        return F.mse_loss(self.fc_property(z), batch.y)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lengths_scaler.match_device(pred_lengths_and_angles)
        # TODO: Perhaps we don't need this scaling either when calculating the loss
        # for the same reason described below
        if self.hparams.data["lattice_scale_method"] == 'scale_length':
            target_lengths = batch.lengths / \
                batch.num_atoms.view(-1, 1).float()**(1/3)
        # Since we already transform the predictions with inverse transform
        # This scaleing here my not be necessary
        # scaled_target_lengths = self.lengths_scaler.transform(target_lengths)
        # Let's ty like that and see if it gets better
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles], dim=-1)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        target_atom_types = target_atom_types - 13
        # Cast target atom types to float
        target_atom_types = target_atom_types.float()
        loss = F.binary_cross_entropy(pred_composition_per_atom.squeeze(),
                               target_atom_types, reduction='none')
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
        # TODO: Perhaps this needs consitent normalizing
        pred_cart_coord_diff = pred_cart_coord_diff / \
            used_sigmas_per_atom[:, None]

        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def type_loss(self, pred_atom_types, target_atom_types,
                  type_noise, batch):
        target_atom_types = target_atom_types
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, reduction='none')
        # rescale loss according to noise
        loss = loss / torch.abs((1 / (1 - type_noise.repeat_interleave(batch.num_atoms, dim=0))))

        return scatter(loss, batch.batch, reduce='mean').mean()

    def kld_loss(self, mu_q, log_var_q, mu_p, log_var_p):
        var_p = log_var_p.exp()
        var_q = log_var_q.exp()

        kld = 0.5 * (
            torch.sum(log_var_q - log_var_p, dim=1)
            - mu_p.size(1)
            + torch.sum(var_p / var_q, dim=1)
            + torch.sum((mu_q - mu_p)**2 / var_q, dim=1)
        )

        return torch.mean(kld, dim=0)

    def domain_pred_loss(self, pred_domain_logits, batch):
        return F.cross_entropy(pred_domain_logits, batch.zeolite_code_enc)

    def norm_hoa_pred_loss(self, pred_norm_hoa, batch):
        return F.mse_loss(pred_norm_hoa, batch.norm_hoa)

    def hoa_mu_pred_loss(self, pred_hoa_mu, batch):
        return F.mse_loss(pred_hoa_mu, batch.hoa_mu)
    
    def hoa_std_pred_loss(self, pred_hoa_std, batch):
        return F.mse_loss(pred_hoa_std, batch.hoa_std)

    @torch.no_grad()
    def final_hoa_loss(self, pred_hoa, batch):
        return F.mse_loss(pred_hoa, batch.hoa)

    def compute_loss(self, batch, outputs, prefix):
        pred_num_atoms = outputs['pred_num_atoms']
        pred_lengths_and_angles = outputs['pred_lengths_and_angles']
        pred_composition_per_atom = outputs['pred_composition_per_atom']
        pred_cart_coord_diff = outputs['pred_cart_coord_diff']
        pred_atom_types = outputs['pred_atom_types']
        noisy_frac_coords = outputs['noisy_frac_coords']
        used_sigmas_per_atom = outputs['used_sigmas_per_atom']
        type_noise = outputs['type_noise']
        mu_d = outputs['mu_d']
        log_var_d = outputs['log_var_d']
        mu_y = outputs['mu_y']
        log_var_y = outputs['log_var_y']
        mu_x = outputs['mu_x']
        log_var_x = outputs['log_var_x']
        zd_p_mu = outputs['zd_p_mu']
        zd_p_log_var = outputs['zd_p_var']
        zy_p_mu = outputs['zy_p_mu']
        zy_p_log_var = outputs['zy_p_var']
        zx_p_mu = outputs['zx_p_mu']
        zx_p_log_var = outputs['zx_p_var']
        domain_pred = outputs['domain_pred']
        hoa_mu_pred = outputs['hoa_mu_pred']
        hoa_std_pred = outputs['hoa_std_pred']
        hoa_pred = outputs['hoa_pred']
        norm_hoa_pred = outputs['norm_hoa_pred']

        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types, batch)
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)
        type_loss = self.type_loss(pred_atom_types, batch.atom_types,
                                   type_noise, batch)

        kld_loss_d = self.kld_loss(mu_d, log_var_d, zd_p_mu, zd_p_log_var)
        kld_loss_y = self.kld_loss(mu_y, log_var_y, zy_p_mu, zy_p_log_var)
        kld_loss_x = self.kld_loss(mu_x, log_var_x, zx_p_mu, zx_p_log_var)
        kld_loss = kld_loss_d + kld_loss_y + kld_loss_x

        domain_pred_loss = self.domain_pred_loss(domain_pred, batch)
        norm_hoa_pred_loss = self.norm_hoa_pred_loss(norm_hoa_pred, batch)
        hoa_mu_pred_loss = self.hoa_mu_pred_loss(hoa_mu_pred, batch)
        hoa_std_pred_loss = self.hoa_std_pred_loss(hoa_std_pred, batch)
        final_hoa_loss = self.final_hoa_loss(hoa_pred, batch)

        loss = (
            self.hparams.cost_natom * num_atom_loss +
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_coord * coord_loss +
            self.hparams.cost_type * type_loss +
            self.hparams.beta * kld_loss +
            self.hparams.cost_composition * composition_loss +
            self.hparams.cost_domain * domain_pred_loss +
            self.hparams.cost_hoa * norm_hoa_pred_loss +
            self.hparams.cost_hoa * hoa_mu_pred_loss +
            self.hparams.cost_hoa * hoa_std_pred_loss
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
            f'{prefix}_kld_loss_x': kld_loss_x,
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
                dim=-1) == (target_atom_types - 1)
            type_accuracy = scatter(type_accuracy.float(
            ), batch.batch, dim=0, reduce='mean').mean()

            log_dict.update({
                f'{prefix}_norm_hoa_pred_loss': norm_hoa_pred_loss,
                f'{prefix}_domain_pred_loss': domain_pred_loss,
                f'{prefix}_hoa_mu_pred_loss': hoa_mu_pred_loss,
                f'{prefix}_hoa_std_pred_loss': hoa_std_pred_loss,
                f'{prefix}_final_hoa_loss': final_hoa_loss,
                f'{prefix}_loss': loss,
                f'{prefix}_natom_accuracy': num_atom_accuracy,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_volumes_mard': volumes_mard,
                f'{prefix}_type_accuracy': type_accuracy,
            })

        return log_dict, loss
    # endregion

    # region PYTORCH HOOKS
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch)
        outputs = self(batch, teacher_forcing, training=True)
        log_dict, loss = self.compute_loss(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_loss(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_loss(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
        )
        return loss
    # endregion