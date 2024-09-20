from pathlib import Path
from typing import Any, Dict
import os


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
import wandb

from utils import add_object, log_config_to_wandb
from data_utils.crystal_utils import frac_to_cart_coords, cart_to_frac_coords, min_distance_sqr_pbc, mard, lengths_angles_to_volume

# Load environment variables
env.load_envs()

MAX_ATOMIC_NUM = 100
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

    return nn.Sequential(*mods)


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
            self.hparams.encoder, num_targets=self.hparams.total_latent_dim)
        
        self.decoder = hydra.utils.instantiate(self.hparams.decoder)

        self.fc_mu = nn.Linear(self.hparams.total_latent_dim,
                               self.hparams.total_latent_dim)
        self.fc_var = nn.Linear(self.hparams.total_latent_dim,
                                self.hparams.total_latent_dim)

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
        self.scaler = None

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * log_var)
        # Add 1.0e-5 to std to avoid numerical issues
        std = std + 1.0e-5
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, batch):
        """
        encode crystal structures to latents.
        """
        hidden = self.encoder(batch)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        z = self.reparameterize(mu, log_var)
        # Temporarily save the hidden layer for debugging
        return mu, log_var, z, hidden

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

    def forward(self, batch, teacher_forcing=False, training=False):
        mu, log_var, z, hidden = self.encode(batch)

        (pred_num_atoms, pred_lengths_and_angles, pred_lengths, pred_angles,
         pred_composition_ratio) = self.decode_stats(
            z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing)

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
            pred_composition_per_atom = noisy_ratios[batch.batch.squeeze()]
            #used_type_sigmas_per_atom = (
            #    self.type_sigmas[type_noise_level].repeat_interleave(
            #        batch.num_atoms, dim=0))

            # Adjust with 13 to end up with only Al and Si atoms
            rand_atom_types = torch.multinomial(torch.stack(
                (pred_composition_per_atom, 1-pred_composition_per_atom), dim=1), num_samples=1).squeeze(1) + 13
        except Exception as e:
            batch = {
                "original_batch": batch,
                "teacher_forcing": teacher_forcing,
                "hidden": hidden,
                "z": z,
                "mu": mu,
                "log_var": log_var,
                "std": torch.exp(0.5 * log_var),
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
        pred_cart_coord_diff, pred_atom_types = self.decoder(
            z, noisy_frac_coords, rand_atom_types, batch.num_atoms, pred_lengths, pred_angles)

        # compute loss.
        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, batch.atom_types, batch)
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, batch)
        type_loss = self.type_loss(pred_atom_types, batch.atom_types,
                                   type_noise, batch)

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

    def predict_lenghts(self, z, num_atoms):
        self.lengths_scaler.match_device(z)
        pred_lengths = self.fc_lengths(z)
        pred_lengths = self.lengths_scaler.inverse_transform(pred_lengths)
        if self.hparams.data["lattice_scale_method"] == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.argmax(dim=-1).view(-1, 1).float()**(1/3)

        return pred_lengths
        
    def predict_angles(self, z):
        pred_angles = self.fc_angles(z)
        # Multiply each angles by 180
        pred_angles = pred_angles * 180
        return pred_angles

    def predict_composition(self, z, num_atoms):
        pred_composition_per_atom = self.fc_composition(z)
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

    def sample(self, num_samples, ld_kwargs, kwargs_conf_name):
        """
        Samples crystals and optionally saves them.

        Args:
            num_samples (int): Number of samples to generate.
            ld_kwargs (dict): Keyword arguments for the Langevin dynamics method.
            save_samples (bool, optional): Whether to save the samples. Defaults to False.
            samples_file (str, optional): The file name to save the samples. Defaults to "samples.pickle".
            save_to_wandb (bool, optional): Whether to save the samples to wandb. Defaults to False.

        Returns:
            samples (torch.Tensor): The generated samples.
        """
        # Log the LD configuration
        log_config_to_wandb(ld_kwargs, kwargs_conf_name, auxiliary_config=True)
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
    # region losses
    def num_atom_loss(self, pred_num_atoms, batch):
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    def property_loss(self, z, batch):
        return F.mse_loss(self.fc_property(z), batch.y)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lengths_scaler.match_device(pred_lengths_and_angles)
        # TODO: Here as well
        if self.hparams.data["lattice_scale_method"] == 'scale_length':
            target_lengths = batch.lengths / \
                batch.num_atoms.view(-1, 1).float()**(1/3)

        scaled_target_lengths = self.lengths_scaler.transform(target_lengths)
        target_lengths_and_angles = torch.cat(
            [scaled_target_lengths, batch.angles], dim=-1)
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

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        return kld_loss
    # endregion

    # region pytorch_lightning hooks
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

    # endregion

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
            scaled_lengths = self.lengths_scaler.inverse_transform(pred_lengths_and_angles[:, :3])
            pred_lengths = scaled_lengths
            pred_angles = pred_lengths_and_angles[:, 3:]

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