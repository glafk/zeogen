import os
from pathlib import Path
import json
import shutil
from datetime import datetime

import torch
import hydra
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from diffusion_model import DiffusionModel
import wandb

from utils import load_from_wandb, log_config_to_wandb, add_object

import env

# Load environment variables
env.load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(env.get_env("PROJECT_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)

def run_training(cfg: DictConfig):
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)
    
    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    model.lengths_scaler = datamodule.lengths_scaler.copy()
    model.scaler = datamodule.scaler.copy()
    torch.save(datamodule.lengths_scaler, hydra_dir / 'lengths_scaler_total_dataset.pt')
    torch.save(datamodule.scaler, hydra_dir / 'prop_scaler_total_dataset.pt')

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb

        # Enable version counter false allows us to overwrite models so that
        # we don't have too many artifacts at the end of the run 
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", enable_version_counter=True, filename=f"model_{cfg.expname}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        wandb_logger = WandbLogger(
            **wandb_config,
            tags=cfg.core.tags,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    # Log the current config used for the experiment
    # Convert OmegaConf object to a dictionary
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Define the filename for saving the config
    config_filename = f'config_{cfg.expname}.json'

    # Save the config dictionary to a JSON file
    with open(config_filename, 'w') as f:
        json.dump(config_dict, f, indent=4)

    # Create a wandb artifact and add the config file
    artifact = wandb.Artifact(config_filename.split('.')[0], type='configuration')
    artifact.add_file(config_filename)

    # Log the artifact to the current run
    wandb.log_artifact(artifact)

    # Clean up the file so that it doesn't hang around
    # Optionally, clean up the local config file
    os.remove(config_filename)

    hydra.utils.log.info("Instantiating the Trainer")
    
    if cfg.model.resume_from_checkpoint:
        assert cfg.model.experiment_name_to_load is not None, "Please provide an experiment name"
        model_path, model_dir = load_from_wandb(cfg.model.experiment_name_to_load)

         
        trainer = pl.Trainer(
            default_root_dir=hydra_dir,
            deterministic=cfg.train.deterministic,
            logger=wandb_logger,
            **cfg.train.pl_trainer,
            accelerator="gpu",
            callbacks=[checkpoint_callback],
            detect_anomaly=True
        )

        hydra.utils.log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=model_path)

        hydra.utils.log.info("Starting testing!")
        trainer.test(datamodule=datamodule)

        # Clean up downloaded files
        shutil.rmtree(model_dir)
    else:
        trainer = pl.Trainer(
            default_root_dir=hydra_dir,
            deterministic=cfg.train.deterministic,
            logger=wandb_logger,
            **cfg.train.pl_trainer,
            accelerator="gpu",
            callbacks=[checkpoint_callback],
            # detect_anomaly=True
        )

        hydra.utils.log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

        hydra.utils.log.info("Starting testing!")
        trainer.test(datamodule=datamodule)
    
    return model


def run_reconstruction(cfg: DictConfig, model: DiffusionModel = None):

    # Log the configuration using wandb.config
    log_config_to_wandb(cfg, f"reconstruction-config-{cfg.model.experiment_name_to_load}")

    if cfg.model.load_model and model is None:
        # Make sure that the model location is provided
        assert cfg.model.model_location is not None

        if cfg.model.model_location == "local":
            assert cfg.model.ckpt_path is not None, "Please provide a path to the model checkpoint"
            # Load model
            hydra.utils.log.info(f"Loading model <{cfg.model._target_}>")
            model = DiffusionModel.load_from_checkpoint(cfg.model.ckpt_path)
        elif cfg.model.model_location == "wandb":
            assert cfg.model.experiment_name_to_load is not None, "Please provide an experiment name"
            model_path, model_dir = load_from_wandb(cfg.model.experiment_name_to_load)
            model = DiffusionModel.load_from_checkpoint(model_path)

            # Clean up downloaded files
            shutil.rmtree(model_dir)
    else:
        raise ValueError("Both load model and arguments model were provided. Ambuguious use of the script")

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    ) 

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    model.lengths_scaler = datamodule.lengths_scaler.copy()
    model.scaler = datamodule.scaler.copy()

    datamodule.setup(stage="predict")
    model.eval()
    predict_dataloader = datamodule.predict_dataloader()
    
    model = model.to("cuda")

    reconstructions_path = os.path.join(f"{PROJECT_ROOT}/reconstructions", cfg.model.reconstructions_file)
    ground_truth_path = os.path.join(f"{PROJECT_ROOT}/reconstructions", cfg.model.reconstructions_file.split('.')[0] + "_gt.pickle")

    for batch in predict_dataloader:
        batch = batch.to("cuda")
        with torch.no_grad():  # No need to track gradients during inference
            model.reconstruct(batch, DictConfig(
                {"n_step_each": 100, 
                 "step_lr": 0.0001, 
                 "min_sigma": 0.01, 
                 "save_traj": True, 
                 "disable_bar": False}), 
                 reconstructions_path, 
                 ground_truth_path,
                 kwargs_conf_name=f"reconstruction-config-{cfg.model.experiment_name_to_load}")

    if cfg.model.save_reconstructions_online:
        artifact_recon = wandb.Artifact(cfg.model.reconstructions_file.split('.')[0], type='dataset')
        artifact_recon.add_file(reconstructions_path)
        artifact_recon_gt = wandb.Artifact(cfg.model.reconstructions_file.split('.')[0] + "_gt", type='dataset')
        artifact_recon_gt.add_file(ground_truth_path)
        
        wandb.log_artifact(artifact_recon)
        wandb.log_artifact(artifact_recon_gt)

        # Clean up the file so that it doesn't hang around
        os.remove(reconstructions_path)
        os.remove(ground_truth_path)


def run_sampling(cfg: DictConfig, model: DiffusionModel = None):

    # Instantiate wandb run
    wandb.init(project="zeogen", entity="glafk", name=cfg.expname)
    # Log the configuration using wandb.config
    log_config_to_wandb(cfg, f"sampling-config-{cfg.model.experiment_name_to_load}")

    if cfg.model.load_model and model is None:
        # Make sure that the model location is provided
        assert cfg.model.model_location is not None

        if cfg.model.model_location == "local":
            assert cfg.model.ckpt_path is not None, "Please provide a path to the model checkpoint"
            # Load model
            hydra.utils.log.info(f"Loading model <{cfg.model._target_}>")
            model = DiffusionModel.load_from_checkpoint(cfg.model.ckpt_path)
        elif cfg.model.model_location == "wandb":
            assert cfg.model.experiment_name_to_load is not None, "Please provide an experiment name"
            model_path, model_dir = load_from_wandb(cfg.model.experiment_name_to_load)
            model = DiffusionModel.load_from_checkpoint(model_path)

            # Clean up downloaded files
            shutil.rmtree(model_dir)
    else:
        raise ValueError("Both load model and arguments model were provided. Ambuguious use of the script")

    # Instantiate datamodule
    # Here we instantiate the datamodule because we need to pass the scaler
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    ) 

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    model.lengths_scaler = datamodule.lengths_scaler.copy()
    model.scaler = datamodule.scaler.copy()

    # datamodule.setup(stage="predict")
    model.eval()

    model = model.to("cuda")

    with torch.no_grad():  # No need to track gradients during inference
        samples = model.sample(cfg.model.num_samples, DictConfig(
            {"n_step_each": 100,
            "step_lr": 0.0001, 
            "min_sigma": 0.01, 
            "save_traj": True, 
            "disable_bar": False}), 
            kwargs_conf_name=f"samples-kwargs-{cfg.model.experiment_name_to_load}")

    print(f"Saving samples to {cfg.model.samples_file}.")
    samples_path = os.path.join(f"{PROJECT_ROOT}/samples", cfg.model.samples_file)
    add_object(samples, samples_path)
    
    if cfg.model.save_samples_online:
        artifact = wandb.Artifact(cfg.model.samples_file.split('.')[0], type='dataset')
        artifact.add_file(samples_path)
        wandb.log_artifact(artifact)

        # Clean up the file so that it doesn't hang around
        os.remove(samples_path)

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="cdivae")
def main(cfg: DictConfig):

    model = None
    # Run training and sampling loop
    if cfg.model.run_training:
        model = run_training(cfg)
    
    # If training was not run the wandb logger will not be initialized
    if not cfg.model.run_training and (cfg.model.run_sampling or cfg.model.run_reconstruction):
        wandb.init(project="zeogen", entity="glafk", name=cfg.expname)

    # Run only sampling from saved model
    if cfg.model.run_sampling:
        run_sampling(cfg, model)

    # Run reconstruction from saved model
    if cfg.model.run_reconstruction:
        run_reconstruction(cfg, model)

if __name__ == "__main__":
    main()