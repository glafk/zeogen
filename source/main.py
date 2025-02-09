import os
from pathlib import Path

import torch
import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.loggers import WandbLogger
from diffusion_model import DiffusionModel
import notebooks
from dataset_codes import SMALL_DATASET_CODES, ALL_CODES
import env
import wandb

# Load environment variables
env.load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(env.get_env("PROJECT_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)


def run_diffusion(cfg: omegaconf.DictConfig):
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
    model.lattice_scaler = datamodule.lattice_scaler.copy()
    model.scaler = datamodule.scaler.copy()
    torch.save(datamodule.lattice_scaler, hydra_dir / 'lattice_scaler.pt')
    torch.save(datamodule.scaler, hydra_dir / 'prop_scaler.pt')

    # Logger instantiation/configuration
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
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

    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=hydra_dir,
        deterministic=cfg.train.deterministic,
        logger=wandb_logger,
        **cfg.train.pl_trainer,
        accelerator="gpu"
    )

    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    hydra.utils.log.info("Starting testing!")
    trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()
    
    return model

def run_reconstruction(cfg: omegaconf.DictConfig, model: DiffusionModel = None):

    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)
    
    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    if model is None:
        # Load model
        hydra.utils.log.info(f"Loading model <{cfg.model._target_}>")
        model = DiffusionModel.load_from_checkpoint(cfg.model.ckpt_path)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    ) 

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    model.lattice_scaler = datamodule.lattice_scaler.copy()
    model.scaler = datamodule.scaler.copy()

    datamodule.setup(stage="predict")
    model.eval()
    predict_dataloader = datamodule.predict_dataloader()
    
    model = model.to("cuda")
    print(f"Saving reconstructions to {cfg.model.reconstructions_file}.")
    reconstructions_path = os.path.join(f"{PROJECT_ROOT}/reconstructions", cfg.model.reconstructions_file)
    gt_path = os.path.join(f"{PROJECT_ROOT}/reconstructions", cfg.model.reconstructions_file.split('.')[0] + "_gt.pickle")
    counter = 1
    for i, batch in enumerate(predict_dataloader):
        if i * cfg.data.datamodule.batch_size.predict >= cfg.model.num_reconstructions:
            break        
        print(f"processsing batch {counter}")
        batch = batch.to("cuda")
        with torch.no_grad():  # No need to track gradients during inference
            model.reconstruct(batch, omegaconf.DictConfig({"n_step_each": 100, "step_lr": 0.0001, "min_sigma": 0.01, "save_traj": False, "disable_bar": False}), reconstructions_path=reconstructions_path, reconstructions_gt_path=gt_path)

        if cfg.model.save_reconstructions_online:
            artifact_recon = wandb.Artifact(cfg.model.reconstructions_file.split('.')[0], type='dataset')
            artifact_recon.add_file(reconstructions_path)
            artifact_recon_gt = wandb.Artifact(cfg.model.reconstructions_file.split('.')[0] + "_gt", type='dataset')
            artifact_recon_gt.add_file(gt_path)
            
            wandb.log_artifact(artifact_recon)
            wandb.log_artifact(artifact_recon_gt)

            # Clean up the file so that it doesn't hang around
            os.remove(reconstructions_path)
            os.remove(gt_path)
    
def run_sampling(cfg: omegaconf.DictConfig, model: DiffusionModel = None, domains_to_sample=None, domains_per_batch = None):
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)
    
    print(cfg.model.latent_dim)
    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    if model is None:
        # Load model
        hydra.utils.log.info(f"Loading model <{cfg.model._target_}>")
        model = DiffusionModel.load_from_checkpoint(cfg.model.ckpt_path)

    # Instantiate datamodule
    # Here we isntantiate the datamodule because we need to pass the scaler
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    ) 

    # Pass scaler from datamodule to model
    hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
    model.lattice_scaler = datamodule.lattice_scaler.copy()
    model.scaler = datamodule.scaler.copy()

    model.eval()

    model = model.to("cuda")
    samples_path = os.path.join(f"{PROJECT_ROOT}/samples", cfg.model.samples_file)

    for i in range(0, len(domains_to_sample), domains_per_batch):
        domains = domains_to_sample[i:i+domains_per_batch]
        model.sample(cfg.model.samples_per_domain, omegaconf.DictConfig({"n_step_each": 100, 
                                            "step_lr": 0.0001, 
                                            "min_sigma": 0.01, 
                                            "save_traj": False, 
                                            "disable_bar": False}), 
                                            save_samples=True, 
                                            samples_path=samples_path,
                                            domains=domains,
                                            hoas=[0, 0.5, 1.0, 1.5, 2.0])
        if cfg.model.save_samples_online:
            artifact = wandb.Artifact(cfg.model.samples_file.split('.')[0], type='dataset')
            artifact.add_file(samples_path)
            wandb.log_artifact(artifact)

            # Clean up the file so that it doesn't hang around
            os.remove(samples_path)

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="diffusion")
def main(cfg: omegaconf.DictConfig):
    model = None
    # Run training and sampling loop
    # model = run_diffusion(cfg)
    
    # Run only sampling from saved model
    run_sampling(cfg, model, SMALL_DATASET_CODES, 3)

    # Run reconstruction from saved model
    # run_reconstruction(cfg)

if __name__ == "__main__":
    main()