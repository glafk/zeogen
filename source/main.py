import os
from pathlib import Path

import torch
import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from hydra.core.hydra_config import HydraConfig
from pytorch_lightning.loggers import WandbLogger

import env

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
    )

    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    hydra.utils.log.info("Starting testing!")
    trainer.test(datamodule=datamodule)

    model.sample(25, omegaconf.DictConfig({"n_step_each": 10, "step_lr": 0.1, "min_sigma": 0.01, "save_traj": False, "disable_bar": False}), save_samples=True)

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb_logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="diffusion")
def main(cfg: omegaconf.DictConfig):
    run_diffusion(cfg)

if __name__ == "__main__":
    main()