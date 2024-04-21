"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

---

Script for calculating the scaling factors used to even out GemNet activation
scales. This generates the `scale_file` specified in the config, which is then
read in at model initialization.
This only needs to be run if the hyperparameters or model change
in places were it would affect the activation scales.
"""

import logging
import os
import sys
from itertools import islice
from pathlib import Path

import torch
from tqdm import trange
from hydra import initialize, compose
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from gemnet.layers.scaling import AutomaticFit
from gemnet.utils import write_json


import env

# Load environment variables
env.load_envs()

# Set the cwd to the project root
PROJECT_ROOT: Path = Path(env.get_env("PROJECT_ROOT"))
assert (
    PROJECT_ROOT.exists()
), "You must configure the PROJECT_ROOT environment variable in a .env file!"

os.chdir(PROJECT_ROOT)

#@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="diffusion")
def fit_scaling(): 

    num_batches = 16  # number of batches to use to fit a single variable
    scale_file = "source/gemnet/scaling_factors_gemnet_diffusion.json"
    # Print current directory
    print(f"DIR {os.getcwd()}")
    logging.info(f"Target scale file: {scale_file}")

    def initialize_scale_file(scale_file):
        # initialize file
        preset = {"comment": "gemnet scaling factors"}
        write_json(scale_file, preset)

    if os.path.exists(scale_file):
        logging.warning(f"Already found existing file: {scale_file}")
        flag = input(
            "Do you want to continue and overwrite the file (1), "
            "only fit the variables not fitted yet (2), or exit (3)? "
        )
        if str(flag) == "1":
            logging.info("Overwriting the current file.")
            initialize_scale_file(scale_file)
        elif str(flag) == "2":
            logging.info("Only fitting unfitted variables.")
        else:
            print(flag)
            logging.info("Exiting script")
            sys.exit()
    else:
        initialize_scale_file(scale_file)

    AutomaticFit.set2fitmode()

    with initialize(version_base=None, config_path="../conf"):

        # Compose the config
        cfg = compose(config_name="test")
        print(cfg.data)

        # Instantiate the datamodule
        datamodule: pl.LightningDataModule = hydra.utils.instantiate(
            cfg.data.datamodule, _recursive_=False
        )

        datamodule.setup("fit")
        # Instantiate the model
        model: pl.LightningModule = hydra.utils.instantiate(
            cfg.model, data=cfg.data, _convert_="partial", _recursive_=False
        )

        # Pass scaler from datamodule to model
        hydra.utils.log.info(f"Passing scaler from datamodule to model <{datamodule.scaler}>")
        model.lattice_scaler = datamodule.lattice_scaler.copy()
        model.scaler = datamodule.scaler.copy()
        torch.save(datamodule.lattice_scaler, 'lattice_scaler_fit.pt')
        torch.save(datamodule.scaler, 'prop_scaler_fit.pt')

        # Get the test dataloader from the datamodule
        train_dataloader = datamodule.train_dataloader()

    # Fitting loop
    logging.info("Start fitting")
    print("Queue")
    print(AutomaticFit.queue)
    print(f"Active var: {AutomaticFit.activeVar}")
    if not AutomaticFit.fitting_completed():
        with torch.no_grad():
            # model.eval()
            # print(dir(train_dataloader))
            # print(type(train_dataloader.dataset)) 
            progress_bar = trange(len(AutomaticFit.queue) + 1)
            for _ in progress_bar:
                for batch_idx, batch in enumerate(islice(train_dataloader, num_batches)):
    
                    # Perform forward pass
                    predictions = model(
                        batch
                        )
    
                    # Delete predictions to manage memory load
                    del predictions
    
                    # Update progress bar
                    progress_bar.set_description(f'Batches: {batch_idx + 1}')
    
                current_var = AutomaticFit.activeVar
                if current_var is not None:
                    current_var.fit()  # fit current variable
                    logging.info(f"Fitting variable: {current_var}")
                else:
                    print("Found no variable to fit. Something went wrong!")
    
        assert AutomaticFit.fitting_completed()
        logging.info(f"Fitting done. Results saved to: {scale_file}")

if __name__ == "__main__":
    fit_scaling()
