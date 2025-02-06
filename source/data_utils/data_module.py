import os
import random
from typing import Optional, Sequence
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import hydra
from omegaconf import DictConfig

from data_utils.sampler import ZeoSampler
print("Imported balanced_sampler")
from data_utils.crystal_utils import get_scaler_from_data_list


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class CrystDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        scaler_path=None,
        lattice_scaler_name=None,
        prop_scaler_name=None,
        samples_per_code=None
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.samples_per_code = samples_per_code

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.get_scaler(scaler_path, lattice_scaler_name, prop_scaler_name)

    def prepare_data(self) -> None:
        # download only
        pass

    def get_scaler(self, scaler_path, lattice_scaler_name, prop_scaler_name):
        # Load once to compute property scaler
        if scaler_path is None:
            # temporarily change this to the test dataset to generate the scaling factors
            # test_dataset = hydra.utils.instantiate(self.datasets.test)
            train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.lattice_scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key='scaled_lattice')
            self.scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key=train_dataset.prop)
        else:
            self.lattice_scaler = torch.load(
                Path(scaler_path) / lattice_scaler_name)
            self.scaler = torch.load(Path(scaler_path) / prop_scaler_name)

    def setup(self, stage: Optional[str] = None):
        """
        construct datasets and assign data scalers.
        """
        suffix = "_preprocessed_all_codes.pt" if "all_codes" in self.datasets.train.path else "_preprocessed_small.pt"
        print("Setting up data module")
        if stage == "fit":
            train_preprocessed_path = self.datasets.train.path.split('.')[0] + suffix
            val_preprocessed_path = self.datasets.val.path.split('.')[0] + suffix
            if os.path.exists(train_preprocessed_path) and os.path.exists(val_preprocessed_path):
                self.train_dataset = torch.load(train_preprocessed_path)
                self.val_dataset = torch.load(val_preprocessed_path)
                # print(f"VALIDATION ZEOLITE CODES: {self.val_dataset.zeolite_codes}")
            else:
                self.train_dataset = hydra.utils.instantiate(self.datasets.train)
                self.val_dataset = hydra.utils.instantiate(self.datasets.val)
            
                # print(f"VALIDATION ZEOLITE CODES: {self.val_dataset.zeolite_codes}")

                # Save preprocessed data
                torch.save(self.train_dataset, train_preprocessed_path)
                torch.save(self.val_dataset, val_preprocessed_path)

            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler
            self.val_dataset.lattice_scaler = self.lattice_scaler
            self.val_dataset.scaler = self.scaler

        if stage == "test" or stage == "predict":
            test_preprocessed_path = self.datasets.test.path.split('.')[0] + suffix
            if os.path.exists(test_preprocessed_path):
                self.test_dataset = torch.load(test_preprocessed_path)
            else:
                self.test_dataset = hydra.utils.instantiate(self.datasets.test)
                # Save preprocessed data
                torch.save(self.test_dataset, test_preprocessed_path)

            print("Instantiating test dataset") 
            self.test_dataset.lattice_scaler = self.lattice_scaler
            self.test_dataset.scaler = self.scaler

        if stage == "predict":
            self.predict_dataset = hydra.utils.instantiate(self.datasets.predict)
            print("Instantiating predict dataset")
            self.predict_dataset.lattice_scaler = self.lattice_scaler
            self.predict_dataset.scaler = self.scaler 

    def train_dataloader(self) -> DataLoader:
        batch_sampler = ZeoSampler(self.train_dataset.zeolite_codes, batch_size=self.batch_size.train, n_samples=self.samples_per_code, origin="TRAIN") # Set num samples to 80 for the large dataset as it is better balanced than the small one

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        batch_sampler = ZeoSampler(self.val_dataset.zeolite_codes, batch_size=self.batch_size.val, n_samples=self.samples_per_code, origin="VAL") # Set num samples to 80 for the large dataset as it is better balanced than the small one

        return DataLoader(
            self.val_dataset,
            batch_sampler=batch_sampler
        )

    def test_dataloader(self) -> Sequence[DataLoader]:
        batch_sampler = ZeoSampler(self.test_dataset.zeolite_codes, batch_size=self.batch_size.test, n_samples=self.samples_per_code, origin="TEST") # Set num samples to 80 for the large dataset as it is better balanced than the small one

        return DataLoader(
            self.test_dataset,
            batch_sampler=batch_sampler
        )

    def predict_dataloader(self) -> Sequence[DataLoader]:
 
        return DataLoader(
                self.predict_dataset,
                shuffle=True,
                batch_size=self.batch_size.predict,
                num_workers=self.num_workers.predict,
                worker_init_fn=worker_init_fn,
                # persistent_workers=True
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )

# @hydra.main(config_path="C:/TUE/thesis/zeogen/conf", config_name="test")
# def main(cfg: DictConfig):
#     datamodule: pl.LightningDataModule = hydra.utils.instantiate(
#         cfg.data.datamodule, _recursive_=False
#     )
#     datamodule.setup('fit')

# if __name__ == "__main__":
#     main()