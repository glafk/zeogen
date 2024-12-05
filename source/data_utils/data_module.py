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

from data_utils.balanced_sampler import SamplerFactory
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
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.get_scaler(scaler_path)

    def prepare_data(self) -> None:
        # download only
        pass

    def get_scaler(self, scaler_path):
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
                Path(scaler_path) / 'lattice_scaler.pt')
            self.scaler = torch.load(Path(scaler_path) / 'prop_scaler.pt')

    def setup(self, stage: Optional[str] = None):
        """
        construct datasets and assign data scalers.
        """
        print("Setting up data module")
        if stage == "fit":
            train_preprocessed_path = self.datasets.train.path.split('.')[0] + '_preprocessed.pt'
            val_preprocessed_path = self.datasets.val.path.split('.')[0] + '_preprocessed.pt'
            if os.path.exists(train_preprocessed_path) and os.path.exists(val_preprocessed_path):
                self.train_dataset = torch.load(train_preprocessed_path)
                self.val_dataset = torch.load(val_preprocessed_path)
            else:
                self.train_dataset = hydra.utils.instantiate(self.datasets.train)
                self.val_dataset = hydra.utils.instantiate(self.datasets.val)

                # Save preprocessed data
                torch.save(self.train_dataset, train_preprocessed_path)
                torch.save(self.val_dataset, val_preprocessed_path)

            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler
            self.val_dataset.lattice_scaler = self.lattice_scaler
            self.val_dataset.scaler = self.scaler

        if stage == "test" or stage == "predict":
            test_preprocessed_path = self.datasets.test.path.split('.')[0] + '_preprocessed.pt'
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
        classes = set([item["zeolite_code"] for item in self.train_dataset.cached_data])
        class_idx = {framework: [idx for idx, item in list(enumerate(self.train_dataset.cached_data)) if item["zeolite_code"] == framework] for framework in classes}

        batch_sampler = SamplerFactory().get(
            class_idxs=class_idx,
            batch_size=self.batch_size.train,
            n_batches=len(self.datasets.train) // batch_size,
            alpha=1
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler
        )

    def val_dataloader(self) -> Sequence[DataLoader]:

        classes = set([item["zeolite_code"] for item in self.val_dataset.cached_data])
        class_idx = {framework: [idx for idx, item in list(enumerate(self.val_dataset.cached_data)) if item["zeolite_code"] == framework] for framework in classes}
        
        batch_sampler = SamplerFactory().get(
            class_idxs=class_idx,
            batch_size=self.batch_size.val,
            n_batches=len(self.datasets.val) // batch_size,
            alpha=1
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler
        )

    def test_dataloader(self) -> Sequence[DataLoader]:

        classes = set([item["zeolite_code"] for item in self.test_dataset.cached_data])
        class_idx = {framework: [idx for idx, item in list(enumerate(self.test_dataset.cached_data)) if item["zeolite_code"] == framework] for framework in classes}
        
        batch_sampler = SamplerFactory().get(
            class_idxs=class_idx,
            batch_size=self.batch_size.test,
            n_batches=len(self.datasets.test) // batch_size,
            alpha=1
        )

        return DataLoader(
            self.train_dataset,
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

@hydra.main(config_path="C:/TUE/thesis/zeogen/conf", config_name="test")
def main(cfg: DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('fit')
    # TODO: Look into this. Don't know what this is
    # import pdb
    # pdb.set_trace()


if __name__ == "__main__":
    main()