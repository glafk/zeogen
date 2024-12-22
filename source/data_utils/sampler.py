from torch.utils.data import BatchSampler
from torch.utils.data import Dataset
from numpy.random import choice, shuffle
import numpy as np

class ZeoSampler(BatchSampler):
    def __init__(self, zeolite_codes: list, batch_size, num_samples=1500):
        self.batch_size = batch_size
        self.num_samples = num_samples

        self.zeolite_codes = zeolite_codes
        self.unique_zeo_codes = set(zeolite_codes)
        self.unique_zeo_codes_num = len(self.unique_zeo_codes)
        self.most_zeo_code = max([zeolite_codes.count(zeo_code) for zeo_code in self.unique_zeo_codes])

    def __iter__(self):
        batch_indices = []

        # Sample indices for each zeolite code
        for zeo_code in self.unique_zeo_codes:
            zeo_code_indices = [idx for idx, value in enumerate(self.zeolite_codes) if zeo_code == value]
            repeats = np.floor(self.most_zeo_code // len(zeo_code_indices)).astype(int)
            n_to_sample = self.most_zeo_code % repeats
            zeo_code_indices = np.tile(zeo_code_indices, repeats).tolist()

            if n_to_sample > 0:
                zeo_code_indices.extend(np.random.choice(zeo_code_indices, size=n_to_sample, replace=False).tolist())

            batch_indices.extend(zeo_code_indices)

        # Shuffle and pad indices to fit batches
        np.random.shuffle(batch_indices)
        indices_to_add = self.batch_size - (len(batch_indices) % self.batch_size)
        if indices_to_add < self.batch_size:
            batch_indices.extend([-1] * indices_to_add)

        # Yield one batch at a time
        for i in range(0, len(batch_indices), self.batch_size):
            batch = batch_indices[i:i + self.batch_size]
            # print([idx for idx in batch if idx != -1])
            yield [idx for idx in batch if idx != -1]  # Remove padded indices

    def __len__(self):
        return int(np.ceil(len(self.unique_zeo_codes) * self.most_zeo_code / self.batch_size))