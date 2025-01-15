from torch.utils.data import BatchSampler
from torch.utils.data import Dataset
from numpy.random import choice, shuffle
import numpy as np


class ZeoSampler(BatchSampler):
    def __init__(self, zeolite_codes: list, batch_size, sample='over_sample', n_samples=None, origin=""):
        '''
        Custom batch sampler for balanced sampling of zeolites
        sub_sample (bool): If True, in a single epoch, each zeolite will be sampled the number of times of the least occuring zeolite
        If False, each zeolite will be sampled the number of times of the most occuring zeolite by repeating the indices and sampling the remaining indices randomly
        '''

        assert sample in ['over_sample', 'under_sample'] or isinstance(n_samples, int), 'sample must be one of ["over_sample", "under_sample"] or n_samples must be an integer'


        self.batch_size = batch_size
        self.sample = sample
        self.origin = origin

        self.zeolite_codes = zeolite_codes
        self.unique_zeo_codes = set(zeolite_codes)
        self.unique_zeo_codes_num = len(self.unique_zeo_codes)
        counts_per_zeo_code = {}
        for zeo_code in self.unique_zeo_codes:
            counts_per_zeo_code.update({zeo_code: self.zeolite_codes.count(zeo_code)})
        
        if n_samples is not None:
            self.samples_per_zeo = n_samples
            self.sample = 'user_defined'
        else:
            if self.sample == 'under_sample':
                self.samples_per_zeo = min([zeolite_codes.count(zeo_code) for zeo_code in self.unique_zeo_codes])
            else:
                self.samples_per_zeo = max([zeolite_codes.count(zeo_code) for zeo_code in self.unique_zeo_codes])

    def __iter__(self):
        batch_indices = []

        # Sample indices for each zeolite code
        for zeo_code in self.unique_zeo_codes:
            zeo_code_indices = [idx for idx, value in enumerate(self.zeolite_codes) if zeo_code == value]
            
            if self.sample == 'under_sample' or (self.sample == 'user_defined' and self.samples_per_zeo < len(zeo_code_indices)):
                zeo_code_indices = np.random.choice(zeo_code_indices, size=self.samples_per_zeo, replace=False).tolist()
            else: 
                repeats = np.floor(self.samples_per_zeo // len(zeo_code_indices)).astype(int)

                n_to_sample = self.samples_per_zeo - (len(zeo_code_indices) * repeats)

                zeo_code_indices = np.tile(zeo_code_indices, repeats).tolist()

                if n_to_sample > 0:
                    zeo_code_indices.extend(np.random.choice(zeo_code_indices, size=n_to_sample, replace=False).tolist())

            batch_indices.extend(zeo_code_indices)

        # Shuffle and pad indices to fit batches
        np.random.shuffle(batch_indices)
        # This takes care of the last batch which might not be full due to
        # non integer multiplication between the batch size and number of indices
        indices_to_add = self.batch_size - (len(batch_indices) % self.batch_size)
        if indices_to_add < self.batch_size:
            batch_indices.extend([-1] * indices_to_add)

        for i in range(0, len(batch_indices), self.batch_size):
            batch = batch_indices[i:i + self.batch_size]
            # print([idx for idx in batch if idx != -1])
            yield [idx for idx in batch if idx != -1]  # Remove padded indices

    def __len__(self):
        return int(np.ceil((len(self.unique_zeo_codes) * self.samples_per_zeo) / self.batch_size))