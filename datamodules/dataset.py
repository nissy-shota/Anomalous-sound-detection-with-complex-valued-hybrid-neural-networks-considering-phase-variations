from typing import Dict, List
from itertools import chain

import numpy as np
import torch
from omegaconf import DictConfig
from utils import util_dirs


class ToyADMOS(torch.utils.data.Dataset):
    def __init__(
        self,
        unique_case_ids: List[str],
        target_dir: str,
        cfg: DictConfig,
        transform=None,
    ):
        super().__init__()
        self.transform = transform
        self.target_dir = target_dir
        self.unique_case_ids = unique_case_ids

        dirs_ea_case = []
        n_dirs_ea_case = []

        for unique_case_id in self.unique_case_ids:
            _dirs = util_dirs.generate_dir_list(
                self.target_dir, unique_case_id, mode="train"
            )
            dirs_ea_case.append(_dirs)
            n_dirs_ea_case.append(len(_dirs))
            print(f"number of {unique_case_id}'s dir: {str(len(_dirs))}")

        self.dir_list = list(chain.from_iterable(dirs_ea_case))
        labels = np.zeros(len(self.dir_list), dtype=int)

        case_label = 0
        start_index = 0
        for num_file in n_dirs_ea_case:
            end_index = start_index + num_file
            labels[start_index:end_index] = case_label
            start_index += num_file
            case_label += 1

        self.labels = labels
        assert len(self.labels) == len(self.dir_list)

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, index):
        sample_dir = self.dir_list[index]
        sample = self.transform(sample_dir)
        label = self.labels[index]

        return sample, label


class FeatureExtractedToyADMOS(torch.utils.data.Dataset):
    def __init__(
        self,
        unique_case_ids: List[str],
        target_dir: str,
        cfg: DictConfig,
        transform=None,
    ):
        super().__init__()
        self.transform = transform
        self.target_dir = target_dir
        self.unique_case_ids = unique_case_ids
        dirs_ea_case = []
        n_dirs_ea_case = []

        for unique_case_id in self.unique_case_ids:
            _dirs = util_dirs.generate_dir_list(
                self.target_dir, unique_case_id, mode="train"
            )
            dirs_ea_case.append(_dirs)
            n_dirs_ea_case.append(len(_dirs))
            print(f"number of {unique_case_id}'s dir: {str(len(_dirs))}")

        self.dir_list = list(chain.from_iterable(dirs_ea_case))
        labels = np.zeros(len(self.dir_list), dtype=int)

        case_label = 0
        start_index = 0
        for num_file in n_dirs_ea_case:
            end_index = start_index + num_file
            labels[start_index:end_index] = case_label
            start_index += num_file
            case_label += 1

        self.labels = labels
        assert len(self.labels) == len(self.dir_list)

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, index):
        sample_dir = self.dir_list[index]
        sample = self.transform(sample_dir)
        label = self.labels[index]

        return sample, label
