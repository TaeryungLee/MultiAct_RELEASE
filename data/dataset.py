import torch
import random
import numpy as np

from data.datasets.BABEL import BABEL


class MultipleDatasets(torch.utils.data.Dataset):
    def __init__(self, dbs, make_same_len=True):
        """
            From Pose2Pose code by Gyeongsik Moon.
        """
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in dbs])
        self.make_same_len = make_same_len

    def __len__(self):
        # all dbs have the same length
        if self.make_same_len:
            return self.max_db_data_num * self.db_num
        # each db has different length
        else:
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        if self.make_same_len:
            db_idx = index // self.max_db_data_num
            data_idx = index % self.max_db_data_num 
            if data_idx >= len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])): # last batch: random sampling
                data_idx = random.randint(0,len(self.dbs[db_idx])-1)
            else: # before last batch: use modular
                data_idx = data_idx % len(self.dbs[db_idx])
        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx-1]

        return self.dbs[db_idx][data_idx]
          

def get_dataloader(cfg, split="train", sampled_file_path=None, sampled=None, filter_label=None, cap=None, drop_last=True):

    # action datasets
    action_datasets = []
    if cfg.action_dataset == "BABEL":
        action_datasets.append(BABEL(cfg, split, sampled_file_path, sampled, filter_label, cap))
    else:
        raise NotImplementedError("unknown dataset")

    action_dataset = MultipleDatasets(action_datasets, make_same_len=True)
    action_dataloader = torch.utils.data.DataLoader(
        dataset=action_dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=drop_last
    )

    return action_dataloader