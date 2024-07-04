import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


def create_dataloaders(args):

    train_dataset = MyDataset(args)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)

    return train_dataloader


class MyDataset(Dataset):
    """
    数据处理类
    """
    def __init__(self, args):
        self.args = args
        self._load_data()
        self.window_num = self.time_num - self.args.seq_len - self.args.pred_len + 1

    def __len__(self):
        return self.window_num * self.station_num

    def _load_data(self):
        self.temp = np.load(os.path.join(self.args.data_path, "temp.npy")).squeeze(-1)  # [T, Station]
        self.wind = np.load(os.path.join(self.args.data_path, "wind.npy")).squeeze(-1)  # [T, Station]
        self.time_num, self.station_num = self.temp.shape
        self.era5 = np.load(os.path.join(self.args.data_path, "global_data.npy")).repeat(3, axis=0)[:self.time_num]
        self.era5 = self.era5.reshape((self.era5.shape[0], 4 * 9, self.era5.shape[-1]))  # [T, 36, Station]

    @staticmethod
    def seq_pad(sequence, max_len):
        if sequence is not None and isinstance(sequence, np.ndarray) and list(sequence) != [0]:
            sequence = list(sequence)
            seq_len = min(len(sequence), max_len)
            sequence = sequence[-max_len:] + [0] * (max_len - seq_len)
        else:
            sequence = [0] * max_len
            seq_len = 0
        return sequence, seq_len

    def __getitem__(self, idx):
        station = idx // self.window_num
        start = idx % self.window_num

        end = start + self.args.seq_len

        temp_x = self.temp[start: end, station: station + 1]   # [seq_len, 1]
        wind_x = self.wind[start: end, station: station + 1]   # [seq_len, 1]
        era5_x = self.era5[start: end, :, station: station + 1].squeeze()  # [seq_len, 36]

        temp_y = self.temp[end: end + self.args.pred_len, station: station + 1]  # [pred_len, 1]
        wind_y = self.wind[end: end + self.args.pred_len, station: station + 1]  # [pred_len, 1]
        era5_y = self.era5[end: end + self.args.pred_len, :, station: station + 1].squeeze()  # [pred_len, 36]

        x = np.concatenate([temp_x, wind_x, era5_x], axis=1)  # [seq_len, 38]
        y = np.concatenate([temp_y, wind_y, era5_y], axis=1)  # [pred_len, 38]

        data = dict(
            x=torch.FloatTensor(x),
            y=torch.FloatTensor(y),
        )

        return data


