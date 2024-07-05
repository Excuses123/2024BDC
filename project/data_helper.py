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

        x = feature_engineer(temp_x, wind_x, era5_x)  # [seq_len, -1]

        temp_y = self.temp[end: end + self.args.pred_len, station: station + 1]  # [pred_len, 1]
        wind_y = self.wind[end: end + self.args.pred_len, station: station + 1]  # [pred_len, 1]
        era5_y = self.era5[end: end + self.args.pred_len, :, station: station + 1].squeeze()  # [pred_len, 36]

        y = feature_engineer(temp_y, wind_y, era5_y)  # [pred_len, -1]

        data = dict(
            x=torch.FloatTensor(x),
            y=torch.FloatTensor(y),
        )

        return data


def feature_engineer(temp, wind, era5):
    """
    temp: [L, 1] | [N, L, 1]
    wind: [L, 1] | [N, L, 1]
    era5: [L, 1] | [N, L, 1]
    """
    def check_shape(arr):
        if len(arr.shape) == 2:
            return arr[None, :, :]
        elif len(temp.shape) == 3:
            return arr
        else:
            raise Exception(f"arr shape: {arr.shape} do not meet the requirements!")

    temp = check_shape(temp)
    wind = check_shape(wind)
    era5 = check_shape(era5)

    N, L, _ = temp.shape

    temp_diff = np.diff(temp, axis=1, prepend=temp[:, :1, :])  # (N, L, 1)
    wind_diff = np.diff(wind, axis=1, prepend=wind[:, :1, :])  # (N, L, 1)

    # L长度不一样，处理特征需注意，不能穿越
    # temp_mean = temp.mean(axis=1)[:, :, None].repeat(L, axis=1)
    # wind_mean = wind.mean(axis=1)[:, :, None].repeat(L, axis=1)

    temp_lag1 = np.concatenate([np.zeros_like(temp[:, :1, :]), temp[:, :-1, :]], axis=1)
    wind_lag1 = np.concatenate([np.zeros_like(wind[:, :1, :]), wind[:, :-1, :]], axis=1)

    temp_wind = temp - wind

    temp_abs = np.abs(temp)

    feat = np.concatenate([temp, wind, temp_diff, wind_diff, era5,
                           temp_wind, temp_abs, temp_lag1, wind_lag1], axis=-1)  # (N, L, -1)

    if feat.shape[0] == 1:
        feat = feat.squeeze(axis=0)

    return feat


def load_test_data(data_path, label=False):
    temp = np.load(os.path.join(data_path, "temp_lookback.npy"))  # (N, L, S, 1)
    wind = np.load(os.path.join(data_path, "wind_lookback.npy"))  # (N, L, S, 1)
    era5 = np.load(os.path.join(data_path, "cenn_data.npy"))  # (N, L/3, 4, 9, S)

    N, L, S, _ = temp.shape  # (N, L, S, 1) -> [71, 168, 60, 1]

    temp = temp.transpose([0, 2, 1, 3]).reshape((N * S, L, -1))  # [N * S, L, 1]
    wind = wind.transpose([0, 2, 1, 3]).reshape((N * S, L, -1))  # [N * S, L, 1]
    era5 = era5.repeat(3, axis=1).reshape((N, L, 4 * 9, S)).transpose([0, 1, 3, 2])  # [N, L, S, 36]
    era5 = era5.transpose([0, 2, 1, 3]).reshape((N * S, L, -1))  # [N * S, L, 36]

    x = feature_engineer(temp, wind, era5)  # [N * S, L, -1]

    data = {
        'x': torch.FloatTensor(x),  # (N * S, L, -1)
    }

    if label:
        label_temp = np.load(
            os.path.join(data_path, "temp_lookback_label.npy")).transpose([0, 2, 1, 3]).reshape((N * S, -1, 1))
        label_wind = np.load(
            os.path.join(data_path, "wind_lookback_label.npy")).transpose([0, 2, 1, 3]).reshape((N * S, -1, 1))

        data['label_temp'] = torch.FloatTensor(label_temp)  # (N * S, P, 1)
        data['label_wind'] = torch.FloatTensor(label_wind)  # (N * S, P, 1)

    return data

