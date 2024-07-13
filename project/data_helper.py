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
        print(f"time_num: {self.time_num}, station_num: {self.station_num}")

    def __len__(self):
        return self.window_num * self.station_num

    def _load_data(self):
        self.temp = np.load(os.path.join(self.args.data_path, "temp.npy")).squeeze(-1)  # [T, S]
        self.wind = np.load(os.path.join(self.args.data_path, "wind.npy")).squeeze(-1)  # [T, S]
        self.era5 = np.load(os.path.join(self.args.data_path, "global_data.npy"))  # [T/3, 4, 9, S]

        if self.args.outlier_strategy == 1:
            # 数据处理，过滤掉变量标准差<=1的站点
            if self.args.pred_var != 'all':
                std = self.temp.std(axis=0) if self.args.pred_var == 'temp' else self.wind.std(axis=0)
                ind = np.where(std > 1)[0]
                self.temp = self.temp[:, ind]
                self.wind = self.wind[:, ind]
                self.era5 = self.era5[:, :, :, ind]

        if self.args.outlier_strategy == 2:
            # 数据处理，过滤掉变量标准差<=1的站点，属于异常数据
            temp_std = self.temp.std(axis=0)
            temp_ind = np.where(temp_std > 1)[0]

            wind_std = self.wind.std(axis=0)
            wind_ind = np.where(wind_std > 1)[0]

            ind = np.intersect1d(temp_ind, wind_ind)
            self.temp = self.temp[:, ind]
            self.wind = self.wind[:, ind]
            self.era5 = self.era5[:, :, :, ind]

        self.time_num, self.station_num = self.temp.shape
        self.era5 = self.era5.transpose([0, 2, 1, 3]).reshape((self.time_num, 3, 4, self.station_num)).transpose([0, 2, 1, 3])  # [T,4,3,S]

    def __getitem__(self, idx):
        station = idx // self.window_num
        start = idx % self.window_num

        end = start + self.args.seq_len

        temp_x = self.temp[start: end, station: station + 1]   # [seq_len, 1]
        wind_x = self.wind[start: end, station: station + 1]   # [seq_len, 1]
        era5_x = self.era5[start: end, :, :, station: station + 1].squeeze()  # [seq_len, 4, 3]

        x = feature_engineer(temp_x, wind_x, era5_x)  # [seq_len, -1]

        temp_y = self.temp[end: end + self.args.pred_len, station: station + 1]  # [pred_len, 1]
        wind_y = self.wind[end: end + self.args.pred_len, station: station + 1]  # [pred_len, 1]

        data = dict(
            x=torch.FloatTensor(x),
            label_temp=torch.FloatTensor(temp_y),
            label_wind=torch.FloatTensor(wind_y)
        )

        return data


def feature_engineer(temp, wind, era5):
    """
    temp: [L, 1] | [N, L, 1]
    wind: [L, 1] | [N, L, 1]
    era5: [L, 4, 3] | [N, L, 4, 3]
    """
    if len(temp.shape) == 2:
        temp = temp[None, :, :]     # [N, L, 1]
    if len(wind.shape) == 2:
        wind = wind[None, :, :]     # [N, L, 1]
    if len(era5.shape) == 3:
        era5 = era5[None, :, :, :]  # [N, L, 4, 3]

    N, L, _ = temp.shape

    # temp, wind 衍生特征
    temp_cumavg = cum_avg(temp.squeeze(-1)).reshape((N, L, -1))   # (N, L, 1)
    wind_cumavg = cum_avg(wind.squeeze(-1)).reshape((N, L, -1))    # (N, L, 1)

    temp_lag1 = np.concatenate([np.zeros_like(temp[:, :1, :]), temp[:, :-1, :]], axis=1)
    wind_lag1 = np.concatenate([np.zeros_like(wind[:, :1, :]), wind[:, :-1, :]], axis=1)

    temp_abs = np.abs(temp)
    temp_wind = temp - wind

    # era5 衍生特征
    """
    1、十米高度的矢量纬向风速10U，正方向为东方（m/s）  范围：(-14.8721923828125, 20.9344482421875)
    2、十米高度的矢量经向风速10V，正方向为北方（m/s）  范围：(-15.188034057617188, 14.637161254882812)
    3、两米高度的温度值T2M（℃）                    范围：(-37.32951965332029, 39.07477722167971)
    4、均一海平面气压MSL（Pa）                     范围：(96975.6875, 105129.1875)
    """
    era5_flatten = era5.reshape((era5.shape[0], era5.shape[1], -1))  # [N, L, 4 * 3]

    # era5_1_var = era5_1.std(axis=-1)[:, :, None]
    # era5_2_var = era5_2.std(axis=-1)[:, :, None]
    # era5_3_var = era5_3.std(axis=-1)[:, :, None]
    # era5_4_var = era5_4.std(axis=-1)[:, :, None]

    # temp, wind 与 era5 交叉特征
    # wind_era5_1_abs = wind - era5_1_abs[:, :, 4:5]
    # wind_era5_2_abs = wind - era5_2_abs[:, :, 4:5]
    # temp_era5_3 = temp - era5_3[:, :, 4:5]
    # temp_era5_4_mean = temp / era5_4_mean

    feat = np.concatenate([temp, wind, era5_flatten, temp_lag1, wind_lag1, temp_abs, temp_wind,
                           temp_cumavg, wind_cumavg], axis=-1)  # (N, L, -1)

    if feat.shape[0] == 1:
        feat = feat.squeeze(axis=0)

    return feat


def load_test_data(data_path, label=False):
    temp = np.load(os.path.join(data_path, "temp_lookback.npy"))  # (N, L, S, 1)
    wind = np.load(os.path.join(data_path, "wind_lookback.npy"))  # (N, L, S, 1)
    # era5 = np.load(os.path.join(data_path, "cenn_data.npy")).repeat(3, axis=1)  # (N, L, 4, 9, S)
    era5 = np.load(os.path.join(data_path, "cenn_data.npy"))  # (N, L/3, 4, 9, S)

    N, L, S, _ = temp.shape  # (N, L, S, 1) -> [71, 168, 60, 1]

    temp = temp.transpose([0, 2, 1, 3]).reshape((N * S, L, -1))  # [N * S, L, 1]
    wind = wind.transpose([0, 2, 1, 3]).reshape((N * S, L, -1))  # [N * S, L, 1]
    era5 = era5.transpose([0, 4, 1, 3, 2]).reshape((N * S, L, 3, 4)).transpose([0, 1, 3, 2])  # [N * S, L, 4, 3]

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


def cum_avg(arr):
    """
    arr: [N, L]
    """
    avg_arr = np.zeros_like(arr)  # [N, L]

    for i, row in enumerate(arr):
        cum_sum = np.cumsum(row)  # 计算累积和
        avg_row = cum_sum / np.arange(1, len(cum_sum) + 1)  # 计算累积均值
        avg_arr[i, :] = avg_row

    return avg_arr
