import os
import torch
import numpy as np
from model import ITransformer
from utils import DictToClass
from sklearn.metrics import mean_squared_error


def invoke(inputs):
    save_path = '/home/mw/project'

    model_path = "./checkpoint"
    checkpoint = torch.load(f"{model_path}/model.bin", map_location='cpu')
    args = DictToClass(checkpoint['args'])

    model = ITransformer(args).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    temp = np.load(os.path.join(inputs, "temp_lookback.npy"))  # (N, L, S, 1)
    wind = np.load(os.path.join(inputs, "wind_lookback.npy"))  # (N, L, S, 1)
    era5 = np.load(os.path.join(inputs, "cenn_data.npy")).repeat(3, axis=1)  # (N, L, 4, 9, S)
    era5 = era5.reshape((era5.shape[0], era5.shape[1], 4 * 9, era5.shape[-1])).transpose([0, 1, 3, 2])  # [N, L, S, 36]

    N, L, S, _ = temp.shape  # (N, L, S, 1) -> [71, 168, 60, 1]

    data = np.concatenate([temp, wind, era5], axis=-1).transpose([0, 2, 1, 3])  # (N, S, L, 38)
    data = {'x': torch.FloatTensor(data.reshape((N * S, L, -1))).cuda()}  # (N * S, L, 38)

    pred_temp, pred_wind = model(data, inference=True)  # [batch, pred_len, 1]

    pred_temp = pred_temp.detach().cpu().numpy()  # (N * S, P, 1)
    pred_wind = pred_wind.detach().cpu().numpy()  # (N * S, P, 1)

    P = pred_temp.shape[1]

    pred_temp = pred_temp.reshape(N, S, P, 1).transpose(0, 2, 1, 3)  # (N, P, S, 1)
    pred_wind = pred_wind.reshape(N, S, P, 1).transpose(0, 2, 1, 3)  # (N, P, S, 1)
    # pred_wind[pred_wind < 0] = 0  # 风速不为负，负数置0

    np.save(os.path.join(save_path, "temp_predict.npy"), pred_temp)
    np.save(os.path.join(save_path, "wind_predict.npy"), pred_wind)


def invoke_eval(inputs):
    model_path = "./checkpoint"
    checkpoint = torch.load(f"{model_path}/model.bin", map_location='cpu')
    args = DictToClass(checkpoint['args'])

    model = ITransformer(args).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    temp = np.load(os.path.join(inputs, "temp_lookback.npy"))  # (N, L, S, 1)
    wind = np.load(os.path.join(inputs, "wind_lookback.npy"))  # (N, L, S, 1)
    era5 = np.load(os.path.join(inputs, "cenn_data.npy")).repeat(3, axis=1)  # (N, L, 4, 9, S)
    era5 = era5.reshape((era5.shape[0], era5.shape[1], 4 * 9, era5.shape[-1])).transpose([0, 1, 3, 2])  # [N, L, S, 36]

    N, L, S, _ = temp.shape  # (N, L, S, 1) -> [71, 168, 60, 1]

    data = np.concatenate([temp, wind, era5], axis=-1).transpose([0, 2, 1, 3])  # (N, S, L, 38)
    data = {'x': torch.FloatTensor(data.reshape((N * S, L, -1))).cuda()}  # (N * S, L, 38)

    pred_temp, pred_wind = model(data, inference=True)  # [batch, pred_len, 1]

    pred_temp = pred_temp.detach().cpu().numpy()  # (N * S, P, 1)
    pred_wind = pred_wind.detach().cpu().numpy()  # (N * S, P, 1)

    P = pred_temp.shape[1]

    pred_temp = pred_temp.reshape(N, S, P, 1).transpose(0, 2, 1, 3)  # (N, P, S, 1)
    pred_wind = pred_wind.reshape(N, S, P, 1).transpose(0, 2, 1, 3)  # (N, P, S, 1)
    # pred_wind[pred_wind < 0] = 0  # 风速不为负，负数置0

    label_temp = np.load(os.path.join(inputs, "temp_lookback_label.npy"))
    label_wind = np.load(os.path.join(inputs, "wind_lookback_label.npy"))

    temp_var = np.var(label_temp)
    wind_var = np.var(label_wind)
    mse_temp = mean_squared_error(pred_temp.flatten(), label_temp.flatten())
    mse_wind = mean_squared_error(pred_wind.flatten(), label_wind.flatten())
    mse = mse_temp / temp_var * 10 + mse_wind / wind_var

    return mse, mse_temp, mse_wind

