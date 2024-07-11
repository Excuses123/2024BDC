import os
import torch
import numpy as np
from itransformer import ITransformer
from utils import DictToClass, todevice
from data_helper import load_test_data
from sklearn.metrics import mean_squared_error

S = 60
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def invoke(inputs):
    save_path = '/home/mw/project'

    model_path = "./checkpoint/online"
    checkpoint = torch.load(f"{model_path}/model.bin", map_location='cpu')
    args = DictToClass(checkpoint['args'])

    model = ITransformer(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    data = load_test_data(inputs)
    data = todevice(data, device)

    pred_temp, pred_wind = model(data, inference=True)  # [batch, pred_len, 1]

    pred_temp = pred_temp.detach().cpu().numpy()  # (N * S, P, 1)
    pred_wind = pred_wind.detach().cpu().numpy()  # (N * S, P, 1)

    P = pred_temp.shape[1]

    pred_temp = pred_temp.reshape((-1, S, P, 1)).transpose(0, 2, 1, 3)  # (N, P, S, 1)
    pred_wind = pred_wind.reshape((-1, S, P, 1)).transpose(0, 2, 1, 3)  # (N, P, S, 1)
    # pred_wind = np.abs(pred_wind)  # 风速不为负

    np.save(os.path.join(save_path, "temp_predict.npy"), pred_temp)
    np.save(os.path.join(save_path, "wind_predict.npy"), pred_wind)


def invoke_eval(inputs):
    model_path = "./checkpoint"
    checkpoint = torch.load(f"{model_path}/model.bin", map_location='cpu')
    args = DictToClass(checkpoint['args'])

    model = ITransformer(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    data = load_test_data(inputs)
    data = todevice(data, device)

    pred_temp, pred_wind = model(data, inference=True)  # [batch, pred_len, 1]

    pred_temp = pred_temp.detach().cpu().numpy()  # (N * S, P, 1)
    pred_wind = pred_wind.detach().cpu().numpy()  # (N * S, P, 1)

    P = pred_temp.shape[1]

    pred_temp = pred_temp.reshape((-1, S, P, 1)).transpose(0, 2, 1, 3)  # (N, P, S, 1)
    pred_wind = pred_wind.reshape((-1, S, P, 1)).transpose(0, 2, 1, 3)  # (N, P, S, 1)
    # pred_wind = np.abs(pred_wind)  # 风速不为负

    label_temp = np.load(os.path.join(inputs, "temp_lookback_label.npy"))
    label_wind = np.load(os.path.join(inputs, "wind_lookback_label.npy"))

    temp_var = np.var(label_temp)
    wind_var = np.var(label_wind)
    mse_temp = mean_squared_error(pred_temp.flatten(), label_temp.flatten())
    mse_wind = mean_squared_error(pred_wind.flatten(), label_wind.flatten())
    mse = mse_temp / temp_var * 10 + mse_wind / wind_var

    return mse, mse_temp, mse_wind

