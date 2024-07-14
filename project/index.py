import os
import torch
import numpy as np
from itransformer import ITransformer
from utils import DictToClass, todevice
from data_helper import load_test_data
from sklearn.metrics import mean_squared_error

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def invoke(inputs):
    save_path = '/home/mw/project'

    data = load_test_data(inputs)
    data = todevice(data, device)

    models = {
        "./checkpoint/online_v1/all_0": {'ind': [0, 1], 'weight': 1},
        "./checkpoint/online_v1/all_2": {'ind': [0, 1], 'weight': 1},
        "./checkpoint/online_v1/temp_0": {'ind': 0, 'weight': 1},
        "./checkpoint/online_v1/wind_0": {'ind': 1, 'weight': 1},
        "./checkpoint/online_v1/temp_1": {'ind': 0, 'weight': 1},
        "./checkpoint/online_v1/wind_1": {'ind': 1, 'weight': 1},
        "./checkpoint/online_v1/temp_2": {'ind': 0, 'weight': 1},
        "./checkpoint/online_v1/wind_2": {'ind': 1, 'weight': 1}
    }

    result_temp, result_wind = 0, 0
    num_temp, num_wind = 0, 0
    for model_path, info in models.items():
        preds = inference(model_path, data)
        if info['ind'] == 0:
            result_temp += preds[info['ind']] * info['weight']
            num_temp += info['weight']
        elif info['ind'] == 1:
            result_wind += preds[info['ind']] * info['weight']
            num_wind += info['weight']
        else:
            result_temp += preds[info['ind'][0]] * info['weight']
            num_temp += info['weight']
            result_wind += preds[info['ind'][1]] * info['weight']
            num_wind += info['weight']

    result_temp = result_temp / num_temp
    result_wind = result_wind / num_wind
    result_wind = np.abs(result_wind)  # 后处理：风速不为负

    np.save(os.path.join(save_path, "temp_predict.npy"), result_temp)
    np.save(os.path.join(save_path, "wind_predict.npy"), result_wind)


def invoke_eval(inputs, models):
    data = load_test_data(inputs)
    data = todevice(data, device)

    result_temp, result_wind = 0, 0
    num_temp, num_wind = 0, 0
    for model_path, info in models.items():
        preds = inference(model_path, data)
        if info['ind'] == 0:
            result_temp += preds[info['ind']] * info['weight']
            num_temp += info['weight']
        elif info['ind'] == 1:
            result_wind += preds[info['ind']] * info['weight']
            num_wind += info['weight']
        else:
            result_temp += preds[info['ind'][0]] * info['weight']
            num_temp += info['weight']
            result_wind += preds[info['ind'][1]] * info['weight']
            num_wind += info['weight']

    result_temp = result_temp / num_temp
    result_wind = result_wind / num_wind
    result_wind = np.abs(result_wind)  # 后处理：风速不为负

    label_temp = np.load(os.path.join(inputs, "temp_lookback_label.npy"))
    label_wind = np.load(os.path.join(inputs, "wind_lookback_label.npy"))

    temp_var = np.var(label_temp)
    wind_var = np.var(label_wind)
    mse_temp = mean_squared_error(result_temp.flatten(), label_temp.flatten())
    mse_wind = mean_squared_error(result_wind.flatten(), label_wind.flatten())
    mse = mse_temp / temp_var * 10 + mse_wind / wind_var

    return mse, mse_temp, mse_wind


def inference(model_path, data):

    checkpoint = torch.load(f"{model_path}/model.bin", map_location='cpu')
    args = DictToClass(checkpoint['args'])

    model = ITransformer(args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    pred_temp, pred_wind = model(data, inference=True)  # [batch, pred_len, 1]

    pred_temp = pred_temp.detach().cpu().numpy()  # (N * S, P, 1)
    pred_wind = pred_wind.detach().cpu().numpy()  # (N * S, P, 1)

    P = pred_temp.shape[1]

    pred_temp = pred_temp.reshape((71, -1, P, 1)).transpose(0, 2, 1, 3)  # (N, P, S, 1)
    pred_wind = pred_wind.reshape((71, -1, P, 1)).transpose(0, 2, 1, 3)  # (N, P, S, 1)

    return pred_temp, pred_wind


