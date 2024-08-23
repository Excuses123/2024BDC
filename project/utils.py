import torch
import random
import numpy as np
import os.path
import zipfile


def zip_specific_files(directory, files_to_zip, zip_name, target_path=None):
    # directory:str         导出的文件所在的文件夹
    # files_to_zip:[str]    待打包的文件名称
    # zip_name:str          导出文件的名称
    # target_path:str       导出的 zip 文件的路径
    # Ensure the directory and files_to_zip are valid
    cwd = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(directory):
        raise ValueError("The provided directory does not exist")
    if not all(isinstance(file, str) for file in files_to_zip):
        raise ValueError("files_to_zip should be a list of file names")
    if target_path and not os.path.isdir(target_path):
        raise ValueError("The provided target path does not exist")
    zip_file_path = target_path if target_path else cwd
    with zipfile.ZipFile(os.path.join(zip_file_path, zip_name + '.zip'), 'w') as zipf:
        for file in files_to_zip:
            file_path = os.path.join(directory, file)
            print("Packing files: {}".format(file_path), flush=True)
            if os.path.isfile(file_path):
                zipf.write(file_path, arcname=file)
    print("Files zipped: {}".format(files_to_zip), flush=True)
    print(f"{zip_name} created successfully with specified files", flush=True)


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def todevice(inputs, device):
    if isinstance(inputs, dict):
        return {k: todevice(v, device) for k, v in inputs.items()}
    elif isinstance(inputs, (tuple, list)):
        return (todevice(v, device) for v in inputs)
    elif isinstance(inputs, torch.Tensor):
        kwargs = {"device": device}
        return inputs.to(**kwargs)


class DictToClass(object):
    def __init__(self, _obj=None):
        if _obj:
            self.__dict__.update(_obj)


def save_model(args, model, epoch, step, save_path):
    model_dict = {
        "args": {k: v for k, v in args.__dict__.items()},
        "epoch": epoch,
        "step": step
    }
    if isinstance(model, (torch.nn.parallel.DataParallel,
                          torch.nn.parallel.DistributedDataParallel)):
        model_dict['model_state_dict'] = model.module.state_dict()
    else:
        model_dict['model_state_dict'] = model.state_dict()

    torch.save(model_dict, save_path)


