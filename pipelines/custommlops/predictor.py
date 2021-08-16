import torch
import os

import json
import numpy as np

from network import Network
from dataset import PMDataset_torch

import subprocess

cmd = ["pip", "install", "sagemaker-containers"]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
p.wait()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)

# # INFERENCE_ACCELERATOR_PRESENT_ENV = "SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT"
# DEFAULT_TS_MODEL_SERIALIZED_FILE = "net.pth"
# os.environ['SM_MODEL_DIR'] = "/opt/ml/model/output"


def input_fn(data, input_content_type):
    # if set content_type as 'image/jpg' or 'applicaiton/x-npy', 
    # the input is also a python bytearray

    content_type = input_content_type.lower(
    ) if input_content_type else "text/csv"
    content_type = content_type.split(";")[0].strip()

    if content_type == 'text/csv':
        predict_ds = PMDataset_torch(
                            data,
                            target_column="target",
                            standardize=True,
                            sensor_headers=["voltage", "current"])
    else:
        raise ValueError("{} not supported by script!".format(content_type))
    
    return predict_ds


def model_fn(model_dir):
    with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
        model_info = torch.load(f)
        pre_trained_model = model_info["net"]
        
        sensor_headers = model_info["sensor_headers"]
        fc_hidden_units = model_info["fc_hidden_units"]
        conv_channels = model_info["conv_channels"]
        
        net = Network(num_features=len(sensor_headers),
                      fc_hidden_units=fc_hidden_units,
                      conv_channels=conv_channels,
                      dropout_strength=0)

        net_dict = net.state_dict()
        
        weight_dict = {}
        for key, value in net_dict.items():
            if key not in pre_trained_model:
                key = "module." + key
            weight_dict[key] = pre_trained_model[key]
            
        for key, value in weight_dict.items():
            net_dict[key] = value
    print("Net loaded")
    net = net.to(device)
    return net


# Perform prediction on the deserialized object, with the loaded model
def predict_fn(predict_ds, model):
    
    input_data = torch.FloatTensor(predict_ds.data).to(device).contiguous()
    output = model(input_data).cpu().detach().numpy()
    prediction = output[:,0].round()
    print(f"prediction : {prediction}")
    
    return prediction