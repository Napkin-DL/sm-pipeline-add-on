# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Evaluation script for measuring model accuracy."""
import argparse
import json
import logging
import os
import pickle
import tarfile
import subprocess
import sys
import boto3
import time

sys.path.append("/opt/ml/processing")

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
except:
    print("Using Codebuild")
    pass

import pandas as pd
import torch
import numpy as np

from source.config import Config
from source.dl_utils.network import Network
from source.dl_utils.dataset import PMDataset_torch

# from torchsummary import summary

# May need to import additional metrics depending on what you are measuring.
# See https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from importlib import import_module


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
      
        
if __name__ == "__main__":
    
    base_dir = "/opt/ml/processing/"
    

    config = Config(base_dir=base_dir, filename=base_dir + "source/config/config.yaml")

    model_path = os.path.join(base_dir, "model/model.tar.gz")
    with tarfile.open(model_path) as tar:
        tar.extractall(path=base_dir)

    for root, dir, file in os.walk(base_dir):
        print(root, dir, file)
        
    logger.debug("Loading model.")
    with open(os.path.join(base_dir, "model.pth"), 'rb') as f:
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

    if torch.cuda.is_available() :
        device = torch.device('cuda')
    else : 
        device = torch.device('cpu')

    model = net.to(device)

    # summary(model, input_size=(1,20, 2), device=device.type) # summary 함수를 통해 임의의 사이즈를 넣어 구조와 파라미터를 확인할 수 있습니다


    print("Loading test input data")
    test_path = os.path.join(base_dir, "test/test_dataset.csv")
    test_ds = PMDataset_torch(
        test_path,
        sensor_headers=config.sensor_headers,
        target_column=config.target_column,
        standardize=True)

    logger.debug("Read test data.")

    num = 120

    chioce_array=np.random.choice(test_ds.data.shape[0], num, replace=False)

    data = torch.from_numpy(test_ds.data[chioce_array]).float().contiguous()
    y_test = test_ds.labels[chioce_array]
    logger.info("Performing predictions against test data.")
    predictions = model(data)

    predictions = predictions.cpu().detach().numpy()

    total_acc = 0.0
    total_auc = 0.0
    print("Creating classification evaluation report")

    pred = predictions[:, 1]
    total_acc += accuracy_score(y_test, pred>0.5)
    total_auc += roc_auc_score(y_test, pred)

    # The metrics reported can change based on the model used, but it must be a specific name per (https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html)
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {
                "value": total_acc,
                "standard_deviation": "NaN",
            },
            "auc": {"value": total_auc, "standard_deviation": "NaN"},
        },
    }

    print("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join(base_dir + "/evaluation", "evaluation.json")
    print("Saving classification report to {}".format(evaluation_output_path))
    
    dirname = os.path.dirname(evaluation_output_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))