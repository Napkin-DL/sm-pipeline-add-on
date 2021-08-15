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
"""Feature engineers the custom env"""
import argparse
import logging
import pathlib

import boto3
import numpy as np
import pandas as pd

import os
import sys
import subprocess
import time

sys.path.append("/opt/ml/processing")

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
except:
    print("Using Codebuild")
    pass
    
from source.config import Config
from source.preprocessing import pivot_data, sample_dataset
from source.dataset import DatasetGenerator

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--code-dir", type=str, required=True)
#     args = parser.parse_args()
    
    print(f"Start Preprocessing")
    
    base_dir = "/opt/ml/processing/"
    
        
    print(f"result  : {os.listdir(base_dir)}")
    config = Config(base_dir=base_dir, filename=base_dir + "source/config/config.yaml", fetch_sensor_headers=False)
    
    config.fleet_dataset_fn = base_dir + config.fleet_dataset_fn
    config.fleet_info_fn = base_dir + config.fleet_info_fn
    config.fleet_sensor_logs_fn = base_dir + config.fleet_sensor_logs_fn
    
    config.train_dataset_fn = base_dir + "train/train_dataset.csv"
    config.test_dataset_fn = base_dir + "test/test_dataset.csv"
    
    dirname = os.path.dirname(config.fleet_dataset_fn)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    dirname = os.path.dirname(config.train_dataset_fn)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    dirname = os.path.dirname(config.test_dataset_fn)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    should_generate_data = True    
    
    if should_generate_data:
        fleet_statistics_fn = base_dir + "data/generation/fleet_statistics.csv"
        generator = DatasetGenerator(fleet_statistics_fn=fleet_statistics_fn,
                                     fleet_info_fn=config.fleet_info_fn, 
                                     fleet_sensor_logs_fn=config.fleet_sensor_logs_fn, 
                                     period_ms=config.period_ms, 
                                     )
        generator.generate_dataset()

        assert os.path.exists(config.fleet_info_fn), "Please copy your data to {}".format(config.fleet_info_fn)
        assert os.path.exists(config.fleet_sensor_logs_fn), "Please copy your data to {}".format(config.fleet_sensor_logs_fn)
    
    
    pivot_data(config)
    sample_dataset(config)
    
    for root, dir, file in os.walk(base_dir):
        print(root, dir, file)