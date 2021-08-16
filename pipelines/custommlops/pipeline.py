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
"""Example workflow pipeline script for CustomerChurn pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""

import os
import sys
import json
import subprocess

import boto3
import sagemaker
import sagemaker.session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput, CreateModelInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep, JsonGet
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig, TuningStep, CreateModelStep

from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
from sagemaker.pytorch import PyTorch
from sagemaker.model import Model
from sagemaker.s3 import S3Uploader

from sagemaker.workflow.functions import Join


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

SOURCE_DIR = 'source'
DATA_DIR = 'data'

def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_config(SOURCE_DIR, config_path='source/config/config.yaml'):
    
    from source.config import Config
    config = Config(base_dir='.', filename=config_path)
    
    with open(SOURCE_DIR + "/stack_outputs.json") as f:
        sagemaker_configs = json.load(f)
    return config, sagemaker_configs


    
def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="CustomMLOpsPackageGroup",  # Choose any name
    pipeline_name="CustomMLOpsDemo-p",  # You can find your pipeline name in the Studio UI (project -> Pipelines -> name)
    base_job_prefix="CustomMLOps",  # Choose any name
):
    """Gets a SageMaker ML Pipeline instance working.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
    
    code_repo = f"s3://{default_bucket}/{SOURCE_DIR}"
    cmd = ["aws", "s3", "sync", "--quiet", SOURCE_DIR, code_repo]
    print(f"Syncing files from {SOURCE_DIR} to {code_repo}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    
    data_repo = f"s3://{default_bucket}/{DATA_DIR}"
    
    cmd = ["aws", "s3", "sync", "--quiet", DATA_DIR, data_repo]
    print(f"Syncing files from {DATA_DIR} to {data_repo}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
            
    config, sagemaker_configs = get_config(SOURCE_DIR)
    
    print(f" config : {config}")
    

    # Parameters for pipeline execution
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.c5.4xlarge"
    )
    
    training_instance_count = ParameterInteger(
        name="TrainingInstanceCount",
        default_value=1
    )
    
    code_repo_name = ParameterString(
        name="CodeRepoName",
        default_value=default_bucket, 
    )
    
    src_dir = ParameterString(
        name="SourceDir",
        default_value=SOURCE_DIR,  # Change this to point to the s3 location of your raw input data.
    )
    
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
    )
    
    code_input = ProcessingInput(
                        source=code_repo,
                        destination=f"/opt/ml/processing/source",
                    )
    data_input = ProcessingInput(
                        source=data_repo,
                        destination="/opt/ml/processing/data",
                    )
    
    
    print(f"Processing step for feature engineering")
    # Processing step for feature engineering
#     sklearn_processor = SKLearnProcessor(
#         framework_version="0.23-1",
#         instance_type=processing_instance_type,
#         instance_count=processing_instance_count,
#         base_job_name=f"{base_job_prefix}/sklearn-CustomMLOps-preprocess",  # choose any name
#         sagemaker_session=sagemaker_session,
#         role=role,
#     )
        
    
    image_uri = sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="0.23-1",
        py_version="py3",
        instance_type=processing_instance_type,
    )

    # Processing step for evaluation
    sklearn_processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-CustomMLOps-preprocess",  # choose any name
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    
    step_process = ProcessingStep(
        name="CustomMLOpsProcess",  # choose any name
        processor=sklearn_processor,
        inputs=[
                code_input,
                data_input
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
    )
    
    print(f"Training step for generating model artifacts")
    # Training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/CustomMLOpsTrain/"


    image_uri_torch = sagemaker.image_uris.retrieve(
        framework="pytorch",
        region=region,
        version="1.5.0",
        py_version="py3",
        instance_type=processing_instance_type,
        image_scope='inference'
    )
    
    max_jobs = 4
    max_parallel_jobs = 2

    max_jobs = 1
    max_parallel_jobs = 1
    

    metric_definitions = [
        {'Name': 'Epoch', 'Regex': 'Epoch: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)'},
        {'Name': 'train_loss', 'Regex': 'Train loss: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)'},
        {'Name': 'train_acc',  'Regex': 'Train acc: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)'},
        {'Name': 'train_auc',  'Regex': 'Train auc: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)'},
        {'Name': 'test_loss', 'Regex': 'Test loss: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)'},
        {'Name': 'test_acc', 'Regex': 'Test acc: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)'},
        {'Name': 'test_auc', 'Regex': 'Test auc: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?)'},
    ]
    
    hyperparameter_ranges = {
        'lr': ContinuousParameter(1e-5, 1e-2),
        'batch_size': IntegerParameter(100, 256),
        'dropout': ContinuousParameter(0.0, 0.8),

        'fc_hidden_units': CategoricalParameter(["[256, 128]", "[256, 128, 128]", "[256, 256, 128]", "[256, 128, 64]"]),
        'conv_channels': CategoricalParameter(["[2, 8, 2]", "[2, 16, 2]", "[2, 16, 16, 2]"]),
    }
    
    estimator = PyTorch(entry_point="train.py",
                    source_dir=src_dir,                    
                    role=role,
                    dependencies=[src_dir + "/dl_utils"],
                    instance_type=training_instance_type,
                    instance_count=training_instance_count,
                    output_path=model_path,
                    framework_version="1.5.0",
                    py_version='py3',
                    base_job_name=f"{base_job_prefix}/CustomMLOps-train",
                    metric_definitions=metric_definitions,
                    hyperparameters= {
                        'epoch': 100,  # tune it according to your need
                        'target_column': config.target_column,
                        'sensor_headers': json.dumps(config.sensor_headers),
                        }
                     )
    
    tuner = HyperparameterTuner(estimator,
                                objective_metric_name='test_auc',
                                objective_type='Maximize',
                                hyperparameter_ranges=hyperparameter_ranges,
                                metric_definitions=metric_definitions,
                                max_jobs=max_jobs,
                                max_parallel_jobs=max_parallel_jobs,
                                base_tuning_job_name=base_job_prefix)

    
    input_train_data = TrainingInput(
                            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                                "train"
                            ].S3Output.S3Uri,
                            content_type="text/csv",
                        )
    
    
    input_test_data = TrainingInput(
                            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                                "test"
                            ].S3Output.S3Uri,
                            content_type="text/csv",
                        )
    
    step_tuning = TuningStep(
                    name="CustomMLOpsTuner",
                    tuner=tuner,
                    inputs={
                        "train": input_train_data,
                        "test": input_test_data,
                    },
#                     cache_config=cache_config
                )


    # Processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri_torch,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/script-CustomMLOps-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    
    
    model_input = ProcessingInput(
                        source=Join(values=[model_path, step_tuning.properties.BestTrainingJob.TrainingJobName,"/output/model.tar.gz"]),
                        destination="/opt/ml/processing/model",
                    )
    
    
    test_input = ProcessingInput(
                        source=step_process.properties.ProcessingOutputConfig.Outputs[
                            "test"
                        ].S3Output.S3Uri,
                        destination="/opt/ml/processing/test",
                    )

    step_eval = ProcessingStep(
        name="CustomMLOpsEval",
        processor=script_eval,
        inputs=[
            code_input,
            data_input,
            model_input,
            test_input
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
    )

    # Register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )
    
    model_output = Join(values=[model_path, step_tuning.properties.BestTrainingJob.TrainingJobName,"/output/model.tar.gz"])
    
    
    create_model = Model(
        name="CustomModel",
        model_data=model_output,
        image_uri=image_uri_torch,
        env = {"SAGEMAKER_PROGRAM": "predictor.py"},
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    
    input_model = CreateModelInput(
                           instance_type="ml.m5.xlarge"
                        )
    
    
    step_model = CreateModelStep(
        name="CustomMLOpsCreateModel",
        model=create_model,
        inputs=input_model
    )
    
    # Register model step that will be conditionally executed
    step_register = RegisterModel(
        name="CustomMLOpsRegisterModel",
        estimator=estimator,
#         model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        model_data=model_output,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics
    )

    # Condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThanOrEqualTo(  # You can change the condition here
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.accuracy.value",  # This should follow the structure of your report_dict defined in the evaluate.py file.
        ),
        right=0.4,  # You can change the threshold here
    )
    step_cond = ConditionStep(
        name="CustomMLOpsAccuracyCond",
        conditions=[cond_lte],
        if_steps=[step_model, step_register],
        else_steps=[],
    )

    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            training_instance_count,
            model_approval_status,
            code_repo_name,
            src_dir,
        ],
        steps=[step_process, step_tuning, step_eval, step_cond],
#         steps=[step_process],
        sagemaker_session=sagemaker_session,
    )
    return pipeline