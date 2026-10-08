import kfp
import argparse
import typing
from typing import Dict
from typing import NamedTuple
from kfp import dsl
from kfp.dsl import (Artifact,
                     Dataset,
                     Input,
                     Model,
                     Output,
                     Metrics,
                     ClassificationMetrics,
                     component,
                     OutputPath,
                     InputPath)
import google.cloud.aiplatform as aip
from google_cloud_pipeline_components.types import artifact_types
from kfp import compiler


# Pipeline Component : Data Ingestion
@dsl.component(
    packages_to_install=["pandas", "google-cloud-storage"],
    base_image="python:3.12.7-slim"
)
def download_data(project_id: str, bucket: str, file_name: str, dataset: Output[Dataset]):
    '''download data'''
    from google.cloud import storage
    import pandas as pd
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Downloading the file from a Google bucket
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket)
    blob = bucket.blob(file_name)
    blob.download_to_filename(dataset.path + ".csv")
    logging.info('Downloaded Data!')


# Pipeline Component : Training Decision Tree Classifier
@dsl.component(
    packages_to_install=['pandas', 'scikit-learn==1.3.2'],
    base_image="python:3.12.7-slim"
)
def train_dt(features: Input[Dataset], out_model: Output[Model]) -> NamedTuple('outputs', metrics=dict):
    '''train a DT with default parameters'''
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, recall_score
    import json
    import logging
    import sys
    import os
    import pickle

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    df = pd.read_csv(features.path + ".csv")
    logging.info(df.columns)

    train_X, val_X, train_y, val_y = train_test_split(df.drop('class', axis=1),
                                                      df['class'], test_size=0.30,
                                                      random_state=42)

    model_dt = DecisionTreeClassifier(criterion="gini", random_state=42)
    # Fit the model
    model_dt.fit(train_X, train_y)
    # Evaluate the model
    val_predictions = model_dt.predict(val_X)
    accuracy = accuracy_score(val_y, val_predictions)
    recall = recall_score(val_y, val_predictions)

    metrics_dict = {
        "accuracy": accuracy,
        "recall": recall,
    }
    logging.info(metrics_dict)

    out_model.metadata["file_type"] = ".pkl"
    out_model.metadata["algo"] = "dt"
    # Save the model
    m_file = out_model.path + ".pkl"
    with open(m_file, 'wb') as f:
        pickle.dump(model_dt, f)

    outputs = NamedTuple('outputs', metrics=dict)
    return outputs(metrics_dict)


# Pipeline Component : Training Logistic Regression
@dsl.component(
    packages_to_install=['pandas', 'scikit-learn==1.3.2'],
    base_image="python:3.12.7-slim"
)
def train_lr(features: Input[Dataset], out_model: Output[Model]) -> NamedTuple('outputs', metrics=dict):
    '''train a LogisticRegression with default parameters'''
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, recall_score
    import json
    import logging
    import sys
    import os
    import pickle

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    df = pd.read_csv(features.path + ".csv")

    logging.info(df.columns)

    x_train, x_test, y_train, y_test = train_test_split(df.drop('class', axis=1),
                                                        df['class'], test_size=0.30,
                                                        random_state=101)
    model_lr = LogisticRegression()
    model_lr.fit(x_train, y_train)

    # Evaluate the model
    predictions = model_lr.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    metrics_dict = {
        "accuracy": accuracy,
        "recall": recall,
    }
    logging.info(metrics_dict)

    out_model.metadata["file_type"] = ".pkl"
    out_model.metadata["algo"] = "lr"
    # Save the model
    m_file = out_model.path + ".pkl"
    with open(m_file, 'wb') as f:
        pickle.dump(model_lr, f)

    outputs = NamedTuple('outputs', metrics=dict)
    return outputs(metrics_dict)


# Pipeline Component : Prediction with Scikit
@dsl.component(
    packages_to_install=['pandas', 'scikit-learn==1.3.2'],
    base_image="python:3.12.7-slim"
)
def predict(model: Input[Model], features: Input[Dataset], results: Output[Dataset]):
    import pandas as pd
    import pickle
    import json
    import logging
    import sys
    import os

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    df = pd.read_csv(features.path + ".csv")

    filename = model.path + ".pkl"

    # Loading the saved model
    model_sk = pickle.load(open(filename, 'rb'))

    xNew = df[['ntp', 'pgc', 'dbp', 'tsft', 'si', 'bmi', 'dpf', 'age']]

    dfcp = df.copy()
    y_classes = model_sk.predict(xNew)
    logging.info(y_classes)
    dfcp['pclass'] = y_classes.tolist()
    dfcp.to_csv(results.path + ".csv", index=False, encoding='utf-8-sig')


# Pipeline Component : Algorithm Selection

@dsl.component(
    base_image="python:3.12.7-slim"
)
def compare_model(dt_metrics: dict, lr_metrics: dict) -> str:
    import logging
    import json
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info(dt_metrics)
    logging.info(lr_metrics)
    if dt_metrics.get("accuracy") > lr_metrics.get("accuracy"):
        return "DT"
    else:
        return "LR"


# Upload Model and Metrics to Google Bucket

@dsl.component(
    packages_to_install=["google-cloud-storage"],
    base_image="python:3.12.7-slim"
)
def upload_model_to_gcs(project_id: str, model_repo: str, model: Input[Model]):
    '''upload model to gsc'''
    from google.cloud import storage
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # upload the model to GCS
    client = storage.Client(project=project_id)
    bucket = client.bucket(model_repo)
    mb_path = str(model.metadata["algo"]) + '/' + str(model.metadata["algo"]) + '_model' + str(
        model.metadata["file_type"])
    blob = bucket.blob(mb_path)
    blob.upload_from_filename(model.path + str(model.metadata["file_type"]))
    logging.info("Saved the model to the folder : " + mb_path + "in the bucket" + model_repo)


# Define the workflow of the pipeline.
@kfp.dsl.pipeline(
    name="diabetes-predictor-training-pipeline")
def ml_pipeline(project_id: str, data_bucket: str, trainset_filename: str, model_repo: str, testset_filename: str):
    di_op = download_data(
        project_id=project_id,
        bucket=data_bucket,
        file_name=trainset_filename
    )

    training_dt_job_run_op = train_dt(
        features=di_op.outputs["dataset"]
    )

    training_lr_job_run_op = train_lr(
        features=di_op.outputs["dataset"]
    )

    pre_di_op = download_data(
        project_id=project_id,
        bucket=data_bucket,
        file_name=testset_filename
    ).after(training_dt_job_run_op, training_lr_job_run_op)

    comp_model__op = compare_model(dt_metrics=training_dt_job_run_op.outputs["metrics"],
                                   lr_metrics=training_lr_job_run_op.outputs["metrics"]).after(training_dt_job_run_op,
                                                                                               training_lr_job_run_op)

    # defining the branching condition
    with dsl.If(comp_model__op.output == "DT"):
        predict_dt_job_run_op = predict(
            model=training_dt_job_run_op.outputs["out_model"],
            features=pre_di_op.outputs["dataset"]
        )
        upload_model_dt_to_gc_op = upload_model_to_gcs(
            project_id=project_id,
            model_repo=model_repo,
            model=training_dt_job_run_op.outputs['out_model']
        ).after(predict_dt_job_run_op)

    with dsl.If(comp_model__op.output == "LR"):
        predict_lr_job_run_op = predict(
            model=training_lr_job_run_op.outputs["out_model"],
            features=pre_di_op.outputs["dataset"]
        )
        upload_model_lr_to_gc_op = upload_model_to_gcs(
            project_id=project_id,
            model_repo=model_repo,
            model=training_lr_job_run_op.outputs['out_model']
        ).after(predict_lr_job_run_op)


def compile_pipeline(target_file):
    compiler.Compiler().compile(pipeline_func=ml_pipeline,
                                package_path=target_file)


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_file', type=str, default="training_pipeline.yaml", help="Pipeline Name")
    args = parser.parse_args()
    return vars(args)

if __name__ == '__main__':
    compile_pipeline(**parse_command_line_arguments())
