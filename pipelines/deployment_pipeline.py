import numpy as np
import pandas as pd
import json
# from materializer.custom_materializer import cs_materializer
from .utils import get_data_for_test
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from materializer.numpy_materializer import NumpyMaterializer

from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.clean_data import clean_df
from steps.evaluate_model import evaluate_model
from steps.ingest_data import ingest_data
from steps.model_train import train_model 

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@step(enable_cache=False)
def dynamic_importer() -> np.ndarray:
    """Downloads the latest data from a mock API."""
    data = get_data_for_test()
    print(type(data))
    return data


class DeploymentTriggerConfig(BaseParameters):
    min_accuracy : float = 0

@step
def deployment_trigger(
    accuracy:float,
    config: DeploymentTriggerConfig
):
    return accuracy >= config.min_accuracy


class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    # get the MLflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with same pipeline name, step name and model name
    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_services[0])
    print(type(existing_services))
    return existing_services[0]

@step()
def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    prediction = service.predict(data)
    return prediction



@pipeline(enable_cache=False,settings={"docker":docker_settings})
def continuous_deployment_pipeline(
    data_path:str,
    min_accuracy: float = 0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_data(data_path=data_path)
    X_train,X_test,y_train,y_test,label_encoder,vocab_size,max_length = clean_df(df)
    model = train_model(X_train, y_train, X_test, y_test,label_encoder,vocab_size,max_length)
    acc = evaluate_model(model,X_test,y_test)
    print(model)
    deployment_decision = deployment_trigger(accuracy = acc)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers = workers,
        timeout = timeout
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)