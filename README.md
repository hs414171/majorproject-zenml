
# Categorize Post Title for Content Moderation

### Problem Statement : 
With the expanding reach of the internet, monitoring the type of content being published online has become increasingly crucial. This project aims to categorize various forms of graphic content, ranging from accidents to ISIS beheadings. The data is sourced from a surface web gore site, watchpeopledie.tv. The goal of this project is /to serve as an intermediary step towards enhancing content moderation on online platforms like Reddit.

The purpose of this repository is to demonstrate how ZenML empowers your business to build and deploy machine learning pipelines in a multitude of ways:

+By offering you a framework and template to base your own work on.

+By integrating with tools like MLflow for deployment, tracking and more

+By allowing you to build and deploy your machine learning pipelines easily

### 🐍 Python Requirements

Let's jump into the Python packages you need. Within the Python environment of your choice, run:

```
git clone https://github.com/hs414171/test_zenml.git
pip install -r requirements_ubuntu.txt
```
ZenML comes bundled with a React-based dashboard. This dashboard allows you to observe your stacks, stack components and pipeline DAGs in a dashboard interface. To access this, you need to launch the ZenML Server and Dashboard locally, but first you must install the optional dependencies for the ZenML server:

```
pip install zenml["server"]
zenml up
```

If you are running the run_deployment.py script, you will also need to install some integrations using ZenML:

```
zenml integration install mlflow -y
```

The project can only be executed with a ZenML stack that has an MLflow experiment tracker and model deployer as a component. Configuring a new stack with the two components are as follows:

```
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set
```

### 👍 The Solution
To initiate our project, we will develop a BiLSTM model designed to predict the category of content titles. This model employs two LSTM networks operating in opposite directions, enhancing the prediction accuracy for categorizing titles.

#### Training Pipeline
Our standard training pipeline consists of several steps:

1) ingest_data: This step will ingest the data and create a DataFrame.
2) clean_data: This step will clean the data and remove the unwanted columns.
3) train_model: This step will train the model and save the model using MLflow autologging.
4) evaluation: This step will evaluate the model and save the metrics -- using MLflow autologging -- into the artifact store.

#### Deployment Pipeline

We have another pipeline, the deployment_pipeline.py, that extends the training pipeline, and implements a continuous deployment workflow. It ingests and processes input data, trains a model and then (re)deploys the prediction server that serves the model if it meets our evaluation criteria. The criteria that we have chosen is a configurable threshold on the accuracy of the training.

The first four steps of the pipeline are the same as above, but we have added the following additional ones:

1)deployment_trigger: The step checks whether the newly trained model meets the criteria set for deployment.

2)model_deployer: This step deploys the model as a service using MLflow (if deployment criteria is met).

In the deployment pipeline, ZenML's MLflow tracking integration is used for logging the hyperparameter values and the trained model itself and the model evaluation metrics -- as MLflow experiment tracking artifacts -- into the local MLflow backend. This pipeline also launches a local MLflow deployment server to serve the latest MLflow model if its accuracy is above a configured threshold.

The MLflow deployment server runs locally as a daemon process that will continue to run in the background after the example execution is complete. When a new pipeline is run which produces a model that passes the accuracy threshold validation, the pipeline automatically updates the currently running MLflow deployment server to serve the new model instead of the old one.

To round it off, we deploy a Streamlit application that consumes the latest model service asynchronously from the pipeline logic. This can be done easily with ZenML within the Streamlit code:

```
model_deployer = MLFlowModelDeployer.get_active_model_deployer()
service = model_deployer.find_model_server(
pipeline_name="continuous_deployment_pipeline",
pipeline_step_name="mlflow_model_deployer_step",
model_name="model",
running = False
)

service[0].start(timeout=10)
result = transform_text(user_input)
pred = service[0].predict(result)  

```
While this ZenML Project trains and deploys a model locally, other ZenML integrations such as the Seldon deployer can also be used in a similar manner to deploy the model in a more production setting (such as on a Kubernetes cluster). We use MLflow here for the convenience of its local deployment.

### 📓 Diving into the code

You can run two pipelines as follows:

Training pipeline:
```
python run_pipeline.py
```
The continuous deployment pipeline:
```
python run_deployment.py
```
### 🕹 Demo Streamlit App
We have developed a demo Streamlit app that allows you to input a title and receive a prediction of its category based on the type of gore content.

If you want to run this Streamlit app in your local system, you can run the following command:-
```
streamlit run streamlit_app.py

```




### Screenshots

![Continuous Deployment Pipeline](https://github.com/hs414171/test_zenml/tree/main/static/Continuous_Deployment.png)
![MLflow Logged Metrics](https://github.com/hs414171/test_zenml/tree/main/static/Logged_Metrics.png)
![Plotted Metrics over Epochs](https://github.com/hs414171/test_zenml/tree/main/static/Model_Metrics.png)
![APP_VA](https://github.com/hs414171/test_zenml/tree/main/static/Vehicular_Accident.png)
![APP_ISIS](https://github.com/hs414171/test_zenml/tree/main/static/Isis_Beheading.png)


### Authors

@hs414171