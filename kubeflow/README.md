# Kubeflow Pipeline for Fraud Detection

This folder contains a minimal Kubeflow Pipelines (KFP) setup for the fraud detection workflow.

## Included pipeline steps

1. Load raw fraud data
2. Validate the dataset
3. Preprocess and engineer features
4. Train the Random Forest model
5. Evaluate the trained model

## Compile the pipeline

From the project root:

```powershell
cd "Fraud Detection"
.\fraud\Scripts\python.exe -m pip install kfp
.\fraud\Scripts\python.exe kubeflow\pipeline.py
```

This will generate:

```text
Fraud Detection/kubeflow/fraud_detection_pipeline.yaml
```

## Run on Kubeflow

After connecting to a Kubeflow Pipelines cluster or a Kubeflow deployment endpoint, upload the generated YAML file in the Kubeflow UI and start the run.

For a cluster-based run, you can also use a client command like:

```python
from kfp import Client
client = Client(host="http://<kubeflow-pipeline-host>:<port>")
client.create_run_from_pipeline_package("kubeflow/fraud_detection_pipeline.yaml", arguments={})
```

## Notes

This is a local KFP pipeline definition. A real Kubernetes/Kubeflow cluster is required to execute the jobs in containers.
