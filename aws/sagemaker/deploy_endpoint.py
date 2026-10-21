import os

from sagemaker.sklearn.model import SKLearnModel


def main():
    role = os.environ.get("SAGEMAKER_ROLE_ARN")
    if not role:
        raise ValueError("Set SAGEMAKER_ROLE_ARN before deploying the model")

    model_s3_path = os.environ.get("MODEL_S3_PATH")
    if not model_s3_path:
        raise ValueError("Set MODEL_S3_PATH to the S3 model artifact, for example s3://my-bucket/models/model.tar.gz")

    model = SKLearnModel(
        model_data=model_s3_path,
        role=role,
        entry_point="train_fraud_model.py",
        framework_version="1.0-1",
    )

    predictor = model.deploy(
        instance_type="ml.m5.large",
        initial_instance_count=1,
        endpoint_name="fraud-detection-endpoint",
    )
    print(f"Endpoint created: {predictor.endpoint_name}")


if __name__ == "__main__":
    main()
