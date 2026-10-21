import os

from sagemaker import get_execution_role
from sagemaker.session import Session
from sagemaker.sklearn.estimator import SKLearn


def main():
    bucket = os.environ.get("S3_BUCKET")
    region = os.environ.get("AWS_REGION", "us-east-1")
    role = os.environ.get("SAGEMAKER_ROLE_ARN") or get_execution_role()

    if not bucket:
        raise ValueError("Set S3_BUCKET before running this script")

    sagemaker_session = Session(default_bucket=bucket)
    train_input = f"s3://{bucket}/fraud-project/data/fraud_data.csv"
    output_path = f"s3://{bucket}/fraud-project/models/"

    estimator = SKLearn(
        entry_point="train_fraud_model.py",
        source_dir="aws/sagemaker",
        framework_version="1.0-1",
        instance_type="ml.m5.large",
        instance_count=1,
        role=role,
        sagemaker_session=sagemaker_session,
        output_path=output_path,
        hyperparameters={"target-column": "is_fraud"},
    )

    estimator.fit({"train": train_input})
    print(f"Training job complete. Model artifact available at: {output_path}")


if __name__ == "__main__":
    main()
