import argparse
from pathlib import Path

import boto3


def upload_file(bucket_name, file_path, s3_key, region_name):
    s3_client = boto3.client("s3", region_name=region_name)

    # This script is intended to upload to an already existing S3 bucket.
    # It does not create a new bucket.
    s3_client.upload_file(str(file_path), bucket_name, s3_key)
    return f"s3://{bucket_name}/{s3_key}"


def main():
    parser = argparse.ArgumentParser(description="Upload fraud project data to an existing S3 bucket")
    parser.add_argument("--bucket", required=True, help="Existing S3 bucket name")
    parser.add_argument("--file", required=True, help="Local file to upload")
    parser.add_argument("--prefix", default="fraud-project", help="S3 key prefix")
    parser.add_argument("--region", default="us-east-1", help="AWS region of the existing bucket")
    args = parser.parse_args()

    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    s3_key = f"{args.prefix.strip('/')}/{file_path.name}"
    uploaded_path = upload_file(args.bucket, file_path, s3_key, args.region)
    print(f"Uploaded: {uploaded_path}")


if __name__ == "__main__":
    main()
