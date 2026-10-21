FROM python:3.11

WORKDIR /app

COPY training/train.py .
COPY feature_repo/fraud_transactions.parquet .

RUN pip install pandas pyarrow scikit-learn joblib

CMD ["python","train.py"]