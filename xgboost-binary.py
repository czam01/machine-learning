#!/usr/bin/env python
# coding: utf-8

bucket = 'YOUR_BUCKET' # Bucket name, not the ARN or the s3:// path
prefix = 'sagemaker/xgboost_credit_risk'

import boto3
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sagemaker
import csv
from sagemaker import get_execution_role
from sagemaker.predictor import csv_serializer

# Execution Sage Maker role
role = get_execution_role() 

# Download data in xls format
get_ipython().system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls')

# Load the sample data and show a part of it
dataset = pd.read_excel('default of credit card clients.xls')
pd.set_option('display.max_rows', 8)
pd.set_option('display.max_columns', 15)
dataset

# Split the data in validation and train files in csv
train_data, validation_data, test_data = np.split(dataset.sample(frac=1, random_state=1729), [int(0.7 * len(dataset)), int(0.9 * len(dataset))])
train_data.to_csv('traind.csv', header=True, index=False)
validation_data.to_csv('validationd.csv', header=True, index=False)

# Xgboost on SageMaker requires that the target value (Y) would be in first column
with open('traind.csv', 'r') as infile, open('train_clean.csv', 'a') as outfile:
    # output dict needs a list for new column ordering
    fieldnames = ['Y', 'Unnamed: 0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23' ]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    # reorder the header first
    writer.writeheader()
    for row in csv.DictReader(infile):
        # writes the reordered rows to the new file
        writer.writerow(row)

with open('validationd.csv', 'r') as infile, open('validation_clean.csv', 'a') as outfile:
    # output dict needs a list for new column ordering
    fieldnames = ['Y', 'Unnamed: 0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23' ]
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    # reorder the header first
    writer.writeheader()
    for row in csv.DictReader(infile):
        # writes the reordered rows to the new file
        writer.writerow(row)

# Remove index and header from csv file        
df = pd.read_csv("validation_clean.csv")
df.to_csv('validation_clean_finally.csv', header=False, index=False)
df = pd.read_csv("train_clean.csv")
df.to_csv('train_clean_finally.csv', header=False, index=False)


# Load containers with Xgboost algorithm
containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'}

sess = sagemaker.Session()

# Define instance to make the trainning process, it must have available memory as the lenght of the dataset
xgb = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
                                    role, 
                                    train_instance_count=1, 
                                    train_instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket, prefix),
                                    sagemaker_session=sess)

# Hyperparameters for the Xgboost
xgb.set_hyperparameters(eta=0.1,
                        objective='binary:logistic',
                        num_round=25)


s3_train_prefix = os.path.join(prefix, 'train/train_clean_finally.csv')
s3_validation_prefix = os.path.join(prefix, 'validation/validation_clean_finally.csv')

s3_train = 's3://{}/{}'.format(bucket, s3_train_prefix)
s3_validation = 's3://{}/{}'.format(bucket, s3_validation_prefix)


# upload to S3
boto3.Session().resource('s3').Bucket(bucket).Object(s3_train_prefix).upload_file('train_clean_finally.csv')
boto3.Session().resource('s3').Bucket(bucket).Object(s3_validation_prefix).upload_file('validation_clean_finally.csv')


# configure sagemaker InputDataConfiguration since by default it takes libsvm, and now we are using csv
s3_input_train = sagemaker.session.s3_input(s3_train, content_type='text/csv')
s3_input_validation = sagemaker.session.s3_input(s3_validation, content_type='text/csv')

# Train model
xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})

xgb_predictor = xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer
xgb_predictor.deserializer = None

def predict(data, rows=500):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        predictions = ','.join([predictions, xgb_predictor.predict(array).decode('utf-8')])

    return np.fromstring(predictions[1:], sep=',')

predictions = predict(test_data.as_matrix()[:, 1:])
predictions
