import pandas as pd

import urllib.request

import numpy as np

import os

import boto3

import sagemaker
from sagemaker.tuner import HyperparameterTuner
from sagemaker import AlgorithmEstimator
from sagemaker.amazon.amazon_estimator import get_image_uri

from datetime import datetime


# Get our data
try:
    urllib.request.urlretrieve ("https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls", "credit_default.xls")
    print('Success: Downloaded credit_default.xls')
except Exception as e:
    print('Data load error: ', e)

try:
    data_xls = pd.read_excel('credit_default.xls', 'Data', index_col=None)
    data_xls.to_csv('data_csv.csv', encoding='utf-8')
    df = pd.read_csv('data_csv.csv', header=1)
    print('Success: Data loaded into dataframe')
except Exception as e:
    print('Data load error: ', e)


# Let's do some cleaning before training the model
# Drop columns containing the String  'ID' and '0'
df = df.drop(['0', 'ID'] , axis='columns')


# Split data into test and train
train_data, val_data, test_data = np.split(df.sample(frac=1, random_state=123), [int(0.8 * len(df)), int(0.9 * len(df))])
print(train_data.shape, test_data.shape)


# Write train, val, and test data to csv and save to directory
try:
    os.mkdir(dir_name)
except:
    print('Directory ' + dir_name + ' already exists')

pd.concat([train_data['default payment next month'], train_data.drop(['default payment next month'], axis=1)],
          axis=1).to_csv(dir_name + '/training_data.csv', index=False, header=False)

pd.concat([val_data['default payment next month'], val_data.drop(['default payment next month'], axis=1)],
          axis=1).to_csv(dir_name + '/validation_data.csv', index=False, header=False)

pd.concat([test_data['default payment next month'], test_data.drop(['default payment next month'], axis=1)],
          axis=1).to_csv(dir_name + '/test_data.csv', index=False, header=False)


# Upload data to S3
bucket = '<YOUR_NAME>-s3-bucket' # <--- PUT YOUR NAME IN LOWER CASE BETWEEN THE <> e.g. adit-s3-bucket
prefix = 'hyperparameter-data'
s3 = boto3.resource('s3')

#Training Data
s3.Bucket(bucket).Object(prefix + '/' + 'training_data.csv').upload_file(dir_name + "/training_data.csv")
s3_training_data = 's3://{}/{}/training_data.csv'.format(bucket, prefix)
print("s3_trianing_data={}".format(s3_training_data))

#Validation Data
s3.Bucket(bucket).Object(prefix + '/' + 'validation_data.csv').upload_file(dir_name + "/validation_data.csv")
s3_validation_data = 's3://{}/{}/validation_data.csv'.format(bucket, prefix)
print("s3_validation_data={}".format(s3_validation_data))

#Testing Data
s3.Bucket(bucket).Object(prefix + '/' + 'test_data.csv').upload_file(dir_name + "/test_data.csv")
s3_testing_data = 's3://{}/{}/test_data.csv'.format(bucket, prefix)
print("s3_testing_data={}".format(s3_testing_data))


# Creating our Estimator
container = get_image_uri(boto3.Session().region_name, 'linear-learner')

sess = sagemaker.Session()

output_location = 's3://{}/{}/output-training-linearlearner'.format(bucket, prefix)
print('training artifacts will be uploaded to: {}'.format(output_location))

role = sagemaker.session.get_execution_role(sagemaker_session=None)
print("role={}".format(role))

linear = sagemaker.estimator.Estimator(container,
                                       role,
                                       train_instance_count=1,
                                       train_instance_type='ml.c4.xlarge',
                                       output_path=output_location,
                                       train_use_spot_instances=True,
                                       train_max_run=1500,
                                       train_max_wait=2000,
                                       sagemaker_session=sess,
                                       tags=[{'Key': 'dataset', 'Value': 'uci_default_of_credit_card_client'},
                                             {'Key': 'algorithm', 'Value': 'linearlearner'}
                                             ])


# Set Hyperparameters
linear.set_hyperparameters(feature_dim=23,
                           predictor_type='binary_classifier',  # type of classification problem
                           learning_rate='auto'

                           # These are the hyperparameters that we're going to tune for:
                           # l1=,
                           # wd=,
                           # use_bias=,
                           # mini_batch_size=
                           )

param_l1 = sagemaker.parameter.ContinuousParameter(0.0005,
                                                   0.01,
                                                   scaling_type='Logarithmic')

param_wd = sagemaker.parameter.ContinuousParameter(0.0005,
                                                   0.01,
                                                   scaling_type='Logarithmic')

param_use_bias = sagemaker.parameter.CategoricalParameter(['True', 'False'])

param_mini_batch_size = sagemaker.parameter.IntegerParameter(100,
                                                             600,
                                                             scaling_type='Linear')

hypertuner = sagemaker.tuner.HyperparameterTuner(linear,
                                                 objective_metric_name = 'test:binary_classification_accuracy',
                                                 hyperparameter_ranges = {'l1' : param_l1,
                                                                          'wd' : param_wd,
                                                                          'use_bias' : param_use_bias,
                                                                          'mini_batch_size': param_mini_batch_size},
                                                 metric_definitions=None,
                                                 strategy='Bayesian',
                                                 objective_type='Maximize',
                                                 max_jobs=20, max_parallel_jobs=3,
                                                 tags=[{'Key': 'dataset',
                                                        'Value': 'uci_default_of_credit_card_client'},
                                                       {'Key': 'algorithm',
                                                        'Value': 'linearlearner'}],
                                                 base_tuning_job_name="NoneNone",
                                                 early_stopping_type='Off')

