import sagemaker
import boto3
from sagemaker.predictor import csv_serializer    # Converts strings for HTTP POST requests on inference

import numpy as np                                # For performing matrix operations and numerical processing
import pandas as pd                               # For manipulating tabular data
from time import gmtime, strftime
import os

region = boto3.Session().region_name
smclient = boto3.Session().client('sagemaker')

from sagemaker import get_execution_role

role = get_execution_role()
print(role)