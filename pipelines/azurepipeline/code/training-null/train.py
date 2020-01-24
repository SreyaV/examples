from __future__ import absolute_import, division, print_function, unicode_literals
import os
import math
import hmac
import json
import hashlib
import argparse
from random import shuffle
from pathlib2 import Path
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from random import randint
import azureml
from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core import Experiment
from azureml.core.authentication import ServicePrincipalAuthentication
from sklearn.linear_model import LinearRegression
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import mlflow
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import mlflow.azureml
import mlflow.sklearn
import os  


def get_ws():
  auth_args = {
    'tenant_id': '72f988bf-86f1-41af-91ab-2d7cd011db47',
    'service_principal_id': 'bc6175f0-8591-4491-9254-7ff163901a21',
    'service_principal_password': '?mGP@E1hGhU@aNty3=G3e53F:L.gMOVf'
  }

  ws_args = {
    'auth': ServicePrincipalAuthentication(**auth_args),
    'subscription_id': 'ad203158-bc5d-4e72-b764-2607833a71dc',
    'resource_group': 'akannava'
  }

  ws = Workspace.get('akannava', **ws_args)
  return ws





#-----------------------------------------------------------------------------------------------------------------

def check_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)
  return Path(path).resolve(strict=False)

def info(msg, char="#", width=75):
  print("")
  print(char * width)
  print(char + "   %0*s" % ((-1 * width) + 5, msg) + char)
  print(char * width)

def generate_hash(dfile, key):
  print('Generating hash for {}'.format(dfile))
  m = hmac.new(str.encode(key), digestmod=hashlib.sha256)
  BUF_SIZE = 65536
  with open(str(dfile), 'rb') as myfile:
    while True:
      data = myfile.read(BUF_SIZE)
      if not data:
        break
      m.update(data)

  return m.hexdigest()


def run(output='model'):
  print("SDK version:", azureml.core.VERSION)
  ws=get_ws()
  mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
  mlflow.set_experiment('experiment-with-kf')
  #exp = Experiment(ws, 'experiment with kf')

  X, y = load_diabetes(return_X_y = True)
  columns = ['age', 'gender', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
  data = {
    "train":{"X": X_train, "y": y_train},        
    "test":{"X": X_test, "y": y_test}
  }

  print ("Data contains", len(data['train']['X']), "training samples and",len(data['test']['X']), "test samples")

  model_save_path = 'model'

  with mlflow.start_run() as run:
    # Log the algorithm parameter alpha to the run
    mlflow.log_metric('alpha', 0.03)
    # Create, fit, and test the scikit-learn Ridge regression model
    regression_model = Ridge(alpha=0.03)
    regression_model.fit(data['train']['X'], data['train']['y'])
    preds = regression_model.predict(data['test']['X'])

    # Log mean squared error
    print('Mean Squared Error is', mean_squared_error(data['test']['y'], preds))
    mlflow.log_metric('mse', mean_squared_error(data['test']['y'], preds))
    
    # Save the model to the outputs directory for capture
    mlflow.sklearn.log_model(regression_model,model_save_path)
    
    # Plot actuals vs predictions and save the plot within the run
    fig = plt.figure(1)
    idx = np.argsort(data['test']['y'])
    plt.plot(data['test']['y'][idx],preds[idx])
    fig.savefig("actuals_vs_predictions.png")
    mlflow.log_artifact("actuals_vs_predictions.png")





if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='transfer learning for binary image task')
  parser.add_argument('-o', '--outputs', help='output directory', default='model')
  args = parser.parse_args()

  info('Using TensorFlow v.{}'.format(tf.__version__))

  target_path = Path('/mnt/azure').resolve(strict=False).joinpath(args.outputs)
  params = Path('/mnt/azure').joinpath('params.json')

  args = {
    "output": str(target_path),
  }

  # printing out args for posterity
  for i in args:
    print('{} => {}'.format(i, args[i]))

  run(**args)

  print('Made it to end of training')
  print(os.environ)
  # python train.py -o model 
