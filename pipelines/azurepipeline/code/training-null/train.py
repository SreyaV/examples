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





def get_ws():
   # argparse stuff for model path and model name
"""   parser = argparse.ArgumentParser(description='sanity check on model')
  parser.add_argument('-b', '--base_path', help='directory to base folder', default='../../data')
  parser.add_argument('-m', '--model', help='path to model file', default='/model/latest.h5')
  parser.add_argument('-n', '--model_name', help='AML Model name', default='tacosandburritos')
  parser.add_argument('-t', '--tenant_id', help='tenant_id')
  parser.add_argument('-s', '--service_principal_id', help='service_principal_id')
  parser.add_argument('-p', '--service_principal_password', help='service_principal_password')
  parser.add_argument('-u', '--subscription_id', help='subscription_id')
  parser.add_argument('-r', '--resource_group', help='resource_group')
  parser.add_argument('-w', '--workspace', help='workspace')
  args = parser.parse_args() """

"""   auth_args = {
    'tenant_id': args.tenant_id,
    'service_principal_id': args.service_principal_id,
    'service_principal_password': args.service_principal_password
  }

  ws_args = {
    'auth': ServicePrincipalAuthentication(**auth_args),
    'subscription_id': args.subscription_id,
    'resource_group': args.resource_group
  } """

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

  ws = Workspace.get(workspace, **ws_args)
  return ws

'''
  def run(mdl_path, model_name, ws, tgs):
  print(ws.get_details())

  print('\nSaving model {} to {}'.format(mdl_path, model_name))

  # Model Path needs to be relative
  mdl_path = relpath(mdl_path, '.')

  Model.register(ws, model_name=model_name, model_path=mdl_path, tags=tgs)
  print('Done!')
'''





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

  ws=get_ws()
  exp = Experiment(ws, 'experiment with kf')
  with exp.start_logging() as run:
    #create outputs directory (might not have to do)
    #save file to outputs/whatever
    #use that as path to register
    
    model = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Dense(64, activation='relu'))
    # Add another:
    model.add(layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    model.add(layers.Dense(10, activation='softmax'))

    layers.Dense(64, activation='sigmoid')

    model = tf.keras.Sequential([
    # Adds a densely-connected layer with 64 units to the model:
    layers.Dense(64, activation='relu', input_shape=(32,)),
    # Add another:
    layers.Dense(64, activation='relu'),
    # Add a softmax layer with 10 output units:
    layers.Dense(10, activation='softmax')])

    model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))
    info('Training')
    model.fit(data, labels, epochs=10, batch_size=32)

    # save model
    info('Saving Model')

    # check existence of base model folder
    output = check_dir(output)

    print('Serializing into saved_model format')
    tf.saved_model.save(model, str(output))
    print('Done!')

    # add time prefix folder
    file_output = str(Path(output).joinpath('latest.h5'))
    print('Serializing h5 model to:\n{}'.format(file_output))
    model.save(file_output)

    return generate_hash(file_output, 'kf_pipeline')




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

  model_signature = run(**args)

  args['model_signature'] = model_signature.upper()
  args['model_type'] = 'tfv2-MobileNetV2'
  print('Writing out params...', end='')
  with open(str(params), 'w') as f:
    json.dump(args, f)

  print(' Saved to {}'.format(str(params)))

  # python train.py -o model 
