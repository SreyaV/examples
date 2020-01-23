from __future__ import absolute_import, division, print_function
import os
import math
import hmac
import json
import hashlib
import argparse
from random import shuffle
from pathlib2 import Path
import numpy as np
import sys
import sklearn
from sklearn.linear_model import LinearRegression


def info(msg, char="#", width=75):
  print("")
  print(char * width)
  print(char + "   %0*s" % ((-1 * width) + 5, msg) + char)
  print(char * width)


def check_dir(path):
  if not os.path.exists(path):
    os.makedirs(path)
  return Path(path).resolve(strict=False)

def load_dataset(base_path, dset, split=None):
  # normalize splits
  if split is None:
    split = [8, 1, 1]
  splits = np.array(split) / np.sum(np.array(split))

  # find labels - parent folder names
  labels = {}
  for (_, dirs, _) in os.walk(base_path):
    print('found {}'.format(dirs))
    labels = {k: v for (v, k) in enumerate(dirs)}
    print('using {}'.format(labels))
    break

  # load all files along with idx label
  print('loading dataset from {}'.format(dset))
  with open(dset, 'r') as d:
    data = [(str(Path(line.strip()).absolute()),
             labels[Path(line.strip()).parent.name]) for line in d.readlines()]

  print('dataset size: {}\nsuffling data...'.format(len(data)))

  # shuffle data
  shuffle(data)

  print('splitting data...')
  # split data
  train_idx = int(len(data) * splits[0])

  return data[:train_idx]

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


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='transfer learning for binary image task')
  parser.add_argument('-s', '--base_path', help='directory to base data', default='../../data')
  parser.add_argument('-d', '--data', help='directory to training and test data', default='trainInput.txt')
  parser.add_argument('-y', '--data2', help='target file to hold output data', default='trainOutput.txt')
  parser.add_argument('-e', '--epochs', help='number of epochs', default=10, type=int)
  parser.add_argument('-b', '--batch', help='batch size', default=32, type=int)
  parser.add_argument('-i', '--image_size', help='image size', default=160, type=int)
  parser.add_argument('-l', '--lr', help='learning rate', default=0.0001, type=float)
  parser.add_argument('-o', '--outputs', help='output directory', default='model')
  parser.add_argument('-f', '--dataset', help='cleaned data listing')
  args = parser.parse_args()

  print("original path " + str(Path(args.base_path)) + "\n")
  data_path = Path(args.base_path).joinpath(args.data).resolve(strict=False)
  data_output_path = Path(args.base_path).joinpath(args.data2).resolve(strict=False)
  target_path = Path(args.base_path).resolve(strict=False).joinpath(args.outputs)
  
  print("input data path " + str(data_path) + "\n")
  print("output data path " + str(data_output_path) + "\n")
  print("target path " + str(target_path) + "\n")
  # dataset = Path(args.base_path).joinpath(args.dataset)
  # print("full data path ", str(dataset) + "\n")
  # image_size = args.image_size


  # params = Path(args.base_path).joinpath('params.json')

  # args = {
  #   "dpath": str(data_path),
  #   "img_size": image_size,
  #   "epochs": args.epochs,
  #   "batch_size": args.batch,
  #   "learning_rate": args.lr,
  #   "output": str(target_path),
  #   "dset": str(dataset)
  # }

  TRAIN_INPUT = np.loadtxt(str(data_path))
  print(TRAIN_INPUT)
  print('\n' + str(np.shape(TRAIN_INPUT)))

  TRAIN_OUTPUT = np.loadtxt(str(data_output_path))
  print(TRAIN_OUTPUT)
  print('\n' + str(np.shape(TRAIN_OUTPUT)))

  # TRAIN_SET_COUNT = 100
  # TRAIN_INPUT= list()
  # TRAIN_OUTPUT = list()

  # with open(str(data_path), 'r') as filehandle:
  #   i = 0
  #   for line in filehandle:
  #     currentPlace = line[:-2]
  #     currentPlace = currentPlace[1:]
  #     if i < TRAIN_SET_COUNT:
  #       currentPlace = list(currentPlace.split(','))
  #       TRAIN_INPUT.append(currentPlace)
  #     else:
  #       # currentPlace = list(currentPlace)
  #       TRAIN_OUTPUT.append(currentPlace)
  #     i += 1

  predictor = LinearRegression(n_jobs=-1)
  predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)
  coefficients = predictor.coef_

  # dataset_signature = generate_hash(dataset, 'kf_pipeline')
  # # printing out args for posterity
  # for i in args:
  #   print('{} => {}'.format(i, args[i]))

  # model_signature = run(**args)

  # args['dataset_signature'] = dataset_signature.upper()
  # args['model_signature'] = model_signature.upper()
  # args['model_type'] = 'tfv2-MobileNetV2'
  # print('Writing out params...', end='')
  # with open(str(params), 'w') as f:
  #   json.dump(args, f)

  print(' Saved to {}'.format(str(predictor)))

  # python train.py -d train -e 3 -b 32 -l 0.0001 -o model -f train.txt
