import os
import shutil
import zipfile
import argparse
from pathlib2 import Path
import wget
from random import randint
import numpy as np


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='data cleaning for binary image task')
  parser.add_argument('-b', '--base_path', help='directory to base data', default='../../data')
  parser.add_argument('-d', '--data', help='directory to training data', default='train')
  parser.add_argument('-t', '--target', help='target file to hold good data', default='trainInput.txt')
  parser.add_argument('-o', '--target2', help='target file to hold output data', default='trainOutput.txt')
  parser.add_argument('-i', '--img_size', help='target image size to verify', default=160, type=int)
  parser.add_argument('-z', '--zipfile', help='source data zip file', default='../../tacodata.zip')
  parser.add_argument('-f', '--force',
                      help='force clear all data', default=False, action='store_true')
  args = parser.parse_args()
  print(args)

  base_path = Path(args.base_path).resolve(strict=False)
  print('Base Path:  {}'.format(base_path))
  data_path = base_path.joinpath(args.data).resolve(strict=False)
  print('Train Path: {}'.format(data_path))
  target_path = Path(base_path).resolve(strict=False).joinpath(args.target)
  print('Train Input File: {}'.format(target_path))
  train_output_path = Path(base_path).resolve(strict=False).joinpath(args.target2)
  print('Train Output File: {}'.format(train_output_path))
  zip_path = args.zipfile

  TRAIN_SET_LIMIT = 1000
  TRAIN_SET_COUNT = 100

  TRAIN_INPUT = list()
  TRAIN_OUTPUT = list()
  for i in range(TRAIN_SET_COUNT):
      a = randint(0, TRAIN_SET_LIMIT)
      b = randint(0, TRAIN_SET_LIMIT)
      c = randint(0, TRAIN_SET_LIMIT)
      op = a + (2*b) + (3*c)
      TRAIN_INPUT.append([a, b, c])
      TRAIN_OUTPUT.append(op)

  # save file
  print('writing dataset to {}'.format(target_path))
  NP_TRAIN_INPUT = np.array(TRAIN_INPUT)
  NP_TRAIN_OUTPUT = np.array([TRAIN_OUTPUT])
  np.savetxt(str(target_path), TRAIN_INPUT)
  data = np.loadtxt(str(target_path))
  print('\n')
  print(data)
  print('\n' + str(np.shape(data)))
  np.savetxt(str(train_output_path), TRAIN_OUTPUT)
  data2 = np.loadtxt(str(train_output_path))
  print('\n')
  print(data2)
  print('\n' + str(np.shape(data2)))
  # with open(str(target_path), 'w+') as f:
    # for item in TRAIN_INPUT:
    #   f.write("%s\n" % item)
    # for item in TRAIN_OUTPUT:
    #   f.write("%s\n" % item)

  # python data.py -z https://aiadvocate.blob.core.windows.net/public/tacodata.zip -t train.txt
