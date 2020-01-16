import os
import shutil
import zipfile
import argparse
from pathlib2 import Path
import wget
from random import randint


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='data cleaning for binary image task')
  parser.add_argument('-b', '--base_path', help='directory to base data', default='../../data')
  parser.add_argument('-d', '--data', help='directory to training data', default='train')
  parser.add_argument('-t', '--target', help='target file to hold good data', default='train.txt')
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
  print('Train File: {}'.format(target_path))
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
  with open(str(target_path), 'w+') as f:
    f.write(TRAIN_OUTPUT)

  # python data.py -z https://aiadvocate.blob.core.windows.net/public/tacodata.zip -t train.txt
