import argparse
import os
import numpy as np
import joblib
import pandas as pd
from azureml.core.run import Run
from azureml.core import Workspace, Dataset

#Import data and understanding it.

import collections
import logging
import os
import pathlib
import re
import string
import sys

import pandas as pd
import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

# Data cleaning


def Data_clean(data):
  #group the data by languages
  trn=data.copy()

  #replace the character categorical data into integer form
  trn=trn.replace(['x','o','b','positive','negative'],[10,20,5,1,0])

  return trn


def main():
  # Add arguements
  parser= argparse.ArgumentParser()
  parser.add_argument('--path', type=str, dest='path', default='train.csv', help='mounting data folder')
  parser.add_argument('--lrate',type=float,default=0.5,help='Learning rate')
  parser.add_argument('--nest',type=int,default=100,help='n_estimators')
  parser.add_argument('--depth',type=int,default=1,help='Maximum depth')

  args=parser.parse_args()

  run.log('learning rate:',np.float(args.lrate))
  run.log('n_estimators:',np.int(args.nest))
  run.log('max_depth:',np.int(args.depth))

  # Load dataset
  path=args.path
  df=pd.read_csv(path)

  x=Data_clean(df)

  xtrn,xtst=train_test_split(x, test_size=0.30, random_state=26)

  clf = GradientBoostingClassifier(n_estimators=args.nest, learning_rate=args.lrate,max_depth=args.depth,
  random_state=26).fit(xtrn[xtrn.columns[:-1]],xtrn[xtrn.columns[-1]])


  acc=clf.score(xtst[xtst.columns[:-1]],xtst[xtst.columns[-1]])

  os.makedirs('outputs', exist_ok=True)
  joblib.dump(clf, 'outputs/model.joblib')

  run.log('Accuracy',acc)



run = Run.get_context()
if __name__ == '__main__':
  main()