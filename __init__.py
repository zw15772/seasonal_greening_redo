# coding=utf-8
import sys
version = sys.version_info.major
assert version == 3, 'Python Version Error'

from lytools import *
from matplotlib import pyplot as plt
import numpy as np
from scipy import interpolate
from scipy import signal
import time
import os
from osgeo import gdal
from tqdm import tqdm
import datetime
from scipy import stats, linalg
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import FontProperties
import copyreg
from scipy.stats import gaussian_kde as kde
import matplotlib as mpl
import multiprocessing
from multiprocessing.pool import ThreadPool as TPool
import types
from scipy.stats import gamma as gam
import math
import copy
import scipy
import sklearn
import random
import h5py
from netCDF4 import Dataset
import shutil
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from operator import itemgetter
from itertools import groupby
# import RegscorePy
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import f_oneway
from mpl_toolkits.mplot3d import Axes3D
import pickle
from dateutil import relativedelta
from sklearn.inspection import permutation_importance
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import glob
from osgeo import osr
from osgeo import ogr
from matplotlib.colors import LogNorm
# try:
#     from jenkspy import jenks_breaks
# except:
#     from jenkspy import JenksNaturalBreaks
import scipy.io
from scipy.stats import kruskal
import psutil
import xgboost as xgb
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from pyhdf.SD import SD, SDC
# import xarray as xr
# import pingouin
from matplotlib.colors import LinearSegmentedColormap
from climate_indices import compute
from climate_indices import indices
import rpy2

this_root = '/Volumes/NVME2T/greening_project_redo/'
# this_root = '/Users/wenzhang/Desktop/'
data_root = this_root + 'data/'
results_root = this_root + 'results/'
temporary_root = this_root + 'temporary/'
# results_root_main_flow = this_root + 'main_flow_results/'
# from LY_Tools import *

'''
pip install xgboost
No such file or directory: 'cmake'
brew install cmake
'''
# exit()
np.seterr('ignore')

def sleep(t=1):
    time.sleep(t)
def pause():
    # ANSI colors: https://gist.github.com/rene-d/9e584a7dd2935d0f461904b9f2950007
    input('\33[7m'+"PRESS ENTER TO CONTINUE."+'\33[0m')

def kill_python_process():
    # print('will kill all python3.9 process, \033[7m\033[31mdouble\33[0m press ENTER to continue')
    # pause()
    # print('One more ENTER...')
    # pause()
    for p in psutil.process_iter():
        name = p.name()
        if 'python3.9' in name:
            p.kill()

T = Tools()
T.mk_dir(temporary_root)
if __name__ == '__main__':
    kill_python_process()
