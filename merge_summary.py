#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 5/9/2025 10:06
# @Author :
import time
from joblib import Parallel, delayed
import random
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import os
os.environ["OMP_NUM_THREADS"] = '7'

import json
import re
from scipy import interpolate
import datetime
import subprocess
from scipy.misc import derivative
from scipy.signal import find_peaks,savgol_filter,argrelextrema
import rasterio
import rioxarray
from sklearn.svm import SVR
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.stattools import pacf,pacf_ols,pacf_yw
from scipy.linalg import toeplitz
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from statsmodels.formula.api import glm
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import geopandas as gpd
from shapely.geometry import Polygon
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
# 自己实现用ols求解pacf的函数
from numpy.dual import lstsq
from statsmodels.tools import add_constant
from statsmodels.tsa.tsatools import lagmat
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from scipy.cluster import hierarchy
import textwrap
import shap
import seaborn as sns
from scipy import signal
from rasterio.transform import from_origin
from sklearn.metrics import make_scorer, r2_score
from scipy.stats import pearsonr
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn_quantile import (
    RandomForestQuantileRegressor,
    SampleRandomForestQuantileRegressor,
)
from rasterio.merge import merge
from rasterio.transform import xy
from shapely.geometry import Point, box
from collections import Counter
from shapely.geometry import mapping
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from scipy.stats import linregress
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import TheilSenRegressor
from sklearn.utils import resample
from affine import Affine
from scipy.ndimage import distance_transform_edt

if __name__ == '__main__':
    other_scenario = 'low_elevation_everygreen'
    country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
                    os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    data_final = pd.DataFrame()
    for i,country_tif in tqdm(enumerate(country_tifs)):
        country_name = os.path.split(country_tif)[-1]
        if os.path.exists(os.path.join(country_tif,f'Wu Yong/{other_scenario}/inform_sum_sa_EVI_SIF_all_biogeo.csv')):
            data_path = os.path.join(country_tif,f'Wu Yong/{other_scenario}/inform_sum_sa_EVI_SIF_all_biogeo.csv')
            data_country = pd.read_csv(data_path)
            data_country['country'] = country_name
            data_country['country_chunk'] = 'all'
            data_final = pd.concat([data_final,data_country])
        else:
            reference_tif_paths = glob.glob(os.path.join(country_tif, f"Wu Yong/{other_scenario}/valid_data_mask*.tif"))
            reference_tif_paths = [item for item in reference_tif_paths if 'number' not in item and 'ratio' not in item]
            if len(reference_tif_paths) == 0:continue
            for path in reference_tif_paths:
                chunk_i,chunk_j = re.search(r"valid_data_mask_(\d+)_(\d+)", path).groups()
                data_path = os.path.join(country_tif, f'Wu Yong/{other_scenario}/inform_sum_sa_EVI_SIF_{chunk_i}_{chunk_j}_biogeo.csv')
                data_country = pd.read_csv(data_path)
                data_country['country'] = country_name
                data_country['country_chunk'] = f'{chunk_i}_{chunk_j}'
                data_final = pd.concat([data_final, data_country])
    data_final.to_csv(f'afterfirst_revision_summary_{other_scenario}.csv')
