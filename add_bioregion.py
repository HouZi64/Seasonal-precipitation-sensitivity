#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 5/9/2025 10:06
# @Author : Chen Mingzheng
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
import itertools
from scipy.stats import pearsonr
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
class SPEI_Process():
    def __init__(self):
        pass

    def generate_clipped_SPEI(self,spei_path,shp_path,clipped_data_output_path):
        if not os.path.exists(clipped_data_output_path):
            spei = xr.open_dataset(spei_path)
            gdf = gpd.read_file(shp_path)
            gdf = gdf.to_crs(spei.crs.crs_wkt)
            spei = spei.rio.write_crs(spei.crs.crs_wkt)
            europe_spei = spei['spei'].rio.clip(gdf.geometry.apply(mapping), gdf.crs, drop=True)
            europe_spei.attrs.pop('grid_mapping', None)
            europe_spei.to_netcdf(clipped_data_output_path)
        else:
            europe_spei = xr.open_dataset(clipped_data_output_path)
        return europe_spei
    def long_term_baseline(self,spei_path,shp_path,clipped_data_output_path,target_period):
        SPEI_clipped = self.generate_clipped_SPEI(spei_path,shp_path,clipped_data_output_path)
        subset = SPEI_clipped.sel(
            time=SPEI_clipped['time'].dt.year.isin(range(target_period['start_year'], target_period['end_year']+1))
        ).where(SPEI_clipped['time'].dt.month.isin(target_period['month']), drop=True)

        # 计算时间维度均值
        mean_matrix = subset.mean(dim='time', skipna=True)
        return mean_matrix

    def offset_calculation(self,spei_path,shp_path,clipped_data_output_path,target_period,drought_year):
        SPEI_clipped = self.generate_clipped_SPEI(spei_path,shp_path,clipped_data_output_path)
        baseline_matrix = self.long_term_baseline(spei_path,shp_path,clipped_data_output_path,target_period)

        matrix_tobe_calculated = SPEI_clipped.sel(
            time=SPEI_clipped['time'].dt.year.isin(range(drought_year, drought_year+1))
        ).where(SPEI_clipped['time'].dt.month.isin(target_period['month']), drop=True)
        matrix_tobe_calculated = matrix_tobe_calculated - baseline_matrix
        return matrix_tobe_calculated

    def offset_weighted_mean(self,offset_matrix,weights_path):
        offset_data = offset_matrix['spei']
        offset_data = offset_data.rio.write_crs("EPSG:4326", inplace=True)  # 视具体情况可能要改
        weights = rioxarray.open_rasterio(weights_path, masked=True).squeeze()
        offset_resampled = offset_data.rio.reproject_match(weights)
        weighted = offset_resampled * weights
        # 5. 计算加权平均（只在有效值上计算）
        weighted_mean = (weighted.sum() / weights.where(~offset_resampled.isnull()).sum()).item()
        return weighted_mean

    def offset_visualization(self,offset_matrix,weights_path):
        return (self.offset_weighted_mean(offset_matrix,weights_path))

def build_biome_nearest_lookup(biome_tif_path, invalid_val=15):
    with rasterio.open(biome_tif_path) as src:
        biome = src.read(1)
        transform = src.transform

    # 有效掩码
    valid_mask = biome != invalid_val

    # 距离变换：返回到最近非15点的索引（按像素）
    distance, indices = distance_transform_edt(~valid_mask, return_indices=True)

    # 构造一个查找表：对于每个点 [i, j]，获取最近的有效 biome 值
    nearest_biome = biome[indices[0], indices[1]]

    return biome, nearest_biome, transform


def get_biome_at_point(lon, lat, transform, biome, nearest_biome, invalid_val=15):
    from rasterio.transform import rowcol

    row, col = rowcol(transform, lon, lat)
    h, w = biome.shape

    if 0 <= row < h and 0 <= col < w:
        val = biome[row, col]
        if val == invalid_val:
            return nearest_biome[row, col]
        else:
            return val
    else:
        return -9999

if __name__ == '__main__':
    country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
                    os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    gdd_path = r'D:\Data Collection\Temperature'
    SPEI_paths = [
        r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei03.nc',
        r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei06.nc']
    tem_path = r'D:\Data Collection\Temperature\T2M\T2M_2000to2023.nc'
    VPD_path = r'D:\Data Collection\Temperature\VPD/VPD_2000to2023.nc'
    TP_path = r'D:\Data Collection\other_factors/total_precipitation.nc'
    TP_path_2224 = r'D:\Data Collection\other_factors/total_precipitation_22_24.nc'
    SIF_path = r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\2 forest sensitivity to water\data collection/SIF_tempory.nc'
    chillingdays_data_path = r'D:\Data Collection\Temperature\chillingdays'
    chillingdays_paths = glob.glob(os.path.join(chillingdays_data_path, '*.nc'))
    drought_years = [2003, 2015, 2018, 2019, 2022]
    norway_index = None
    geobioregion_path = r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\2 forest sensitivity to water\Biogeo_region/Biogeoregion_Europe.tif'

    other_scenario = 'low_elevation_everygreen'
    for i, country_tif in tqdm(enumerate(country_tifs)):

        if ('ukraine' in country_tif or 'belarus' in country_tif
                or 'andorra' in country_tif or 'liechtenstein' in country_tif or 'luxembourg' in country_tif): continue
        reference_tif_path = os.path.join(country_tif,f'Wu Yong/{other_scenario}/valid_data_mask_all.tif')
        if os.path.exists(reference_tif_path):
            information_path = os.path.join(country_tif,f'Wu Yong/{other_scenario}/inform_sum_sa_EVI_SIF_all.csv')
            with rasterio.open(reference_tif_path) as ref_src, rasterio.open(geobioregion_path) as bio_src:
                # 读取csv
                df = pd.read_csv(information_path)

                biogeo_values = []
                for _, row in df.iterrows():
                    r, c = int(row["row"]), int(row["col"])

                    # 根据行列号获取空间坐标 (x, y)
                    x, y = ref_src.xy(r, c)

                    # 转换为 biogeoregion 栅格的行列号
                    try:
                        br_row, br_col = bio_src.index(x, y)
                        bio_value = bio_src.read(1)[br_row, br_col]
                        if bio_value == 15:
                            biome, nearest_biome, transform = build_biome_nearest_lookup(geobioregion_path)
                            biome_val = get_biome_at_point(x, y, transform, biome, nearest_biome)
                            bio_value = biome_val
                    except IndexError:
                        bio_value = None  # 超出范围

                    biogeo_values.append(bio_value)
                df["biogeo"] = biogeo_values
                df.to_csv(information_path.replace('.csv','_biogeo.csv'), index=False)
        else:
            reference_tif_paths = glob.glob(os.path.join(country_tif, f"Wu Yong/{other_scenario}/valid_data_mask*.tif"))
            reference_tif_paths = [item for item in reference_tif_paths if 'number' not in item and 'ratio' not in item]
            if len(reference_tif_paths) == 0:continue
            for path in reference_tif_paths:
                chunk_i,chunk_j = re.search(r"valid_data_mask_(\d+)_(\d+)", path).groups()
                information_path = os.path.join(country_tif, f'Wu Yong/{other_scenario}/inform_sum_sa_EVI_SIF_{chunk_i}_{chunk_j}.csv')
                with rasterio.open(path) as ref_src, rasterio.open(geobioregion_path) as bio_src:
                    # 读取csv
                    df = pd.read_csv(information_path)
                    biogeo_values = []
                    for _, row in df.iterrows():
                        r, c = int(row["row"]), int(row["col"])

                        # 根据行列号获取空间坐标 (x, y)
                        x, y = ref_src.xy(r, c)

                        # 转换为 biogeoregion 栅格的行列号
                        try:
                            br_row, br_col = bio_src.index(x, y)
                            bio_value = bio_src.read(1)[br_row, br_col]
                            if bio_value == 15:
                                biome, nearest_biome, transform = build_biome_nearest_lookup(geobioregion_path)
                                biome_val = get_biome_at_point(x, y, transform, biome, nearest_biome)
                                bio_value = biome_val
                        except IndexError:
                            bio_value = None  # 超出范围

                        biogeo_values.append(bio_value)
                    df["biogeo"] = biogeo_values
                    df.to_csv(information_path.replace('.csv', '_biogeo.csv'), index=False)