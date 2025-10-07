#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 5/22/2024 15:56
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
from osgeo import gdal,osr, ogr
import shutil
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
from rasterio.windows import Window
import tempfile
import shutil
spei_path = r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei03.nc'

EVI_path = r'D:\Data Collection\RS\MODIS\EVI_merge'
tqdm.pandas()
# 定义格式化函数
def format_year(x, pos):
    return '%d' % x


def logistic_function(t, a, b, c):
    return a / (1 + np.exp(-b * (t - c)))

def smooth_data(data, window_size=7, poly_order=2):
    return savgol_filter(data, window_size, poly_order)

def calculate_sos_eos(time, evi, smoothed_evi, sos_threshold=0.3, eos_threshold=0.3):
    # Fit logistic model
    popt, _ = curve_fit(logistic_function, time, smoothed_evi, maxfev=10000)
    fitted_evi = logistic_function(time, *popt)

    # Calculate annual amplitude
    amplitude = np.max(fitted_evi) - np.min(fitted_evi)

    # Find SOS and EOS
    sos_index = np.argmax(fitted_evi >= (np.min(fitted_evi) + sos_threshold * amplitude))
    eos_index = len(fitted_evi) - np.argmax(fitted_evi[::-1] <= (np.min(fitted_evi) + eos_threshold * amplitude)) - 1

    sos_date = time[sos_index]
    eos_date = time[eos_index]

    return sos_date, eos_date, fitted_evi

def read_evi_tif(file_path):
    dataset = gdal.Open(file_path)
    band = dataset.GetRasterBand(1)
    evi_data = band.ReadAsArray()
    if evi_data.min() == -3000:evi_data = evi_data / 10000
    return evi_data
def process_region_evi_files(region_path):
    region_indexs = os.listdir(region_path)
    for region_index in region_indexs:
        sub_region_path = os.path.join(region_path,region_index)
        evi_files = sorted(glob.glob(os.path.join(region_path, '*.tif')))
        years = range(2000, 2024)
        time_points = np.arange(1, 366, 16)
        sos_list = []
        eos_list = []

        for year, evi_file in zip(years, evi_files):
            evi_data = read_evi_tif(evi_file)
            # Assuming EVI data is averaged over the region
            avg_evi = np.mean(evi_data, axis=(0, 1))
            smoothed_evi = smooth_data(avg_evi)
            sos_date, eos_date, _ = calculate_sos_eos(time_points, avg_evi, smoothed_evi)
            sos_list.append(sos_date)
            eos_list.append(eos_date)

        return sos_list, eos_list



class legacy_effects():
    def __init__(self):
       pass

    def drought_resampleing(self,drought_data,transforms_information,target_file,time_scale,drought_average_length):
        '''

        :param drought_data: 干旱数据
        :param transforms_information: 投影信息
        :param target_file: 目标文件（用于提取目标文件的分辨率）
        :param time_scale: spei 对应的参数 01-48
        :param drought_average_length: 取均值的月份数
        :return:
        '''

        # 定义输出文件名
        output_filename = 'temp/resampled_classified_drought/classified_spei.tif'

        # 获取 spei_average 的行数和列数
        rows, cols = drought_data.shape

        # 定义地理范围（经纬度范围）
        lon_min, lon_max = transforms_information[0].min(), transforms_information[0].max()
        lat_min, lat_max = transforms_information[1].min(), transforms_information[1].max()

        # 计算分辨率（像素大小）
        pixel_width = (lon_max - lon_min) / cols
        pixel_height = (lat_max - lat_min) / rows

        # 创建栅格数据集
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(output_filename, cols, rows, 1, gdal.GDT_Byte)

        # 设置地理变换参数
        ds.SetGeoTransform((lon_min, pixel_width, 0, lat_max, 0, -pixel_height))

        # 设置投影信息
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # 使用 WGS84 坐标系
        ds.SetProjection(srs.ExportToWkt())

        # 写入数据
        band = ds.GetRasterBand(1)
        band.WriteArray(np.flipud(drought_data))

        # 关闭数据集
        ds = None

        print("GeoTIFF 文件已生成：", output_filename)

        # 使用 GDAL 来重新采样 TIFF 文件到500米分辨率
        gdal.Warp("temp/resampled_classified_drought/resampled_classified_spei{}_{}.tif".format(str(time_scale),str(drought_average_length)), "temp/resampled_classified_drought/classified_spei.tif", xRes=target_file.GetGeoTransform()[1], yRes=abs(target_file.GetGeoTransform()[5]))
    def drought_classification(self,drought_data):
        # 创建一个新的数组来存储分类后的数据
        classified_array = np.zeros_like(drought_data)
        classified_array[drought_data < -2] = 2
        classified_array[(drought_data >= -2) & (drought_data < -1)] = 1
        return classified_array
    def drought_plot_cut(self,drought_path,plot_path):
        '''
        geotransform = (
            originX,    # Top left x-coordinate
            pixelWi     dth, # Pixel width (x-resolution)
            xRotation,  # Rotation (usually 0)
            originY,    # Top left y-coordinate
            yRotation,  # Rotation (usually 0)
            pixelHeight # Pixel height (y-resolution, typically negative)
        )
        originX 和 originY 是栅格数据左上角的坐标（通常是西北角）。
        pixelWidth 和 pixelHeight 定义了栅格像素的大小。注意：pixelHeight 通常是负值，因为 Y 坐标是从上到下递减的。
        xRotation 和 yRotation 是旋转项，通常为 0，表示栅格没有旋转。
        minx = plot_data_geo_transform[0]

        获取栅格数据的左上角X坐标，即栅格的最西端（最左边）的X坐标。
        maxy = plot_data_geo_transform[3]

        获取栅格数据的左上角Y坐标，即栅格的最北端（最上边）的Y坐标。
        maxx = minx + plot_data_geo_transform[1] * plot_data.RasterXSize

        计算栅格数据的右边界（最大X坐标）。
        plot_data_geo_transform[1] 是像素宽度。
        plot_data.RasterXSize 是栅格数据的列数（X方向的像素数）。
        maxx 是左上角X坐标加上总宽度，得到栅格的最东端（最右边）的X坐标。
        miny = maxy + plot_data_geo_transform[5] * plot_data.RasterYSize

        计算栅格数据的下边界（最小Y坐标）。
        plot_data_geo_transform[5] 是像素高度（通常是负值）。
        plot_data.RasterYSize 是栅格数据的行数（Y方向的像素数）。
        miny 是左上角Y坐标加上总高度，得到栅格的最南端（最下面）的Y坐标。
        :param drought_path:
        :param plot_path:
        :return:
        '''
        plot_data = gdal.Open(plot_path)
        # 打开plot的 TIFF 文件以获取其边界
        plot_data_geo_transform = plot_data.GetGeoTransform()
        plot_data_proj = plot_data.GetProjection()
        # 获取小 TIFF 文件的边界
        minx = plot_data_geo_transform[0]
        maxy = plot_data_geo_transform[3]
        maxx = minx + plot_data_geo_transform[1] * plot_data.RasterXSize
        miny = maxy + plot_data_geo_transform[5] * plot_data.RasterYSize
        # # 使用 gdal.Warp 进行裁剪
        # gdal.Warp(
        #     'plot_drought.tif', drought_path,
        #     outputBounds=(minx, miny, maxx, maxy),
        #     dstSRS=plot_data_proj
        # )
        # 使用 gdal.Translate 进行裁剪
        gdal.Translate(
            'plot_drought.tif', drought_path,
            projWin=[minx, maxy, maxx, miny]
        )
        # 关闭数据集
        plot_data = None
    def drought_phenology_extraction(self,phenology_path,drought_data,transform_information,time_scale,drought_average_length):
        if not os.path.exists('phenology_results/MODIS500/phenology/spei_{}_average{}'.format(time_scale,drought_average_length)):
            os.makedirs('phenology_results/MODIS500/phenology/spei_{}_average{}'.format(time_scale,drought_average_length))
        drought_data = self.drought_classification(drought_data)
        drought_data_resample = None
        invalid_drought = {'spei{}_{}'.format(str(time_scale),str(drought_average_length)):[]}
        if os.path.exists('invalid_droughts_phenology.json'):
            with open('invalid_droughts_phenology.json','r') as f:
                invalid_drought = json.load(f)

        plots = os.listdir(phenology_path)
        df_all_plots_drought = pd.DataFrame()   #干旱开始对应
        df_all_plots_edrought = pd.DataFrame()  #极端干旱对应的pd
        for index,plot in enumerate(plots):
            phenology_plot_path = os.path.join(phenology_path,plot)
            phenology_plot_files = glob.glob(os.path.join(phenology_plot_path,'*_reprojection240611.tif'))
            if plot in invalid_drought['spei{}_{}'.format(str(time_scale),str(drought_average_length))]:continue
            # 干旱重采样到modis的分辨率
            self.drought_resampleing(drought_data,transform_information,gdal.Open(phenology_plot_files[0]),time_scale,drought_average_length)
            # 对应plot的drought数据裁剪
            self.drought_plot_cut("temp/resampled_classified_drought/resampled_classified_spei{}_{}.tif".format(str(time_scale),str(drought_average_length)),phenology_plot_files[0])
            drought_data_resample_plot = gdal.Open('plot_drought.tif').ReadAsArray()
            if drought_data_resample_plot[drought_data_resample_plot==1].__len__() == 0 and drought_data_resample_plot[drought_data_resample_plot==2].__len__() == 0:
                if plot not in invalid_drought['spei{}_{}'.format(str(time_scale),str(drought_average_length))]:
                    invalid_drought['spei{}_{}'.format(str(time_scale),str(drought_average_length))].append(plot)
                    with open('invalid_droughts_phenology.json', 'w') as f:
                        json.dump(invalid_drought,f)
                continue
            #     干旱开始【-2，-1】,分类为1，与极端干旱【<-2】分类为2,掩膜
            drought_mask = np.where(drought_data_resample_plot == 1)
            extreme_drought_mask = np.where(drought_data_resample_plot == 2)

            result_df_drought =pd.DataFrame({
                        'plot':plot,
                        'row': drought_mask[0],
                        'col': drought_mask[1],
                    })
            result_df_edrought =pd.DataFrame({
                        'plot':plot,
                        'row': extreme_drought_mask[0],
                        'col': extreme_drought_mask[1],
                    })
            # sos_data = np.empty((len(phenology_plot_files),drought_data_resample_plot.shape[0],drought_data_resample_plot.shape[1]))
            # eos_data = np.empty((len(phenology_plot_files), drought_data_resample_plot.shape[0],drought_data_resample_plot.shape[1]))
            for phenology_plot_file in phenology_plot_files:
                # 使用重投影后的数据分析
                # phenology_plot_file:'D:\\Data Collection\\RS\\MODIS\\phenology\\10\\2001_01_01_days_reprojection.tif'
                year_information = re.search(r'(\d{4})_\d{2}_\d{2}', phenology_plot_file).group(1)
                print('正在处理第{}块第{}年'.format(plot,year_information))
                phenology_plot_file_array = gdal.Open(phenology_plot_file).ReadAsArray()
                sos_data_plot = phenology_plot_file_array[0,:,:]
                eos_data_plot = phenology_plot_file_array[2,:,:]
                # 干旱开始对应的数据
                sos_data_plot_drought_mask = sos_data_plot[drought_mask]
                eos_data_plot_drought_mask = eos_data_plot[drought_mask]
                # 极端干旱对应的数据
                sos_data_plot_edrought_mask = sos_data_plot[extreme_drought_mask]
                eos_data_plot_edrought_mask = eos_data_plot[extreme_drought_mask]

                # 填充结果
                result_df_drought['sos_{}'.format(year_information)] = sos_data_plot_drought_mask
                result_df_drought['eos_{}'.format(year_information)] = eos_data_plot_drought_mask

                result_df_edrought['sos_{}'.format(year_information)] = sos_data_plot_edrought_mask
                result_df_edrought['eos_{}'.format(year_information)] = eos_data_plot_edrought_mask

            result_df_drought.to_csv('phenology_results/MODIS500/phenology/spei_{}_average{}/originaldata/drought_result_{}.csv'.format(time_scale,drought_average_length,plot))
            result_df_edrought.to_csv('phenology_results/MODIS500/phenology/spei_{}_average{}/originaldata/extreme_drought_result_{}.csv'.format(time_scale,drought_average_length,plot))
            df_all_plots_drought = pd.concat([df_all_plots_drought,result_df_drought],ignore_index=True)
            df_all_plots_edrought = pd.concat([df_all_plots_edrought,result_df_edrought],ignore_index=True)


        df_all_plots_drought.to_csv('phenology_results/MODIS500/phenology/spei_{}_average{}/originaldata/drought_result_summary.csv'.format(time_scale,drought_average_length),index=False)
        df_all_plots_edrought.to_csv('phenology_results/MODIS500/phenology/spei_{}_average{}/originaldata/extreme_drought_result_summary.csv'.format(time_scale,drought_average_length), index=False)
    def drought_indices_subset_filter(self,indices_set):
        # 去除无效数据超过序列长度四分之三的数据
        threshold = int(indices_set.keys()[3:].__len__() * 0.75)

        invalid_data_counts1 = np.sum(indices_set[indices_set.keys()[3:]].values==-3000,axis=1)
        invalid_data_counts2 = np.sum(indices_set[indices_set.keys()[3:]].values==-32768,axis=1)
        # Filter the rows
        valid_rows = (invalid_data_counts1 < threshold ) & (invalid_data_counts2 < threshold)
        filtered_df = indices_set[valid_rows]
        return filtered_df
    def drought_indices_extraction(self,indices_path,drought_data,transform_information,time_scale,drought_average_length):
        if not os.path.exists('phenology_results/MODIS250/indices/spei_{}_average{}'.format(time_scale,drought_average_length)):
            os.makedirs('phenology_results/MODIS250/indices/spei_{}_average{}'.format(time_scale,drought_average_length))
        drought_data = self.drought_classification(drought_data)
        drought_data_resample = None
        invalid_drought = {'spei{}_{}'.format(str(time_scale),str(drought_average_length)):[]}
        if os.path.exists('invalid_droughts_phenology_indices.json'):
            with open('invalid_droughts_phenology_indices.json','r') as f:
                invalid_drought = json.load(f)

        plots = os.listdir(indices_path)
        df_all_plots_drought = pd.DataFrame()   #干旱开始对应
        df_all_plots_edrought = pd.DataFrame()  #极端干旱对应的pd
        for index,plot in enumerate(plots):
            indices_plot_path = os.path.join(indices_path,plot)
            indices_plot_files = glob.glob(os.path.join(indices_plot_path,'*_reprojection.tif'))
            if plot in invalid_drought['spei{}_{}'.format(str(time_scale),str(drought_average_length))]:continue
            # 干旱重采样到modis的分辨率
            self.drought_resampleing(drought_data,transform_information,gdal.Open(indices_plot_files[0]),time_scale,drought_average_length)
            # 对应plot的drought数据裁剪
            self.drought_plot_cut("temp/resampled_classified_drought/resampled_classified_spei{}_{}.tif".format(str(time_scale),str(drought_average_length)),indices_plot_files[0])
            drought_data_resample_plot = gdal.Open('plot_drought.tif').ReadAsArray()
            if drought_data_resample_plot[drought_data_resample_plot==1].__len__() == 0 and drought_data_resample_plot[drought_data_resample_plot==2].__len__() == 0:
                if plot not in invalid_drought['spei{}_{}'.format(str(time_scale),str(drought_average_length))]:
                    invalid_drought['spei{}_{}'.format(str(time_scale),str(drought_average_length))].append(plot)
                    with open('invalid_droughts_phenology_indices.json', 'w') as f:
                        json.dump(invalid_drought,f)
                continue
            #     干旱开始【-2，-1】,分类为1，与极端干旱【<-2】分类为2,掩膜
            drought_mask = np.where(drought_data_resample_plot == 1)
            extreme_drought_mask = np.where(drought_data_resample_plot == 2)

            result_df_drought =pd.DataFrame({
                        'plot':plot,
                        'row': drought_mask[0],
                        'col': drought_mask[1],
                    })
            result_df_edrought =pd.DataFrame({
                        'plot':plot,
                        'row': extreme_drought_mask[0],
                        'col': extreme_drought_mask[1],
                    })
            # sos_data = np.empty((len(indices_plot_files),drought_data_resample_plot.shape[0],drought_data_resample_plot.shape[1]))
            # eos_data = np.empty((len(indices_plot_files), drought_data_resample_plot.shape[0],drought_data_resample_plot.shape[1]))
            # 用于收集新列数据的字典
            new_columns_drought = {}
            new_columns_edrought = {}
            for indice_plot_file in indices_plot_files:
                # 使用重投影后的数据分析
                # phenology_plot_file:'D:\\Data Collection\\RS\\MODIS\\phenology\\10\\2001_01_01_days_reprojection.tif'
                year_information = re.search(r'(\d{4})_\d{2}_\d{2}', indice_plot_file).group(1)
                print('正在处理第{}块第{}年'.format(plot,year_information))
                indice_plot_file_array = gdal.Open(indice_plot_file).ReadAsArray()

                evi_plot_file = indice_plot_file_array.copy()
                evi_plot_drought_mask = evi_plot_file[drought_mask]    # 干旱开始对应的数据
                evi_plot_edrought_mask = evi_plot_file[extreme_drought_mask]    # 极端干旱对应的数据

                # 填充结果
                # result_df_drought[re.search(r'(\d{4})_\d{2}_\d{2}', indice_plot_file).group()] = evi_plot_drought_mask
                # result_df_edrought[re.search(r'(\d{4})_\d{2}_\d{2}', indice_plot_file).group()] = evi_plot_edrought_mask
                col_name = re.search(r'(\d{4})_\d{2}_\d{2}', indice_plot_file).group()
                new_columns_drought[col_name] = evi_plot_drought_mask
                new_columns_edrought[col_name] = evi_plot_edrought_mask
            # 一次性将所有新列添加到DataFrame中
            result_df_drought = pd.concat([result_df_drought, pd.DataFrame(new_columns_drought)], axis=1)
            result_df_edrought = pd.concat([result_df_edrought, pd.DataFrame(new_columns_edrought)], axis=1)
            # 由于其中包含大量无效数据，即很多数据是-3000和-32768，此处做一个筛选，即去除掉这些异常数据比例过大所在的行
            result_df_drought = self.drought_indices_subset_filter(result_df_drought)
            result_df_edrought = self.drought_indices_subset_filter(result_df_edrought)
            result_df_drought.to_csv('phenology_results/MODIS250/indices/spei_{}_average{}/drought_evi_result_{}.csv'.format(time_scale,drought_average_length,plot))
            result_df_edrought.to_csv('phenology_results/MODIS250/indices/spei_{}_average{}/extreme_drought_evi_result_{}.csv'.format(time_scale,drought_average_length,plot))
            df_all_plots_drought = pd.concat([df_all_plots_drought,result_df_drought],ignore_index=True)
            df_all_plots_edrought = pd.concat([df_all_plots_edrought,result_df_edrought],ignore_index=True)


        df_all_plots_drought.to_csv('phenology_results/MODIS250/indices/spei_{}_average{}/drought_result_summary.csv'.format(time_scale,drought_average_length),index=False)
        df_all_plots_edrought.to_csv('phenology_results/MODIS250/indices/spei_{}_average{}/extreme_drought_result_summary.csv'.format(time_scale,drought_average_length), index=False)
    def get_geographic_corrd(self,geotransform,rasterxsize,rasterysize):
        # 获取栅格的宽度和高度
        width = rasterxsize
        height = rasterysize

        # 计算边界坐标
        min_lon = geotransform[0]
        max_lon = geotransform[0] + width * geotransform[1]
        max_lat = geotransform[3]
        min_lat = geotransform[3] + height * geotransform[5]

        # 创建经纬度网格
        lons = np.linspace(min_lon, max_lon, width)
        lats = np.linspace(max_lat, min_lat, height)

        # 使用 meshgrid 创建二维网格
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # 将结果组合成一个三维数组，其中最后一个维度包含经度和纬度
        coords = np.stack((lon_grid, lat_grid), axis=-1)
        print(f"经度范围: {min_lon} to {max_lon}")
        print(f"纬度范围: {min_lat} to {max_lat}")
        print(f"坐标数组形状: {coords.shape}")
        return lats,lons,coords
    def drought_indices_extraction_nc(self,indices_path,drought_data,transform_information,time_scale,drought_average_length):
        '''
        干旱区域指数提取，存为nc
        :param indices_path:
        :param drought_data:
        :param transform_information:
        :param time_scale:
        :param drought_average_length:
        :return:
        '''
        if not os.path.exists('phenology_results/MODIS250/indices/spei_{}_average{}'.format(time_scale,drought_average_length)):
            os.makedirs('phenology_results/MODIS250/indices/spei_{}_average{}'.format(time_scale,drought_average_length))
        drought_data = self.drought_classification(drought_data)
        invalid_drought = {'spei{}_{}'.format(str(time_scale),str(drought_average_length)):[]}
        if os.path.exists('invalid_droughts_phenology_indices.json'):
            with open('invalid_droughts_phenology_indices.json','r') as f:
                invalid_drought = json.load(f)

        plots = os.listdir(indices_path)
        df_all_plots_drought = pd.DataFrame()   #干旱开始对应
        df_all_plots_edrought = pd.DataFrame()  #极端干旱对应的pd
        # evi_data_merge_path = r'D:\Data Collection\RS\MODIS\EVI_merge/2000_03_05_reprojection.tif'
        # evi_data_merge = gdal.Open(evi_data_merge_path)
        # lats_merge,lons_merge,coords_merge = self.get_geographic_corrd(evi_data_merge.GetGeoTransform(),evi_data_merge.RasterXSize,evi_data_merge.RasterYSize)
        # date_range = pd.to_datetime(self.dates)
        # date_range = date_range[date_range.year>2000]
        # years = np.arange(2001,2024)
        # data = np.full((int(date_range.shape[0]/years.shape[0]), len(lats_merge), len(lons_merge)),-32768)
        #
        # ds = xarray.Dataset(
        #     {
        #         'EVI': (['time', 'lat', 'lon'], data)
        #     },
        #     coords={
        #         'time': date_range,
        #         'lat': lats_merge,
        #         'lon': lons_merge
        #     }
        # )
        # ds.time.attrs['long_name'] = 'time'
        # ds.time.attrs['units'] = 'years since 2001-01-01'
        # ds.lat.attrs['long_name'] = 'latitude'
        # ds.lat.attrs['units'] = 'degrees_north'
        # ds.lon.attrs['long_name'] = 'longitude'
        # ds.lon.attrs['units'] = 'degrees_east'
        # ds.vegetation_index.attrs['long_name'] = 'Vegetation Index'
        # ds.vegetation_index.attrs['units'] = '1'
        # # 添加全局属性
        # ds.attrs['title'] = 'Vegetation Index Data'
        # ds.attrs['summary'] = 'Vegetation index data for the European region from 2001 to 2023'
        # ds.attrs['history'] = 'Created ' + pd.Timestamp.now().isoformat()
        # 保存为NetCDF文件
        # output_path = 'test.nc'
        for index,plot in enumerate(plots):
            indices_plot_path = os.path.join(indices_path,plot)
            indices_plot_files = glob.glob(os.path.join(indices_plot_path,'*_reprojection.tif'))
            if plot in invalid_drought['spei{}_{}'.format(str(time_scale),str(drought_average_length))]:continue
            # 干旱重采样到modis的分辨率
            self.drought_resampleing(drought_data,transform_information,gdal.Open(indices_plot_files[0]),time_scale,drought_average_length)
            # 对应plot的drought数据裁剪
            self.drought_plot_cut("temp/resampled_classified_drought/resampled_classified_spei{}_{}.tif".format(str(time_scale),str(drought_average_length)),indices_plot_files[0])
            drought_data_resample_plot = gdal.Open('plot_drought.tif').ReadAsArray()
            if drought_data_resample_plot[drought_data_resample_plot==1].__len__() == 0 and drought_data_resample_plot[drought_data_resample_plot==2].__len__() == 0:
                if plot not in invalid_drought['spei{}_{}'.format(str(time_scale),str(drought_average_length))]:
                    invalid_drought['spei{}_{}'.format(str(time_scale),str(drought_average_length))].append(plot)
                    with open('invalid_droughts_phenology_indices.json', 'w') as f:
                        json.dump(invalid_drought,f)
                continue


            #     干旱开始【-2，-1】,分类为1，与极端干旱【<-2】分类为2,掩膜
            drought_mask = np.where(drought_data_resample_plot == 1)
            extreme_drought_mask = np.where(drought_data_resample_plot == 2)

            result_df_drought =pd.DataFrame({
                        'plot':plot,
                        'row': drought_mask[0],
                        'col': drought_mask[1],
                    })
            result_df_edrought =pd.DataFrame({
                        'plot':plot,
                        'row': extreme_drought_mask[0],
                        'col': extreme_drought_mask[1],
                    })
            # sos_data = np.empty((len(indices_plot_files),drought_data_resample_plot.shape[0],drought_data_resample_plot.shape[1]))
            # eos_data = np.empty((len(indices_plot_files), drought_data_resample_plot.shape[0],drought_data_resample_plot.shape[1]))
            # 用于收集新列数据的字典
            new_columns_drought = {}
            new_columns_edrought = {}
            for indice_plot_file in indices_plot_files:
                # 使用重投影后的数据分析
                # phenology_plot_file:'D:\\Data Collection\\RS\\MODIS\\phenology\\10\\2001_01_01_days_reprojection.tif'
                year_information = re.search(r'(\d{4})_\d{2}_\d{2}', indice_plot_file).group(1)
                print('正在处理第{}块第{}年'.format(plot,year_information))
                indice_plot_file_array = gdal.Open(indice_plot_file).ReadAsArray()

                evi_plot_file = indice_plot_file_array.copy()
                evi_plot_drought_mask = evi_plot_file[drought_mask]    # 干旱开始对应的数据
                evi_plot_edrought_mask = evi_plot_file[extreme_drought_mask]    # 极端干旱对应的数据

                # 填充结果
                # result_df_drought[re.search(r'(\d{4})_\d{2}_\d{2}', indice_plot_file).group()] = evi_plot_drought_mask
                # result_df_edrought[re.search(r'(\d{4})_\d{2}_\d{2}', indice_plot_file).group()] = evi_plot_edrought_mask
                col_name = re.search(r'(\d{4})_\d{2}_\d{2}', indice_plot_file).group()
                new_columns_drought[col_name] = evi_plot_drought_mask
                new_columns_edrought[col_name] = evi_plot_edrought_mask
            # 一次性将所有新列添加到DataFrame中
            result_df_drought = pd.concat([result_df_drought, pd.DataFrame(new_columns_drought)], axis=1)
            result_df_edrought = pd.concat([result_df_edrought, pd.DataFrame(new_columns_edrought)], axis=1)
            # 由于其中包含大量无效数据，即很多数据是-3000和-32768，此处做一个筛选，即去除掉这些异常数据比例过大所在的行
            result_df_drought = self.drought_indices_subset_filter(result_df_drought)
            result_df_edrought = self.drought_indices_subset_filter(result_df_edrought)
            result_df_drought.to_csv('phenology_results/MODIS250/indices/spei_{}_average{}/drought_evi_result_{}.csv'.format(time_scale,drought_average_length,plot))
            result_df_edrought.to_csv('phenology_results/MODIS250/indices/spei_{}_average{}/extreme_drought_evi_result_{}.csv'.format(time_scale,drought_average_length,plot))
            df_all_plots_drought = pd.concat([df_all_plots_drought,result_df_drought],ignore_index=True)
            df_all_plots_edrought = pd.concat([df_all_plots_edrought,result_df_edrought],ignore_index=True)


        df_all_plots_drought.to_csv('phenology_results/MODIS250/indices/spei_{}_average{}/drought_result_summary.csv'.format(time_scale,drought_average_length),index=False)
        df_all_plots_edrought.to_csv('phenology_results/MODIS250/indices/spei_{}_average{}/extreme_drought_result_summary.csv'.format(time_scale,drought_average_length), index=False)

    def invalid_sos_eos(self,row,sos_columns, eos_columns):
        invalid_pairs = 0
        for sos, eos in zip(sos_columns, eos_columns):
            if row[sos] > row[eos]:
                invalid_pairs += 1
        return invalid_pairs

    # Function to check invalid data points (-3278)
    def invalid_data_points(self,row,sos_columns, eos_columns):
        # 选取所有soseos列中为-32768的值，这是布尔数组
        # 对布尔数组，np.sum True 为 1 False 为 0，这样求和就是间接完成计数
        return np.sum(row[sos_columns + eos_columns] == -32768)
    def drought_phenology_clean(self,drought_phenology_result_path):
        '''
        清洗提取的干旱区的结果数据，sos》eos超过一半；异常值超过一半的都过滤掉
        :param drought_phenology_result_path:
        :return:
        '''
        drought_phenology_result = pd.read_csv(drought_phenology_result_path)

        sos_columns = [col for col in drought_phenology_result.columns if 'sos' in col]
        eos_columns = [col for col in drought_phenology_result.columns if 'eos' in col]
        drought_phenology_result['invalid_sos_eos'] = drought_phenology_result.progress_apply(lambda row : self.invalid_sos_eos(row,sos_columns,eos_columns), axis=1)
        drought_phenology_result['invalid_data_points'] = drought_phenology_result.progress_apply(lambda row: self.invalid_data_points(row,sos_columns,eos_columns), axis=1)
        filtered_df = drought_phenology_result[
            (drought_phenology_result['invalid_sos_eos'] <= len(sos_columns) / 2) & (drought_phenology_result['invalid_data_points'] <= len(sos_columns) / 2)]

        # Drop the helper columns
        filtered_df = filtered_df.drop(columns=['invalid_sos_eos', 'invalid_data_points'])

        output_path = os.path.join(os.path.split(drought_phenology_result_path)[0],'drought_result_summary_clean.csv')
        filtered_df.to_csv(output_path)

    def drought_phenology_clean_vector(self, drought_phenology_result_path):
        '''
        向量化运算，提高速度
        清洗提取的干旱区的结果数据，sos》eos超过一半；异常值超过一半的都过滤掉
        :param drought_phenology_result_path:
        :return:
        '''
        drought_phenology_result = pd.read_csv(drought_phenology_result_path)

        sos_columns = [col for col in drought_phenology_result.columns if 'sos' in col]
        eos_columns = [col for col in drought_phenology_result.columns if 'eos' in col]

        sos_values = drought_phenology_result[sos_columns].values
        eos_values = drought_phenology_result[eos_columns].values

        # Calculate invalid SOS/EOS pairs
        invalid_sos_eos_counts = np.sum(sos_values > eos_values, axis=1)

        # Calculate invalid data points
        combined_values = np.concatenate([sos_values, eos_values], axis=1)
        invalid_data_points_counts = np.sum(combined_values == -32768, axis=1)

        # Filter the rows
        valid_rows = (invalid_sos_eos_counts <= len(sos_columns) / 2) & (invalid_data_points_counts <= len(sos_columns) / 2)
        filtered_df = drought_phenology_result[valid_rows]

        output_path = os.path.join(os.path.split(drought_phenology_result_path)[0], 'drought_result_summary_clean.csv')
        filtered_df.to_csv(output_path, index=False)

    def baseline_long_term_calculation(self,row,sos_columns,eos_columns):
        # 构建有效索引，即无效值和异常值不参与基线计算
        valid_sos = [row[col] for col in sos_columns if
                     row[col] != -32768 and ~np.isnan(row[col]) and row[col] < row[col.replace('sos', 'eos')]]
        valid_eos = [row[col] for col in eos_columns if
                     row[col] != -32768 and ~np.isnan(row[col]) and row[col] > row[col.replace('eos', 'sos')]]
        if len(valid_sos) == 0 or len(valid_eos) == 0:
            return np.nan,np.nan
        else:
            sos_mean = np.mean(valid_sos)
            eos_mean = np.mean(valid_eos)
            return sos_mean, eos_mean

    def baseline_fixed_term_subset_filter(self,drought_phenology_result,drought_year,term_length):
        '''
        固定长度基线（包含镜像和固定长度）的特征子集筛选
        :param drought_phenology_result:
        :param drought_year:
        :param term_length:
        :return:
        '''

        sos_columns = ['sos_{}'.format(x) for x in range(drought_year - term_length,drought_year)]
        eos_columns = ['eos_{}'.format(x) for x in range(drought_year - term_length, drought_year)]
        sos_values = drought_phenology_result[sos_columns].values
        eos_values = drought_phenology_result[eos_columns].values

        # Calculate invalid SOS/EOS pairs
        invalid_sos_eos_counts = np.sum(sos_values > eos_values, axis=1)
        combined_values = np.concatenate([sos_values, eos_values], axis=1)
        invalid_data_points_counts = np.sum((combined_values == -32768) | (np.isnan(combined_values)), axis=1)
        # Filter the rows
        valid_rows = (invalid_sos_eos_counts == 0 ) & (invalid_data_points_counts == 0)
        filtered_df = drought_phenology_result[valid_rows]
        return filtered_df

    def after_drought_subset_filter(self,baseline_result,drought_year,drought_length,end_year):
        '''
        干旱后子集筛选，有的干旱后异常的值都给他过滤掉，比如，2018年干旱，2020年的sos和eos如果都是261，或者2020年sos>eos的这种都属于异常值
        :param baseline_result: 带有baseline列的结果
        :param drought_year: 干旱年份
        :param drought_length: 干旱持续时间
        :param end_year: 分析结束时间
        :return:
        '''


        sos_columns = ['sos_{}'.format(x) for x in range(drought_year + drought_length,end_year)]
        eos_columns = ['eos_{}'.format(x) for x in range(drought_year + drought_length, end_year)]
        sos_values = baseline_result[sos_columns].values
        eos_values = baseline_result[eos_columns].values

        # Calculate invalid SOS/EOS pairs
        invalid_sos_eos_counts = np.sum(sos_values >= eos_values, axis=1)
        combined_values = np.concatenate([sos_values, eos_values], axis=1)
        invalid_data_points_counts = np.sum((combined_values == -32768) | (np.isnan(combined_values)), axis=1)
        # Filter the rows
        valid_rows = (invalid_sos_eos_counts == 0 ) & (invalid_data_points_counts == 0)
        filtered_df = baseline_result[valid_rows]
        return filtered_df

    def baseline_analysis_visualation(self,df_,years,save_path,baseline_type):
     df = df_.copy()
     if baseline_type == 'mirror': baseline_name_sos,baseline_name_eos = 'sos_mirror_baseline','eos_mirror_baseline'
     elif baseline_type == 'long_term': baseline_name_sos,baseline_name_eos = 'sos_long_term_baseline','eos_long_term_baseline'
     else: baseline_name_sos,baseline_name_eos = 'sos_fixed_term_baseline','eos_fixed_term_baseline'
     # 计算差异
     for year in years:
        df[f'sos_diff_{year}'] = df[f'sos_{year}'] - df[baseline_name_sos]
        df[f'eos_diff_{year}'] = df[f'eos_{year}'] - df[baseline_name_eos]

     # 汇总数据
     sos_diffs = df[[f'sos_diff_{year}' for year in years]]
     eos_diffs = df[[f'eos_diff_{year}' for year in years]]

     sos_mean = sos_diffs.mean(axis=0)
     sos_sem = sos_diffs.sem(axis=0)
     eos_mean = eos_diffs.mean(axis=0)
     eos_sem = eos_diffs.sem(axis=0)

     expansion_factor = 10

     # 设置字体和大小
     plt.rcParams['font.family'] = 'Arial'
     plt.rcParams['font.size'] = 12

     # 绘制带阴影的折线图
     plt.figure(figsize=(12, 6))

     plt.subplot(1, 2, 1)
     plt.plot(years, sos_mean, 'o-',label='SOS Difference', color='blue')
     plt.fill_between(years, sos_mean - sos_sem * expansion_factor, sos_mean + sos_sem * expansion_factor, color='blue',
                      alpha=0.2)
     plt.xlabel('Year', fontsize=14)
     plt.ylabel('SOS Difference', fontsize=14)
     plt.title('SOS Difference with {} Baseline'.format(baseline_type), fontsize=16)
     plt.legend(fontsize=12)
     plt.xticks(years)
     plt.subplot(1, 2, 2)
     plt.plot(years, eos_mean, 'o-', label='EOS Difference', color='red')
     plt.fill_between(years, eos_mean - eos_sem * expansion_factor, eos_mean + eos_sem * expansion_factor, color='red',
                      alpha=0.2)
     plt.xlabel('Year', fontsize=14)
     plt.ylabel('EOS Difference', fontsize=14)
     plt.title('EOS Difference with {} Baseline'.format(baseline_type), fontsize=16)
     plt.legend(fontsize=12)
     plt.xticks(years)
     plt.tight_layout()

     # 保存图像
     plt.savefig(os.path.join(save_path,baseline_type+'sos_eos_difference.eps'), format='eps', dpi=300)
     plt.savefig(os.path.join(save_path,baseline_type+'sos_eos_difference.pdf'), format='pdf', dpi=300)
     plt.savefig(os.path.join(save_path,baseline_type+'sos_eos_difference.jpg'), format='jpeg', dpi=300)

    def legacy_analysis_baselines(self,drought_phenology_result_path,drought_year,drought_length,fixed_length,save_path):
        '''
        基于基线模式的遗留效应
        :param drought_phenology_result_path:
        :param drought_year:
        :param drought_length:
        :param fixed_length:
        :param save_path:
        :return:
        '''
        drought_phenology_result = pd.read_csv(drought_phenology_result_path)
        years = [2020, 2021, 2022, 2023]
        # 构建镜像基线
        mirror_sos_columns = ['sos_{}'.format(x) for x in range(drought_year-drought_length,drought_year)]
        mirror_eos_columns = ['eos_{}'.format(x) for x in range(drought_year - drought_length, drought_year)]

        mirror_filtered_df = self.baseline_fixed_term_subset_filter(drought_phenology_result,drought_year,drought_length)
        # mirror_filtered_df['sos_mirror_baseline'] = mirror_filtered_df[mirror_sos_columns].values.mean(1)
        # mirror_filtered_df['eos_mirror_baseline'] = mirror_filtered_df[mirror_eos_columns].values.mean(1)
        mirror_filtered_df = mirror_filtered_df.copy()
        mirror_filtered_df.loc[:, 'sos_mirror_baseline'] = mirror_filtered_df[mirror_sos_columns].mean(axis=1)
        mirror_filtered_df.loc[:, 'eos_mirror_baseline'] = mirror_filtered_df[mirror_eos_columns].mean(axis=1)
        # 过滤掉干旱后面有异常的行
        mirror_filtered_df = self.after_drought_subset_filter(mirror_filtered_df,drought_year,drought_length,years[-1]+1)

        mirror_filtered_df.to_csv(os.path.join(save_path,'mirror.csv'))
        self.baseline_analysis_visualation(mirror_filtered_df,years,save_path,'mirror')

        # 长期均值基线
        long_term_sos_columns = [col for col in drought_phenology_result.columns if 'sos' in col]
        long_term_eos_columns = [col for col in drought_phenology_result.columns if 'eos' in col]
        long_term_sos_values = drought_phenology_result[long_term_sos_columns].values
        long_term_eos_values = drought_phenology_result[long_term_eos_columns].values
        # Calculate invalid SOS/EOS pairs
        invalid_long_term_sos_eos_counts = np.sum(long_term_sos_values > long_term_eos_values, axis=1)
        long_term_combined_values = np.concatenate([long_term_sos_values, long_term_eos_values], axis=1)
        invalid_long_term_data_points_counts = np.sum((long_term_combined_values == -32768) | (np.isnan(long_term_combined_values)), axis=1)
        # Filter the rows
        long_term_valid_rows = (invalid_long_term_sos_eos_counts == 0) & (invalid_long_term_data_points_counts == 0)
        long_term_invalid_rows = ~long_term_valid_rows
        # 选取有效子集和异常子集，有效子集直接向量化运算，异常机子需要通过每行判断
        long_term_valid_filtered_df = drought_phenology_result[long_term_valid_rows]
        long_term_invalid_filtered_df = drought_phenology_result[long_term_invalid_rows]

        valid_sos_lt_baseline_value = long_term_valid_filtered_df[long_term_sos_columns].values.mean(1)
        valid_eos_lt_baseline_value = long_term_valid_filtered_df[long_term_eos_columns].values.mean(1)
        sos_eos_invalid_values = long_term_invalid_filtered_df.progress_apply(lambda row: self.baseline_long_term_calculation(row,long_term_sos_columns,long_term_eos_columns), axis = 1)
        sos_eos_invalid_values_ = pd.DataFrame(sos_eos_invalid_values)
        sos_eos_invalid_values_[['sos','eos']] = pd.DataFrame(sos_eos_invalid_values.to_list(), index=sos_eos_invalid_values.index)
        invalid_sos_lt_baseline_value, invalid_eos_lt_baseline_value = sos_eos_invalid_values_['sos'],sos_eos_invalid_values_['eos']

        # 构建空数组，并且根据有效和异常子集索引填入对应的数据
        final_sos_lt_baseline_value,final_eos_lt_baseline_value = np.empty_like(long_term_valid_rows,dtype=np.float64),np.empty_like(long_term_valid_rows,dtype=np.float64)
        final_sos_lt_baseline_value[long_term_valid_rows],final_sos_lt_baseline_value[long_term_invalid_rows] = valid_sos_lt_baseline_value,invalid_sos_lt_baseline_value
        final_eos_lt_baseline_value[long_term_valid_rows],final_eos_lt_baseline_value[long_term_invalid_rows] = valid_eos_lt_baseline_value,invalid_eos_lt_baseline_value

        drought_phenology_result_lt = drought_phenology_result
        drought_phenology_result_lt['sos_long_term_baseline'] = final_sos_lt_baseline_value
        drought_phenology_result_lt['eos_long_term_baseline'] = final_eos_lt_baseline_value
        # 过滤掉干旱后面有异常的行
        drought_phenology_result_lt = self.after_drought_subset_filter(drought_phenology_result_lt,drought_year,drought_length,years[-1]+1)
        drought_phenology_result_lt.to_csv(os.path.join(save_path,'long_term.csv'))
        self.baseline_analysis_visualation(drought_phenology_result_lt, years, save_path, 'long_term')
        # 构建固定长度基线:ft表示fixed_term
        ft_sos_columns = ['sos_{}'.format(x) for x in range(drought_year - fixed_length,drought_year)]
        ft_eos_columns = ['eos_{}'.format(x) for x in range(drought_year - fixed_length, drought_year)]
        ft_filtered_df = self.baseline_fixed_term_subset_filter(drought_phenology_result,drought_year,fixed_length)
        ft_filtered_df = ft_filtered_df.copy()
        ft_filtered_df.loc[:, 'sos_fixed_term_baseline'] = ft_filtered_df[ft_sos_columns].mean(axis=1)
        ft_filtered_df.loc[:, 'eos_fixed_term_baseline'] = ft_filtered_df[ft_eos_columns].mean(axis=1)

        # 过滤掉干旱后面有异常的行
        ft_filtered_df = self.after_drought_subset_filter(ft_filtered_df,drought_year,fixed_length,years[-1]+1)
        ft_filtered_df.to_csv(os.path.join(save_path,'fixed_term.csv'))
        self.baseline_analysis_visualation(ft_filtered_df,years,save_path,'fixed_term')

    def cal_my_pacf_ols(self,x, nlags):
       """
       自己实现pacf，原理使用的就是ols(最小二乘法)
       :param x:
       :param nlags:
       :return:
       """
       pacf = np.empty(nlags + 1) * 0
       pacf[0] = 1.0

       xlags, x0 = lagmat(x, nlags, original="sep")
       xlags = add_constant(xlags)

       for k in range(1, nlags + 1):
          params = lstsq(xlags[k:, :(k + 1)], x0[k:], rcond=None)[0]
          pacf[k] = params[-1]

       return pacf

    def cal_my_yule_walker(self,x, nlags):
       """
       自己实现yule_walker理论
       :param x:
       :param nlags:
       :return:
       """
       x = np.array(x, dtype=np.float64)
       x -= x.mean()
       n = x.shape[0]

       r = np.zeros(shape=nlags + 1, dtype=np.float64)
       r[0] = (x ** 2).sum() / n

       for k in range(1, nlags + 1):
          r[k] = (x[0:-k] * x[k:]).sum() / (n - k * 1)

       R = toeplitz(c=r[:-1])
       result = np.linalg.solve(R, r[1:])
       return result

    def cal_my_pacf_yw(self,x, nlags):
     """
     自己通过yule_walker方法求出pacf的值
     :param x:
     :param nlags:
     :return:
     """
     try:
       pacf = np.empty(nlags + 1) * 0
       pacf[0] = 1.0
       for k in range(1, nlags + 1):
          pacf[k] = self.cal_my_yule_walker(x, nlags=k)[-1]
       return pacf
     except:
        return None

    def invalid_data_process(self,data,indicator):
       """
       对传入的数据进行滤波和预处理，如果nan超过一半就舍弃，剩下的数据进行插值处理
       :param data:
       :return:
       """
       data_ = data.dropna(subset=[indicator+'_2018',indicator+'_2019'], how='any')

       nan_count = data_.isnull().sum(axis=1)
       threshold = int(data_.shape[1] / 2)
       filtered_rows = data_[nan_count <= threshold]
       interpolated_data = filtered_rows.interpolate(method='linear', axis=1)
       return interpolated_data
    def legacy_analysis_AR(self,drought_phenology_result_path,drought_year,save_path):
        drought_phenology_result = pd.read_csv(drought_phenology_result_path)
        end_year = 2024
        # 筛选用于计算的子集
        sos_columns = ['plot','row','col'] + ['sos_{}'.format(x) for x in range(drought_year,end_year)]
        eos_columns = ['plot','row','col'] + ['eos_{}'.format(x) for x in range(drought_year, end_year)]
        drought_phenology_result_sos = drought_phenology_result[sos_columns]
        drought_phenology_result_eos = drought_phenology_result[eos_columns]
        # 数据清理
        drought_phenology_result_sos = self.invalid_data_process(drought_phenology_result_sos,'sos')
        drought_phenology_result_eos = self.invalid_data_process(drought_phenology_result_eos,'eos')
        # 定义结果dataframe
        sos_pacf_ = drought_phenology_result_sos[['plot','row','col']]
        eos_pacf_ = drought_phenology_result_eos[['plot', 'row', 'col']]
        # 并行计算相关系数
        sos_pacf = Parallel(n_jobs=-1)(
            delayed(self.cal_my_pacf_yw)(row, 5) for index, row in tqdm(drought_phenology_result_sos[['sos_{}'.format(x) for x in range(drought_year,end_year)]].iterrows(),total=drought_phenology_result_sos.shape[0]))
        eos_pacf = Parallel(n_jobs=-1)(
            delayed(self.cal_my_pacf_yw)(row, 5) for index, row in tqdm(drought_phenology_result_eos[['eos_{}'.format(x) for x in range(drought_year,end_year)]].iterrows(),total=drought_phenology_result_eos.shape[0]))
        sos_pacf = np.array([
         # 检查元素 x 是否为 None
         x if x is not None else
         # 如果 x 是 None，找到 sos_pacf 中第一个非 None 的元素，获取它的形状，然后创建一个相同形状的 None 数组
         np.full(next((x for x in sos_pacf if x is not None), None).shape, None)
         # 遍历 sos_pacf 中的每个元素
         for x in sos_pacf
        ])

        # 处理 eos_pacf 数组
        eos_pacf = np.array([
         # 检查元素 x 是否为 None
         x if x is not None else
         # 如果 x 是 None，找到 eos_pacf 中第一个非 None 的元素，获取它的形状，然后创建一个相同形状的 None 数组
         np.full(next((x for x in eos_pacf if x is not None), None).shape, None)
         # 遍历 eos_pacf 中的每个元素
         for x in eos_pacf
        ])
        sos_pacf_[['sos_{}'.format(x) for x in range(drought_year,end_year)]] = sos_pacf
        eos_pacf_[['eos_{}'.format(x) for x in range(drought_year,end_year)]] = eos_pacf
        sos_pacf_.dropna()
        eos_pacf_.dropna()
        # 保存结果
        sos_pacf_.to_csv(os.path.join(save_path,'sos_pacf.csv'))
        eos_pacf_.to_csv(os.path.join(save_path,'eos_pacf.csv'))
        # 可视化
        sos_pacf_['mean_post'] = sos_pacf_[['sos_{}'.format(x) for x in range(drought_year, end_year)]].mean(axis=1)
        eos_pacf_['mean_post'] = eos_pacf_[['eos_{}'.format(x) for x in range(drought_year, end_year)]].mean(axis=1)
        sos_pacf_ = sos_pacf_[(sos_pacf_['mean_post'] >= -1) & (sos_pacf_['mean_post'] <= 1)]
        eos_pacf_ = eos_pacf_[(eos_pacf_['mean_post'] >= -1) & (eos_pacf_['mean_post'] <= 1)]
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 12
        plt.figure(figsize=(10, 5))
        plt.plot(sos_pacf_[['sos_{}'.format(x) for x in range(drought_year, end_year)]].mean(), marker='o', label='PACF(post drought)')
        # 添加数据标签
        for i, txt in enumerate(sos_pacf_[['sos_{}'.format(x) for x in range(drought_year, end_year)]].mean()):
         plt.text(i, txt, f'{txt:.2f}', ha='center', va='bottom')
        plt.title('PACF')
        plt.xlabel('Post drought years')
        plt.ylabel('PACF')
        plt.legend()
        # 保存图像
        plt.savefig(os.path.join(save_path, 'sos_pacf.eps'), format='eps', dpi=300)
        plt.savefig(os.path.join(save_path, 'sos_pacf.pdf'), format='pdf', dpi=300)
        plt.savefig(os.path.join(save_path, 'sos_pacf.jpg'), format='jpeg', dpi=300)
        plt.close()
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 12
        plt.figure(figsize=(10, 5))
        plt.plot(eos_pacf_[['eos_{}'.format(x) for x in range(drought_year, end_year)]].mean(), marker='o', label='PACF(post drought)')
        # 添加数据标签
        for i, txt in enumerate(eos_pacf_[['eos_{}'.format(x) for x in range(drought_year, end_year)]].mean()):
         plt.text(i, txt, f'{txt:.2f}', ha='center', va='bottom')
        plt.title('PACF')
        plt.xlabel('Post drought years')
        plt.ylabel('PACF')
        plt.legend()
        # 保存图像
        plt.savefig(os.path.join(save_path, 'eos_pacf.eps'), format='eps', dpi=300)
        plt.savefig(os.path.join(save_path, 'eos_pacf.pdf'), format='pdf', dpi=300)
        plt.savefig(os.path.join(save_path, 'eos_pacf.jpg'), format='jpeg', dpi=300)


    # 4. 定义函数，用于判断物候数据栅格点对应地理范围内满足条件的土地利用数据和扰动数据的比例
    def check_landcover(self,phenology_row, phenology_col, landuse_array, landuse_geotransform,phenology_geotransform,required_trees,ifeverygreen = False):
        '''
        判断物候数据栅格点对应地理范围内满足条件的土地利用数据栅格点的比例
        :param phenology_row: 物候数据（掩膜数据）行列
        :param phenology_col:
        :param landuse_array:土地利用数据 选择森林区域
        :param landuse_geotransform:土地利用数据地理参数
        :param phenology_geotransform:物候数据（掩膜数据）地理参数
        :required_trees: 需要的种类
        :ifeverygreen: 是否常绿
        :return:
        '''

        # 计算物候数据栅格点对应地理范围的左上角坐标
        phenology_ulx = phenology_geotransform[0] + phenology_col * phenology_geotransform[1]
        phenology_uly = phenology_geotransform[3] + phenology_row * phenology_geotransform[5]
        # 计算物候数据栅格点对应地理范围的右下角坐标
        phenology_lrx = phenology_ulx + phenology_geotransform[1]
        phenology_lry = phenology_uly + phenology_geotransform[5]

        # 根据土地利用数据的地理信息，计算物候数据栅格点对应地理范围内的土地利用数据的行号和列号
        landuse_row_start = int((phenology_uly - landuse_geotransform[3]) / landuse_geotransform[5])
        landuse_row_end = int((phenology_lry - landuse_geotransform[3]) / landuse_geotransform[5]) + 1
        landuse_col_start = int((phenology_ulx - landuse_geotransform[0]) / landuse_geotransform[1])
        landuse_col_end = int((phenology_lrx - landuse_geotransform[0]) / landuse_geotransform[1]) + 1

        # 提取物候数据栅格点对应地理范围内的土地利用数据
        landuse_sub_array = landuse_array[landuse_row_start:landuse_row_end, landuse_col_start:landuse_col_end]
        if ifeverygreen:
            landuse_ratio = np.sum((landuse_sub_array == 24)) / landuse_sub_array.size
            landuse_number = np.sum((landuse_sub_array == 24))
            # 计算满足条件的土地利用数据栅格点的数量
            count_24 = np.sum(landuse_sub_array == 24)
            # 找到数量最多的类别
            category_counts = {24: count_24}
            max_category = max(category_counts, key=category_counts.get)  # 获取数量最多的类别
            return landuse_number, landuse_sub_array.size, max_category
        else:
            if len(required_trees) == 2:
                # 计算满足条件的土地利用数据栅格点的比例
                landuse_ratio = np.sum((landuse_sub_array == 23) | (landuse_sub_array == 25)) / landuse_sub_array.size
                landuse_number = np.sum((landuse_sub_array == 23) | (landuse_sub_array == 25))

                # 计算满足条件的土地利用数据栅格点的数量
                count_23 = np.sum(landuse_sub_array == 23)
                count_25 = np.sum(landuse_sub_array == 25)
                # 找到数量最多的类别
                category_counts = {23: count_23, 25: count_25}
                max_category = max(category_counts, key=category_counts.get)  # 获取数量最多的类别

                return landuse_number,landuse_sub_array.size,max_category
            else:
                # 计算满足条件的土地利用数据栅格点的比例
                landuse_ratio = np.sum((landuse_sub_array == 23)) / landuse_sub_array.size
                landuse_number = np.sum((landuse_sub_array == 23))
                # 计算满足条件的土地利用数据栅格点的数量
                count_23 = np.sum(landuse_sub_array == 23)
                # 找到数量最多的类别
                category_counts = {23: count_23}
                max_category = max(category_counts, key=category_counts.get)  # 获取数量最多的类别
                return landuse_number, landuse_sub_array.size, max_category

    def check_disturbance(self,phenology_row, phenology_col, disturbance_array, disturbance_geotransform, year,phenology_geotransform):
        '''
        判断物候数据栅格点对应地理范围内满足条件的扰动数据的比例
        :param phenology_row: 物候数据（掩膜数据）栅格点行列
        :param phenology_col:
        :param disturbance_array: 扰动数据
        :param disturbance_geotransform: 扰动数据地理参数
        :param year: 年份对象
        :param phenology_geotransform: 物候数据（掩膜数据）地理参数，用于确定物候数据（掩膜数据）对应栅格点对应的扰动栅格点的范围
        :return:
        '''

        # 计算物候数据栅格点对应地理范围的左上角坐标
        phenology_ulx = phenology_geotransform[0] + phenology_col * phenology_geotransform[1]
        phenology_uly = phenology_geotransform[3] + phenology_row * phenology_geotransform[5]
        # 计算物候数据栅格点对应地理范围的右下角坐标
        phenology_lrx = phenology_ulx + phenology_geotransform[1]
        phenology_lry = phenology_uly + phenology_geotransform[5]

        # 根据扰动数据的地理信息，计算物候数据栅格点对应地理范围内的扰动数据的行号和列号
        disturbance_row_start = int((phenology_uly - disturbance_geotransform[3]) / disturbance_geotransform[5])
        disturbance_row_end = int((phenology_lry - disturbance_geotransform[3]) / disturbance_geotransform[5]) + 1
        disturbance_col_start = int((phenology_ulx - disturbance_geotransform[0]) / disturbance_geotransform[1])
        disturbance_col_end = int((phenology_lrx - disturbance_geotransform[0]) / disturbance_geotransform[1]) + 1

        # 提取物候数据栅格点对应地理范围内的扰动数据
        disturbance_sub_array = disturbance_array[disturbance_row_start:disturbance_row_end,
                                disturbance_col_start:disturbance_col_end]
        # 计算满足条件的扰动数据栅格点的比例
        disturbance_ratio = np.sum((disturbance_sub_array == year)) / disturbance_sub_array.size
        disturbance_number = np.sum((disturbance_sub_array == year))
        return disturbance_number,disturbance_sub_array.size


    def generate_DEM(self,phenology_data_path,DEM_path):
        output_tif = os.path.join(os.path.split(phenology_data_path)[0],'Wu Yong/DEM.tif')
        if not os.path.exists(output_tif):
            with rasterio.open(phenology_data_path) as src:
                bounds = src.bounds
                coountry_crs = src.crs
                phenology_res = src.res  # 获取分辨率
                phenology_width = src.width  # 获取宽度
                phenology_height = src.height  # 获取高度
            country_geom = gpd.GeoDataFrame(geometry=[Polygon([(bounds[0], bounds[1]),
                       (bounds[2], bounds[1]),
                       (bounds[2], bounds[3]),
                       (bounds[0], bounds[3])])], crs=coountry_crs)
            with rasterio.open(DEM_path) as src:
                # 裁剪 DEM 数据
                out_image, out_transform = mask(src, country_geom.to_crs(src.crs).geometry, crop=True)
                out_meta = src.meta
                dem_nodata = src.nodata  # Get the nodata value of the DEM

            # 更新输出数据的元数据
            out_meta.update({
                "driver": "GTiff",
                "height": phenology_height,
                "width": phenology_width,
                "transform": out_transform,
                "crs": coountry_crs,
                "nodata": dem_nodata
            })
            # 将裁剪后的 DEM 数据写入文件
            with rasterio.open(output_tif, "w", **out_meta) as dst:
                dst.write(out_image)
            return gdal.Open(output_tif)
        else:return gdal.Open(output_tif)
    def resample_DEM(self,evi_path,dem_path,output_path):
        if os.path.exists(output_path): return gdal.Open(output_path)
        # 打开 EVI 文件以获取目标分辨率和尺寸
        with rasterio.open(evi_path) as evi_src:
            evi_transform = evi_src.transform  # EVI的仿射变换矩阵
            evi_width = evi_src.width  # EVI的宽度（像素数）
            evi_height = evi_src.height  # EVI的高度（像素数）
            evi_crs = evi_src.crs  # EVI的坐标参考系统

        # 打开 DEM 文件并重采样
        with rasterio.open(dem_path) as dem_src:
            dem_data = dem_src.read(1)  # 读取DEM的第一个波段
            dem_profile = dem_src.profile  # 获取DEM的元数据

            # 更新元数据以匹配EVI的分辨率和尺寸
            dem_profile.update({
                'transform': evi_transform,
                'width': evi_width,
                'height': evi_height,
                'crs': evi_crs
            })

            # 执行重采样
            dem_resampled = rasterio.warp.reproject(
                source=dem_data,
                destination=rasterio.io.MemoryFile().open(**dem_profile).read(1),  # 输出到内存
                src_transform=dem_src.transform,
                src_crs=dem_src.crs,
                dst_transform=evi_transform,
                dst_crs=evi_crs,
                resampling=Resampling.nearest  # 重采样方法，可选 nearest, bilinear, cubic 等
            )[0]

        # 保存重采样后的DEM
        with rasterio.open(output_path, 'w', **dem_profile) as dst:
            dst.write(dem_resampled, 1)
        return gdal.Open(output_path)
    def generate_mask(self,EVI_data_path,phenology_band,landcover_data_path,disturbance_data_path,DEM_path,output_path):
        '''
        生成掩膜(土地利用掩膜，DEM掩膜，扰动掩膜，物候基线掩膜（筛选 sos_baseline_whole 和 eos_baseline_whole 在特定时间范围内的数据）)
        :param phenology_data_path: 物候数据路径，用于确定掩膜大小和分辨率
        :param phenology_band: 物候数据波段 1为SOS， 3为EOS
        :param landcover_data_path: 土地利用数据路径
        :param disturbance_data_path: 扰动数据路径
        :param output_path: 掩膜输出路径字典
        :param drought_year: 干旱年份
        :return:
        '''
        # 新建一个吴勇文章的文件夹存放数据，同时新增txt文件，说明该文件夹的作用
        new_path = os.path.join(os.path.split(EVI_data_path)[0],'Wu Yong')
        if not os.path.exists(new_path):os.makedirs(new_path)
        if not os.path.exists(os.path.join(new_path,'description.txt')):
            with open(os.path.join(new_path,'description.txt'), "w", encoding="utf-8") as file:
                file.write("该文件夹用于存放吴勇论文的各种数据，与遗留效应的数据进行区分")
        # 如果基于土地利用的掩膜已经算过了就不算了
        if not os.path.exists(output_path['landcover_mask']): if_landmask = False
        else: if_landmask = True

        # 1. 加载数据
        evi_data = gdal.Open(EVI_data_path)
        landcover_data = gdal.Open(landcover_data_path)
        disturbance_data = gdal.Open(disturbance_data_path)
        # DEM数据
        DEM_data = self.resample_DEM(EVI_data_path,os.path.join(os.path.split(EVI_data_path)[0],'DEM.tif'),os.path.join(new_path,'DEM.tif'))
        # 2. 获取数据信息
        evi_band = evi_data.GetRasterBand(phenology_band)
        evi_array = evi_band.ReadAsArray()
        landcover_array = landcover_data.ReadAsArray()

        # 3. 获取地理信息
        evi_geotransform = evi_data.GetGeoTransform()
        landcover_geotransform = landcover_data.GetGeoTransform()
        disturbance_geotransform = disturbance_data.GetGeoTransform()

        # 5. 创建掩膜
        if not if_landmask:landcover_mask = np.zeros_like(evi_array, dtype=np.int8)
        else: landcover_mask = gdal.Open(output_path['landcover_mask']).ReadAsArray()

        combined_mask = np.zeros_like(evi_array, dtype=np.int8)
        DEM_mask = np.where(DEM_data.ReadAsArray() <= 800, 1, 0)
        # 6. 使用向量化运算创建掩膜
        for row in range(evi_array.shape[0]):
            for col in range(evi_array.shape[1]):
                if np.isnan(evi_array[row, col]):  # 处理无效值
                    # if not if_landmask:landcover_mask[row, col] = -128
                    landcover_mask[row, col] = -128
                    DEM_mask[row, col] = -128
                else:
                    # 计算土地利用数据比例
                    if not if_landmask:
                      landuse_number,all_number,max_category = self.check_landcover(row, col, landcover_array, landcover_geotransform,evi_geotransform,[24],ifeverygreen=True)
                      if landuse_number >= int(all_number*0.8):
                          landcover_mask[row, col] = 1
                      else:
                          landcover_mask[row, col] = 0
                    # 创建组合掩膜
                    if landcover_mask[row, col] == 1 and DEM_mask[row,col] == 1:
                        combined_mask[row, col] = 1
                    else:
                        combined_mask[row, col] = 0

        # 7. 保存掩膜
        # 创建一个新的GeoTIFF文件，并将掩膜数据写入文件
        driver = gdal.GetDriverByName("GTiff")
        landcover_mask_dataset = driver.Create(output_path['landcover_mask'],evi_array.shape[1], evi_array.shape[0], 1,
                                             gdal.GDT_Byte)
        landcover_mask_dataset.SetGeoTransform(evi_geotransform)
        landcover_mask_dataset.GetRasterBand(1).WriteArray(landcover_mask)
        landcover_mask_dataset = None

        DEM_mask_dataset = driver.Create(output_path['DEM_mask'], evi_array.shape[1], evi_array.shape[0], 1,
                                             gdal.GDT_Byte)
        DEM_mask_dataset.SetGeoTransform(evi_geotransform)
        DEM_mask_dataset.GetRasterBand(1).WriteArray(DEM_mask)
        DEM_mask_dataset = None

        combined_mask_with_phenologybaseline_dataset = driver.Create(output_path['combined_mask'], evi_array.shape[1], evi_array.shape[0], 1,
                                              gdal.GDT_Byte)
        combined_mask_with_phenologybaseline_dataset.SetGeoTransform(evi_geotransform)
        combined_mask_with_phenologybaseline_dataset.GetRasterBand(1).WriteArray(combined_mask)
        combined_mask_with_phenologybaseline_dataset = None

        print("掩膜文件创建完成！")

    def stack_bands(self, phenology_files,sos_band_number,eos_band_number):
       '''
       # 读取tif文件并提取指定波段（第1波段和第3波段），生成堆叠的物候序列数据
       :param phenology_files: 物候数据文件夹
       :param sos_band_number: sos波段号
       :param eos_band_number: eos波段号
       :return:
       '''
       sos_stack = []  # 存储第1波段 (SOS)
       eos_stack = []  # 存储第3波段 (EOS)
       # 遍历所有tif文件，提取第1和第3波段
       for file in phenology_files:
          with rasterio.open(file) as dataset:
             sos_band = dataset.read(sos_band_number)  # 读取第1波段
             eos_band = dataset.read(eos_band_number)  # 读取第3波段

             sos_stack.append(sos_band)  # 加入堆叠
             eos_stack.append(eos_band)  # 加入堆叠

       # 将堆叠的结果转换为numpy数组
       sos_stack = np.stack(sos_stack, axis=0)  # 形成一个shape为 (21, height, width) 的数组
       eos_stack = np.stack(eos_stack, axis=0)  # 形成一个shape为 (21, height, width) 的数组

       return sos_stack, eos_stack, dataset.meta  # 返回元数据用于写入tif文件
    def stack_bands_tonc_EVI(self, EVI_files,band_number):
       '''
       :param EVI_files: EVI文件夹
       :param band_number: 波段号
       :return:
       '''

       evi_stack = []  # 存储第1波段 (SOS)
       # 获取地理信息和数据维度
       with rasterio.open(EVI_files[5]) as dataset:
           transform = dataset.transform
           width = dataset.width
           height = dataset.height
           meta = dataset.meta
           lon_min, lat_max = transform * (0, 0)  # 左上角坐标
           lon_max, lat_min = transform * (width, height)  # 右下角坐标
           # 计算lon和lat数组
       lons = np.linspace(lon_min, lon_max, width)
       lats = np.linspace(lat_max, lat_min, height)

       # 遍历所有tif文件，提取第1和第3波段
       for file in EVI_files:
          with rasterio.open(file) as dataset:
             band_data = dataset.read(band_number)  # 读取第1波段
             evi_stack.append(band_data)  # 加入堆叠

       # 将堆叠的结果转换为numpy数组
       evi_stack = np.stack(evi_stack, axis=0)  # 形成一个shape为 (21, height, width) 的数组
       # 创建时间序列
       # 生成从2001年到2023年的年度时间序列
       start_year = 2001  # 开始年份
       end_year = 2023  # 结束年份
       # 创建时间序列，使用pandas创建从2001年到2023年的年度时间序列
       times = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', freq='YS').to_pydatetime().tolist()
       # 创建xarray Dataset
       ds = xr.Dataset(
           {
               "EVI": (["time", "lat", "lon"], evi_stack)
           },
           coords={
               "lon": lons,
               "lat": lats,
               "time": times,
           },
           attrs={
               "Conventions": "CF-1.8",
               "title": "Phenology Data",
               "summary": "Stacked EVI data with geographic coordinates",
               "spatial_ref": "EPSG:4326"
           }
       )

       # 保存为NetCDF文件
       # ds.to_netcdf(output_nc_path)
       return ds

    def generate_phenology_baseline(self,phenology_files,sos_band_number,eos_band_number):
        '''
        计算物候基线
        :param phenology_files:
        :param sos_band_number:
        :param eos_band_number:
        :return:
        '''
        output_nc_path = os.path.join(os.path.split(phenology_files[0])[0], 'phenology_baseline.nc')
        if os.path.exists(output_nc_path): return xr.open_dataset(output_nc_path)
        else:
            phenology_data = self.stack_bands_tonc(phenology_files,sos_band_number,eos_band_number)
            sos_mean = phenology_data['SOS'].mean(dim='time')
            eos_mean = phenology_data['EOS'].mean(dim='time')
            # 创建xarray Dataset
            ds = xr.Dataset(
                {
                    "SOS_baseline": (["lat", "lon"], sos_mean.data),
                    "EOS_baseline": (["lat", "lon"], eos_mean.data),
                },
                coords={
                    'lat': phenology_data['lat'],
                    'lon': phenology_data['lon']
                },
                attrs={
                    "Conventions": "CF-1.8",
                    "title": "Phenology Baseline Data",
                    "summary": "Calculate the longterm baseline of SOS and EOS",
                    "spatial_ref": "EPSG:4326"
                }
            )

            # 保存为NetCDF文件
            ds.to_netcdf(output_nc_path)
            return ds
    def tif_tonc(self,tif):
        # 获取地理信息和数据维度
        with rasterio.open(tif) as dataset:
            transform = dataset.transform
            width = dataset.width
            height = dataset.height
            meta = dataset.meta
            lon_min, lat_max = transform * (0, 0)  # 左上角坐标
            lon_max, lat_min = transform * (width, height)  # 右下角坐标
            # 计算lon和lat数组
        lons = np.linspace(lon_min, lon_max, width)
        lats = np.linspace(lat_max, lat_min, height)

        with rasterio.open(tif) as dataset:
            data = dataset.read(1)  # 读取第1波段

        # 替换NoData值为NaN
        data_ = np.where((data == 128) | (data == 0), np.nan, data)
        # 创建xarray Dataset
        ds = xr.Dataset(
            {
                "data": ([ "lat", "lon"], data_),
            },
            coords={
                "lon": lons,
                "lat": lats,
            },
            attrs={
                "Conventions": "CF-1.8",
                "title": "Phenology Data",
                "summary": "Stacked SOS and EOS data with geographic coordinates",
            }
        )
        # 添加CRS信息，明确指定为WGS84投影
        ds['crs'] = xr.DataArray(np.array([0]), attrs={
            "grid_mapping_name": "latitude_longitude",
            "longitude_of_prime_meridian": 0.0,
            "semi_major_axis": 6378137.0,
            "inverse_flattening": 298.257223563,
            "spatial_ref": "EPSG:4326",  # 添加EPSG代码以明确WGS84投影
            "crs_wkt": (
                "GEOGCS[\"WGS 84\","
                "DATUM[\"WGS_1984\","
                "SPHEROID[\"WGS 84\",6378137,298.257223563]],"
                "PRIMEM[\"Greenwich\",0],"
                "UNIT[\"degree\",0.0174532925199433]]"
            )
        })
        return ds
    def get_SIF_current_region(self,sif_path,evi_path):
        evi_data = rioxarray.open_rasterio(evi_path)
        # 获取物候数据栅格中心点的经纬度坐标
        lons = evi_data.x.values
        lats = evi_data.y.values
        # 获取物候数据的投影和仿射变换信息
        crs = evi_data.rio.crs
        transform = evi_data.rio.transform()
        # 初始化年份列表（2000-2023）
        years = list(range(2001, 2017))
        # 读取 SPEI 数据
        sif_data = xr.open_dataset(sif_path)
        # 初始化一个空的 xarray.Dataset，用于存储每年的累积值
        ds = xr.Dataset(
            {
                "SIF": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons))))
            },
            coords={
                "year": years,
                "lat": lats,
                "lon": lons,
            }
        )

        # 将物候数据的 CRS 和 transform 信息写入新建的 Dataset
        ds.rio.set_crs(crs)
        ds.rio.write_transform(transform)
        # 提取对应时间的 SPEI 数据
        for i,year_ in enumerate(years):
            sif = sif_data['annual_SIF'].sel(year=year_)
            ds['SIF'][i] = sif.reindex(lat=lats, lon=lons, method="nearest")
        # 保存汇总后的数据为NetCDF文件
        # ds.to_netcdf(os.path.join(os.path.split(evi_path)[0],"Wu Yong/SIF.nc"))
        return ds
    def get_SPEI_current_region(self,spei_path,evi_path,spei_scale):
        evi_data = rioxarray.open_rasterio(evi_path)
        # 获取物候数据栅格中心点的经纬度坐标
        lons = evi_data.x.values
        lats = evi_data.y.values
        # 获取物候数据的投影和仿射变换信息
        crs = evi_data.rio.crs
        transform = evi_data.rio.transform()
        # 初始化年份列表（2000-2023）
        years = list(range(2000, 2024))
        # 读取 SPEI 数据
        spei_data = xr.open_dataset(spei_path)
        if spei_scale == '03':
            # 初始化一个空的 xarray.Dataset，用于存储每年的累积值
            ds = xr.Dataset(
                {
                    "annual_spei": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                    "spring_spei": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                    "summer_spei": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                    "autumn_spei": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                },
                coords={
                    "year": years,
                    "lat": lats,
                    "lon": lons,
                }
            )

            # 将物候数据的 CRS 和 transform 信息写入新建的 Dataset
            ds.rio.set_crs(crs)
            ds.rio.write_transform(transform)
            # 提取对应时间的 SPEI 数据
            for i,year in enumerate(years):
                annual_spei = spei_data['spei'].sel(time=f'{year}').mean(dim='time')  # 计算年平均 SPEI
                spring_spei = spei_data['spei'].sel(time=slice(f'{year}-03-01', f'{year}-05-31')).mean(dim='time')
                summer_spei = spei_data['spei'].sel(time=slice(f'{year}-06-01', f'{year}-08-31')).mean(dim='time')
                autumn_spei = spei_data['spei'].sel(time=slice(f'{year}-09-01', f'{year}-11-30')).mean(dim='time')
                ds['annual_spei'][i] = annual_spei.reindex(lat=lats, lon=lons, method="nearest")
                ds['spring_spei'][i] = spring_spei.reindex(lat=lats, lon=lons, method="nearest")
                ds['summer_spei'][i] = summer_spei.reindex(lat=lats, lon=lons, method="nearest")
                ds['autumn_spei'][i] = autumn_spei.reindex(lat=lats, lon=lons, method="nearest")
            # 保存汇总后的数据为NetCDF文件
            # ds.to_netcdf(os.path.join(os.path.split(evi_path)[0],"Wu Yong/spei{}_seasonal_sums_2000_2023.nc".format(spei_scale)))
        else:
            # 初始化一个空的 xarray.Dataset，用于存储每年的累积值
            ds = xr.Dataset(
                {
                    "annual_spei": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                    "summer_half_spei": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                    "winter_half_spei": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                },
                coords={
                    "year": years,
                    "lat": lats,
                    "lon": lons,
                }
            )

            # 将物候数据的 CRS 和 transform 信息写入新建的 Dataset
            ds.rio.set_crs(crs)
            ds.rio.write_transform(transform)
            # 提取对应时间的 SPEI 数据
            for i, year in enumerate(years):

                annual_spei = spei_data['spei'].sel(time=f'{year}').mean(dim='time')  # 计算年平均 SPEI
                summer_half_spei = spei_data['spei'].sel(time=slice(f'{year}-04-01', f'{year}-09-30')).mean(dim='time')
                winter_half_spei = spei_data['spei'].sel(time=slice(f'{year}-10-01', f'{year+1}-03-31')).mean(dim='time')
                ds['annual_spei'][i] = annual_spei.reindex(lat=lats, lon=lons, method="nearest")
                ds['summer_half_spei'][i] = summer_half_spei.reindex(lat=lats, lon=lons, method="nearest")
                ds['winter_half_spei'][i] = winter_half_spei.reindex(lat=lats, lon=lons, method="nearest")


            # 保存汇总后的数据为NetCDF文件
            # ds.to_netcdf(os.path.join(os.path.split(evi_path)[0],
            #                           "Wu Yong/spei{}_seasonal_sums_2000_2023.nc".format(spei_scale)))
        return ds

    def get_Temperature_average_current_region(self,tm_path,evi_path):

        evi_data = rioxarray.open_rasterio(evi_path)
        lons = evi_data.x.values
        lats = evi_data.y.values
        crs = evi_data.rio.crs
        transform = evi_data.rio.transform()

        # 初始化年份范围和分块处理的逻辑
        all_years = list(range(2000, 2024))  # 总年份列表
        chunk_size = 6  # 每次处理的年数
        chunks = [all_years[i:i + chunk_size] for i in range(0, len(all_years), chunk_size)]  # 分块

        # 创建一个空的 Dataset，用于存放最终结果
        ds_final = None

        # 打开温度数据
        tm_data = xr.open_dataset(tm_path)
        tm_c = tm_data['t2m'] - 273.15  # 转换为摄氏度

        # 按块处理数据
        for chunk_number,years in enumerate(chunks):
            # 初始化当前块的 Dataset
            ds_chunk = xr.Dataset(
                {
                    "annual_avg": (
                    ("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                    "spring_avg": (
                    ("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                    "summer_avg": (
                    ("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                    "autumn_avg": (
                    ("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                    "summerhalf_avg": (
                    ("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                    "winterhalf_avg": (
                    ("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                    "chilling_avg": (
                    ("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                },
                coords={
                    "year": years,
                    "lat": lats,
                    "lon": lons,
                }
            )

            # 填充当前块的数据
            for i, year in enumerate(years):
                # 计算各个时间段的平均值
                annual_avg = tm_c.sel(time=slice(f'{year}-01-01', f'{year}-12-31')).mean(dim='time')
                spring_avg = tm_c.sel(time=slice(f'{year}-03-01', f'{year}-05-31')).mean(dim='time')
                summer_avg = tm_c.sel(time=slice(f'{year}-06-01', f'{year}-08-31')).mean(dim='time')
                autumn_avg = tm_c.sel(time=slice(f'{year}-09-01', f'{year}-11-30')).mean(dim='time')
                summerhalf_avg = tm_c.sel(time=slice(f'{year}-04-01', f'{year}-09-30')).mean(dim='time')
                winterhalf_avg = tm_c.sel(time=slice(f'{year}-10-01', f'{year + 1}-03-30')).mean(dim='time')
                chilling_avg = tm_c.sel(time=slice(f'{year}-11-01', f'{year + 1}-02-28')).mean(dim='time')

                # 插值到 EVI 数据的经纬度网格
                ds_chunk['annual_avg'][i] = annual_avg.sel(latitude=lats, longitude=lons, method="nearest")
                ds_chunk['spring_avg'][i] = spring_avg.sel(latitude=lats, longitude=lons, method="nearest")
                ds_chunk['summer_avg'][i] = summer_avg.sel(latitude=lats, longitude=lons, method="nearest")
                ds_chunk['autumn_avg'][i] = autumn_avg.sel(latitude=lats, longitude=lons, method="nearest")

                ds_chunk['summerhalf_avg'][i] = summerhalf_avg.sel(latitude=lats, longitude=lons, method="nearest")
                ds_chunk['winterhalf_avg'][i] = winterhalf_avg.sel(latitude=lats, longitude=lons, method="nearest")
                ds_chunk['chilling_avg'][i] = chilling_avg.sel(latitude=lats, longitude=lons, method="nearest")
            # ds_chunk.to_netcdf(os.path.join(os.path.split(evi_path)[0], f"Wu Yong/tm_seasonal_average_2000_2023_{chunk_number}.nc"))
            # 合并当前块到最终的 Dataset
            if ds_final is None:
                ds_final = ds_chunk  # 初始化
            else:
                ds_final = xr.concat([ds_final, ds_chunk], dim="year")

        # 写入 CRS 和 transform 信息
        ds_final.rio.set_crs(crs)
        ds_final.rio.write_transform(transform)
        # 保存汇总后的数据为NetCDF文件
        # ds_final.to_netcdf(os.path.join(os.path.split(evi_path)[0],"Wu Yong/tm_seasonal_average_2000_2023.nc"))
        ds = ds_final.copy()
        return ds
    def get_Totalprecipiation_average_current_region(self,tp_path,evi_path,tp_path_22_23):

        # if os.path.exists(os.path.join(os.path.split(evi_path)[0],"Wu Yong/tp_seasonal_average_2000_2023.nc")):
        #     ds = xr.open_dataset(os.path.join(os.path.split(evi_path)[0],"Wu Yong/tp_seasonal_average_2000_2023.nc"))
        # else:
        evi_data = rioxarray.open_rasterio(evi_path)
        # 获取物候数据栅格中心点的经纬度坐标
        lons = evi_data.x.values
        lats = evi_data.y.values
        # 获取物候数据的投影和仿射变换信息
        crs = evi_data.rio.crs
        transform = evi_data.rio.transform()
        # 初始化年份列表（2000-2023）
        years = list(range(2000, 2024))

        # 初始化一个空的 xarray.Dataset，用于存储每年的累积值
        ds = xr.Dataset(
            {
                "annual_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                "spring_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                "summer_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                "autumn_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                "winter_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                "summerhalf_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                "winterhalf_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32))
            },
            coords={
                "year": years,
                "lat": lats,
                "lon": lons,
            }
        )

        # 将物候数据的 CRS 和 transform 信息写入新建的 Dataset
        ds.rio.set_crs(crs)
        ds.rio.write_transform(transform)
        tp_data = xr.open_dataset(tp_path)
        tp_c = tp_data['tp']
        for i,year in enumerate(years):
            if year>2021:tp_c = xr.open_dataset(tp_path_22_23)['tp'].rename({'valid_time': 'time'})
            # 计算全年平均值
            annual_avg = tp_c.sel(time=slice('{}-01-01'.format(year), '{}-12-31'.format(year))).mean(dim='time')
            # 计算春季平均值 (3月1日到5月31日)
            spring_avg = tp_c.sel(time=slice('{}-03-01'.format(year), '{}-05-31'.format(year))).mean(dim='time')
            # 计算夏季平均值 (6月1日到8月31日)
            summer_avg = tp_c.sel(time=slice('{}-06-01'.format(year), '{}-08-31'.format(year))).mean(dim='time')
            # 计算秋季平均值 (9月1日到11月30日)
            autumn_avg = tp_c.sel(time=slice('{}-09-01'.format(year), '{}-11-30'.format(year))).mean(dim='time')
            # 计算冬季平均值 (12月1日到2月29日)
            winter_avg = tp_c.sel(time=slice('{}-12-01'.format(year), '{}-02-28'.format(year+1))).mean(dim='time')
            # 计算夏半年平均值 (4月1日到9月30日)
            summerhalf_avg = tp_c.sel(time=slice('{}-04-01'.format(year), '{}-09-30'.format(year))).mean(dim='time')
            # 计算冬半年平均值 (10月1日到明年的3月30日)
            winterhalf_avg = tp_c.sel(time=slice('{}-10-01'.format(year), '{}-03-30'.format(year+1))).mean(dim='time')
            # 使用坐标从 GDD 数据中提取对应值
            ds['annual_avg'][i] = annual_avg.sel(latitude=lats, longitude=lons, method="nearest")
            ds['spring_avg'][i] = spring_avg.sel(latitude=lats, longitude=lons, method="nearest")
            ds['summer_avg'][i] = summer_avg.sel(latitude=lats, longitude=lons, method="nearest")
            ds['autumn_avg'][i] = autumn_avg.sel(latitude=lats, longitude=lons, method="nearest")
            ds['winter_avg'][i] = winter_avg.sel(latitude=lats, longitude=lons, method="nearest")
            ds['summerhalf_avg'][i] = summerhalf_avg.sel(latitude=lats, longitude=lons, method="nearest")
            ds['winterhalf_avg'][i] = winterhalf_avg.sel(latitude=lats, longitude=lons, method="nearest")
        # 保存汇总后的数据为NetCDF文件
        name = os.path.split(evi_path)[-1].replace('.tif','')
        path = os.path.join(os.path.split(os.path.split(evi_path)[0])[0],f'tp_seasonal_average_2000_2023_{name}.nc')
        # ds.to_netcdf(path)
        return ds

    def get_VPD_average_current_region(self,VPD_path,evi_path):

        evi_data = rioxarray.open_rasterio(evi_path)
        # 获取物候数据栅格中心点的经纬度坐标
        lons = evi_data.x.values
        lats = evi_data.y.values
        # 获取物候数据的投影和仿射变换信息
        crs = evi_data.rio.crs
        transform = evi_data.rio.transform()
        # 初始化年份列表（2000-2023）
        years = list(range(2000, 2024))
        # 初始化一个空的 xarray.Dataset，用于存储每年的累积值
        ds = xr.Dataset(
            {
                "annual_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                "spring_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                "summer_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                "autumn_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                "winter_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32))
            },
            coords={
                "year": years,
                "lat": lats,
                "lon": lons,
            }
        )

        # 将物候数据的 CRS 和 transform 信息写入新建的 Dataset
        ds.rio.set_crs(crs)
        ds.rio.write_transform(transform)
        VPD_data = xr.open_dataset(VPD_path)
        VPD_c = VPD_data['VPD']
        for i,year in enumerate(years):
            # 计算全年平均值
            annual_avg = VPD_c.sel(time=slice('{}-01-01'.format(year), '{}-12-31'.format(year))).mean(dim='time')
            # 计算春季平均值 (3月1日到5月31日)
            spring_avg = VPD_c.sel(time=slice('{}-03-01'.format(year), '{}-05-31'.format(year))).mean(dim='time')
            # 计算夏季平均值 (6月1日到8月31日)
            summer_avg = VPD_c.sel(time=slice('{}-06-01'.format(year), '{}-08-31'.format(year))).mean(dim='time')
            # 计算秋季平均值 (9月1日到11月30日)
            autumn_avg = VPD_c.sel(time=slice('{}-09-01'.format(year), '{}-11-30'.format(year))).mean(dim='time')
            # 计算冬季平均值 (12月1日到明年2月30日)
            winter_avg = VPD_c.sel(time=slice('{}-12-01'.format(year), '{}-02-28'.format(year+1))).mean(dim='time')
            # 使用坐标从 GDD 数据中提取对应值
            ds['annual_avg'][i] = annual_avg.reindex(lat=lats, lon=lons, method="nearest")
            ds['spring_avg'][i] = spring_avg.reindex(lat=lats, lon=lons, method="nearest")
            ds['summer_avg'][i] = summer_avg.reindex(lat=lats, lon=lons, method="nearest")
            ds['autumn_avg'][i] = autumn_avg.reindex(lat=lats, lon=lons, method="nearest")
            ds['winter_avg'][i] = winter_avg.reindex(lat=lats, lon=lons, method="nearest")

        # 保存汇总后的数据为NetCDF文件
        # ds.to_netcdf(os.path.join(os.path.split(evi_path)[0],"Wu Yong/VPD_seasonal_average_2000_2023.nc"))
        return ds

    def get_otherfactor_average_current_region(self, otherfactor_path, phenology_path,otherfactor_name):

        if os.path.exists(os.path.join(os.path.split(phenology_path)[0], f"{otherfactor_name}_seasonal_average_2000_2023.nc")):
            ds = xr.open_dataset(os.path.join(os.path.split(phenology_path)[0], f"{otherfactor_name}_seasonal_average_2000_2023.nc"))
        else:

            phenology_data = rioxarray.open_rasterio(phenology_path)
            # 获取物候数据栅格中心点的经纬度坐标
            lons = phenology_data.x.values
            lats = phenology_data.y.values
            # 获取物候数据的投影和仿射变换信息
            crs = phenology_data.rio.crs
            transform = phenology_data.rio.transform()
            # 初始化年份列表（2000-2023）
            years = list(range(2000, 2024))

            # 初始化一个空的 xarray.Dataset，用于存储每年的累积值
            ds = xr.Dataset(
                {
                    "annual_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)))),
                    "spring_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)))),
                    "summer_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)))),
                    "autumn_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)))),
                    "summerhalf_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)))),
                    "winterhalf_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)))),
                },
                coords={
                    "year": years,
                    "lat": lats,
                    "lon": lons,
                }
            )

            # 将物候数据的 CRS 和 transform 信息写入新建的 Dataset
            ds.rio.set_crs(crs)
            ds.rio.write_transform(transform)
            ofactor_data = xr.open_dataset(otherfactor_path)
            ofactor_c = ofactor_data[otherfactor_name]

            for i, year in enumerate(years):
                # 计算全年平均值
                annual_avg = ofactor_c.sel(time=slice('{}-01-01'.format(year), '{}-12-31'.format(year))).mean(dim='time')

                # 计算春季平均值 (3月1日到5月31日)
                spring_avg = ofactor_c.sel(time=slice('{}-03-01'.format(year), '{}-05-31'.format(year))).mean(dim='time')

                # 计算夏季平均值 (6月1日到8月31日)
                summer_avg = ofactor_c.sel(time=slice('{}-06-01'.format(year), '{}-08-31'.format(year))).mean(dim='time')

                # 计算秋季平均值 (9月1日到11月30日)
                autumn_avg = ofactor_c.sel(time=slice('{}-09-01'.format(year), '{}-11-30'.format(year))).mean(dim='time')
                # 计算夏半年平均值 (4月1日到9月30日)
                summerhalf_avg = ofactor_c.sel(time=slice('{}-04-01'.format(year), '{}-09-30'.format(year))).mean(
                    dim='time')
                # 计算冬半年平均值 (10月1日到明年的3月30日)
                winterhalf_avg = ofactor_c.sel(time=slice('{}-10-01'.format(year), '{}-03-30'.format(year + 1))).mean(
                    dim='time')

                # 使用坐标从 GDD 数据中提取对应值
                ds['annual_avg'][i] = annual_avg.reindex(latitude=lats, longitude=lons, method="nearest")
                ds['spring_avg'][i] = spring_avg.reindex(latitude=lats, longitude=lons, method="nearest")
                ds['summer_avg'][i] = summer_avg.reindex(latitude=lats, longitude=lons, method="nearest")
                ds['autumn_avg'][i] = autumn_avg.reindex(latitude=lats, longitude=lons, method="nearest")

                ds['summerhalf_avg'][i] = summerhalf_avg.reindex(latitude=lats, longitude=lons, method="nearest")
                ds['winterhalf_avg'][i] = winterhalf_avg.reindex(latitude=lats, longitude=lons, method="nearest")
            # 保存汇总后的数据为NetCDF文件
            ds.to_netcdf(os.path.join(os.path.split(phenology_path)[0], f"{otherfactor_name}_seasonal_average_2000_2023.nc"))
        return ds

    def summarize_raster_basedon_mask(self,phenology_tifs,mask_tifs,gdd_paths,SPEI_paths,temperature_path):
        '''
        基于掩膜数据的栅格综合成表格，步骤如下：
        1. 堆叠SOS,EOS
        2.生成对应区域的gdd，spei，temperature，这里没有使用任何改变分辨率的手段，直接对应原数据，比如（x,y）对应位置的SPEI,GDD,TM,如果(x+1,y+1)对应的数据和（x，y）一样，那就一样
        3.遍历mask把数据写入csv
        由于数据分辨率不同，这会造成很多点的物候数据不一样，但是对应的SPEI,GDD,TM可能是一样的，因此在后面的一个方法里面添加了空间聚合，具体见方法summarize_raster_by_mask_aggregation
        :param phenology_tifs:
        :param mask_tifs:
        :param gdd_paths:
        :param SPEI_paths:
        :param temperature_path:
        :return:
        '''
        # 设置波段索引
        band_sos_number = 1  # SOS对应的波段
        band_eos_number = 3  # EOS对应的波段
        sos_stack,eos_stack,phenology_tif_geography = self.stack_bands(phenology_tifs,band_sos_number,band_eos_number)
        # 获取GDD
        gdd = self.get_GDD_current_region(gdd_paths, phenology_tifs[0])
        # # 获取SPEI
        spei_03 = self.get_SPEI_current_region(SPEI_paths[0],phenology_tifs[0],'03')
        spei_06 = self.get_SPEI_current_region(SPEI_paths[1], phenology_tifs[0],'06')
        temperature = self.get_Temperature_average_current_region(temperature_path, phenology_tifs[0])

        # 获取土地利用类别做准备
        phenology_data = gdal.Open(phenology_tifs[0])
        landcover_data = gdal.Open(phenology_tifs[0].replace('reprojection_phenology2001','land_cover'))
        landcover_array = landcover_data.ReadAsArray()
        # 3. 获取地理信息
        phenology_geotransform = phenology_data.GetGeoTransform()
        landcover_geotransform = landcover_data.GetGeoTransform()
        for mask_tif in mask_tifs:
            drouht_year =  re.search(r'_(\d{4})\.tif$', mask_tif).group(1)
            with rasterio.open(mask_tif) as dataset:
                mask = dataset.read(1)
                mask = mask==1
                rows, cols = np.where(mask)
                # 创建一个空的列表来存储所有行的数据
                data_chunk = []
                # 2. Set a chunk size for writing to CSV
                chunk_size = 1000

                # 定义年份范围
                years = list(range(2000, 2024))  # 2000年到2023年

                for i, (row, col) in enumerate(zip(rows, cols)):
                    sos_values = sos_stack[:, row, col]  # SOS 时序数据
                    eos_values = eos_stack[:, row, col]  # EOS 时序数据

                    landuse_number, all_number, max_category = self.check_landcover(row, col, landcover_array,
                                                                                    landcover_geotransform,
                                                                                    phenology_geotransform)
                    gdd_fixed_sum = gdd['fixed_sum'].data[:,row,col]

                    tm_annual = temperature['annual_avg'].data[:, row, col]
                    tm_spring = temperature['spring_avg'].data[:, row, col]
                    tm_summer = temperature['summer_avg'].data[:, row, col]
                    tm_autumn = temperature['autumn_avg'].data[:, row, col]

                    spei_03_annual_spei = spei_03['annual_spei'].data[:,row,col]
                    spei_03_spring_spei = spei_03['spring_spei'].data[:, row, col]
                    spei_03_summer_spei = spei_03['summer_spei'].data[:, row, col]
                    spei_03_autumn_spei = spei_03['autumn_spei'].data[:, row, col]

                    spei_06_annual_spei = spei_06['annual_spei'].data[:,row,col]
                    spei_06_summer_half_spei = spei_06['summer_half_spei'].data[:, row, col]
                    spei_06_winter_half_spei = spei_06['winter_half_spei'].data[:, row, col]
                    # 将数据添加到列表中
                    data_chunk.append({
                        'row': row,
                        'col': col,
                        'max_category': max_category,
                        **{f'sos_{year}': sos_values[i] for i, year in enumerate(list(range(2001, 2024)) )},  # SOS 数据
                        **{f'eos_{year}': eos_values[i] for i, year in enumerate(list(range(2001, 2024)) )},  # EOS 数据
                        **{f'gdd_fixed_sum_{year}': gdd_fixed_sum[i] for i, year in enumerate(years)},
                        **{f'tm_annual_{year}': tm_annual[i] for i, year in enumerate(years)},
                        **{f'tm_spring_{year}': tm_spring[i] for i, year in enumerate(years)},
                        **{f'tm_summer_{year}': tm_summer[i] for i, year in enumerate(years)},
                        **{f'tm_autumn_{year}': tm_autumn[i] for i, year in enumerate(years)},

                        **{f'spei_03_annual_spei_{year}': spei_03_annual_spei[i] for i, year in enumerate(years)},
                        **{f'spei_03_spring_spei_{year}': spei_03_spring_spei[i] for i, year in enumerate(years)},
                        **{f'spei_03_summer_spei_{year}': spei_03_summer_spei[i] for i, year in enumerate(years)},
                        **{f'spei_03_autumn_spei_{year}': spei_03_autumn_spei[i] for i, year in enumerate(years)},
                        **{f'spei_06_annual_spei_{year}': spei_06_annual_spei[i] for i, year in enumerate(years)},
                        **{f'spei_06_summer_half_spei_{year}': spei_06_summer_half_spei[i] for i, year in enumerate(years)},
                        **{f'spei_06_winter_half_spei_{year}': spei_06_winter_half_spei[i] for i, year in enumerate(years)},
                    })
                    if (i + 1) % chunk_size == 0 or i == len(rows) - 1:
                        # Convert the chunk to a DataFrame
                        df_chunk = pd.DataFrame(data_chunk)

                        # Write the chunk to the CSV file
                        csv_path = os.path.join(os.path.split(mask_tif)[0],
                                                f'{os.path.split(mask_tif)[-1].replace(".tif", "")}_inform_sum.csv')
                        # Append to the CSV if it already exists, otherwise create it
                        if os.path.isfile(csv_path):
                            df_chunk.to_csv(csv_path, mode='a', header=False, index=False)
                        else:
                            df_chunk.to_csv(csv_path, index=False)

                        # Clear the chunk for the next batch
                        data_chunk = []

                # # 将数据转换为DataFrame
                # df = pd.DataFrame(data)
                #
                # # 将DataFrame写入CSV文件
                # df.to_csv(os.path.join(os.path.split(mask_tif)[0],'{}_inform_sum.csv'.format(os.path.split(mask_tif)[-1].replace('.tif',''))), index=False)

    def spatial_aggregation(self,high_resolution_data,low_resolution_data,aggregation_method):

        # 计算聚合因子
        hr_data_lon = high_resolution_data['lon'].diff('lon').mean().values
        hr_data_lat = high_resolution_data['lat'].diff('lat').mean().values
        lr_data_lon = low_resolution_data['lon'].diff('lon').mean().values
        lr_data_lat = low_resolution_data['lat'].diff('lat').mean().values
        lat_factor = int(abs(lr_data_lat)/abs(hr_data_lat))
        lon_factor = int(abs(lr_data_lon)/abs(hr_data_lon))
        if aggregation_method == 'mean':
            data_aggregation = high_resolution_data.coarsen(lat=lat_factor, lon=lon_factor, boundary="trim").mean()
        if aggregation_method == 'sum':
            data_aggregation = high_resolution_data.coarsen(lat=lat_factor, lon=lon_factor, boundary="trim").sum()
            data_aggregation = {'number':data_aggregation,'ratio':data_aggregation/(lat_factor*lon_factor)}
        return data_aggregation
    def summarize_raster_by_mask_aggregation(self,evi_tifs,mask_tifs,SPEI_paths,temperature_path,VPD_path,SIF_path,TP_path,TP_path_2224,df_save_part):
        # 设置波段索引
        band_sos_number = 1  # SOS对应的波段
        band_eos_number = 3  # EOS对应的波段
        # 新建一个吴勇文章的文件夹存放数据，同时新增txt文件，说明该文件夹的作用
        new_path = os.path.join(os.path.split(evi_tifs[5])[0],'Wu Yong')
        if not os.path.exists(new_path):os.makedirs(new_path)
        if not os.path.exists(os.path.join(new_path,'description.txt')):
            with open(os.path.join(new_path,'description.txt'), "w", encoding="utf-8") as file:
                file.write("该文件夹用于存放吴勇论文的各种数据，与遗留效应的数据进行区分")
        # sos_stack,eos_stack,phenology_tif_geography = self.stack_bands(phenology_tifs,band_sos_number,band_eos_number)
        EVI_data = self.stack_bands_tonc_EVI(evi_tifs, 1)
        # 获取GDD,SPEI,温度
        spei_03 = self.get_SPEI_current_region(SPEI_paths[0], evi_tifs[0], '03')
        spei_06 = self.get_SPEI_current_region(SPEI_paths[1], evi_tifs[0], '06')
        temperature = self.get_Temperature_average_current_region(temperature_path, evi_tifs[0])
        VPD = self.get_VPD_average_current_region(VPD_path, evi_tifs[0])
        SIF = self.get_SIF_current_region(SIF_path,evi_tifs[0])
        TP = self.get_Totalprecipiation_average_current_region(TP_path,evi_tifs[0],TP_path_2224)
        mask_tif = mask_tifs[0]

        # 看聚合后的栅格中，每个栅格有多少原来的点（经过掩膜后过滤的点）参与了计算,同时绘制直方图
        mask_nc = self.tif_tonc(mask_tif)
        mask_num_nc = self.spatial_aggregation(mask_nc, xr.open_dataset(VPD_path), 'sum')
        data_mask_num = mask_num_nc['number']['data'].values.flatten()
        df_mask_num = pd.DataFrame({'grid_id': range(len(data_mask_num)), 'data': data_mask_num})
        # 直接定义阈值为1，后续采用加权回归模型
        mask_num_threshold = 1.0
        mask_array = mask_nc['data'].data
        mask_array_evi = np.repeat(np.expand_dims(mask_array.data,0),23,axis=0)
        mask_array_SIF = np.repeat(np.expand_dims(mask_array.data, 0), 16, axis=0)
        EVI_sa = self.spatial_aggregation(
            EVI_data.where(mask_array_evi == 1),
            xr.open_dataset(VPD_path), 'mean')
        del EVI_data
        SIF_sa = self.spatial_aggregation(
            SIF.where(mask_array_SIF == 1),
            xr.open_dataset(VPD_path), 'mean')
        del SIF
        mask_array_others = np.repeat(np.expand_dims(mask_array.data, 0), 24, axis=0)     #这里面除了物候数据，其他数据年数都是2000年到2023年，因此统一用一个expandmask的形状就好了
        spei_03_sa = self.spatial_aggregation(spei_03.where(mask_array_others == 1),xr.open_dataset(VPD_path), 'mean')
        del spei_03
        spei_06_sa = self.spatial_aggregation(spei_06.where(mask_array_others == 1),xr.open_dataset(VPD_path), 'mean')
        del spei_06
        temperature_sa = self.spatial_aggregation(temperature.where(mask_array_others == 1),xr.open_dataset(VPD_path), 'mean')
        del temperature
        VPD_sa = self.spatial_aggregation(VPD.where(mask_array_others == 1), xr.open_dataset(VPD_path), 'mean')
        del VPD
        TP_sa = self.spatial_aggregation(TP.where(mask_array_others == 1), xr.open_dataset(VPD_path), 'mean')
        del TP
        # TP_sa.to_netcdf(os.path.join(os.path.split(mask_tif)[0], f'TP_sa_{df_save_part}.nc'))
        # 如果该点的有效数据数量小于阈值，就放弃
        rows, cols = np.where(mask_num_nc['number']['data']>mask_num_threshold)
        if len(rows) == 0 and len(cols) == 0: return None

        valid_data_mask = np.zeros(mask_num_nc['number']['data'].shape, dtype=np.uint8)
        # 根据有效数据的位置标记 1
        valid_data_mask[rows, cols] = 1
        # 定义输出tif文件路径
        output_tif = os.path.join(os.path.split(mask_tif)[0], f'valid_data_mask.tif')
        mask_sa_tif = mask_num_nc['number']['data'].copy()
        mask_sa_tif = mask_sa_tif.rename({'lon':'x','lat':'y'})
        mask_sa_tif.data = valid_data_mask
        mask_sa_tif.rio.write_crs("EPSG:4326", inplace=True).rio.to_raster(output_tif.replace('valid_data_mask.tif',f'valid_data_mask_{df_save_part}.tif'))
        mask_num_nc['number']['data'].copy().rename({'lon':'x','lat':'y'}).rio.write_crs("EPSG:4326", inplace=True).rio.to_raster(os.path.join(os.path.split(mask_tif)[0], f'valid_data_mask_number_{df_save_part}.tif'))
        mask_num_nc['ratio']['data'].copy().rename({'lon': 'x', 'lat': 'y'}).rio.write_crs("EPSG:4326",inplace=True).rio.to_raster(os.path.join(os.path.split(mask_tif)[0], f'valid_data_mask_ratio_{df_save_part}.tif'))

        '''
        写入csv
        '''
        # 创建一个空的列表来存储所有行的数据
        data= []
        # 定义年份范围
        years = list(range(2000, 2024))  # 2000年到2023年

        for i, (row, col) in tqdm(enumerate(zip(rows, cols))):

            evi_values = EVI_sa['EVI'][:, row, col].data  # SOS 时序数据

            sif_values = SIF_sa['SIF'][:, row, col].data

            tp_annual = TP_sa['annual_avg'].data[:, row, col]
            tp_spring = TP_sa['spring_avg'].data[:, row, col]
            tp_summer = TP_sa['summer_avg'].data[:, row, col]
            tp_autumn = TP_sa['autumn_avg'].data[:, row, col]
            tp_winter = TP_sa['winter_avg'].data[:, row, col]
            tp_summerhalf = TP_sa['summerhalf_avg'].data[:, row, col]
            tp_winterhalf = TP_sa['winterhalf_avg'].data[:, row, col]

            VPD_annual = VPD_sa['annual_avg'].data[:, row, col]
            VPD_spring = VPD_sa['spring_avg'].data[:, row, col]
            VPD_summer = VPD_sa['summer_avg'].data[:, row, col]
            VPD_autumn = VPD_sa['autumn_avg'].data[:, row, col]
            VPD_winter = VPD_sa['winter_avg'].data[:, row, col]

            tm_annual = temperature_sa['annual_avg'].data[:, row, col]
            tm_spring = temperature_sa['spring_avg'].data[:, row, col]
            tm_summer = temperature_sa['summer_avg'].data[:, row, col]
            tm_autumn = temperature_sa['autumn_avg'].data[:, row, col]
            tm_summerhalf = temperature_sa['summerhalf_avg'].data[:, row, col]
            tm_winterhalf = temperature_sa['winterhalf_avg'].data[:, row, col]
            tm_chilling = temperature_sa['chilling_avg'].data[:, row, col]


            spei_03_annual_spei = spei_03_sa['annual_spei'].data[:,row,col]
            spei_03_spring_spei = spei_03_sa['spring_spei'].data[:, row, col]
            spei_03_summer_spei = spei_03_sa['summer_spei'].data[:, row, col]
            spei_03_autumn_spei = spei_03_sa['autumn_spei'].data[:, row, col]

            spei_06_annual_spei = spei_06_sa['annual_spei'].data[:,row,col]
            spei_06_summer_half_spei = spei_06_sa['summer_half_spei'].data[:, row, col]
            spei_06_winter_half_spei = spei_06_sa['winter_half_spei'].data[:, row, col]
            # 将数据添加到列表中
            data.append({
                'row': row,
                'col': col,
                'weights':mask_num_nc['ratio']['data'].data[row, col],
                **{f'evi_{year}': evi_values[i] for i, year in enumerate(list(range(2001, 2024)) )},
                **{f'sif_{year}': sif_values[i] for i, year in enumerate(list(range(2001, 2017)))},

                **{f'tp_annual_{year}': tp_annual[i] for i, year in enumerate(years)},
                **{f'tp_spring_{year}': tp_spring[i] for i, year in enumerate(years)},
                **{f'tp_summer_{year}': tp_summer[i] for i, year in enumerate(years)},
                **{f'tp_autumn_{year}': tp_autumn[i] for i, year in enumerate(years)},
                **{f'tp_winter_{year}': tp_winter[i] for i, year in enumerate(years)},
                **{f'tp_summerhalf_{year}': tp_summerhalf[i] for i, year in enumerate(years)},
                **{f'tp_winterhalf_{year}': tp_winterhalf[i] for i, year in enumerate(years)},

                **{f'vpd_annual_{year}': VPD_annual[i] for i, year in enumerate(years)},
                **{f'vpd_spring_{year}': VPD_spring[i] for i, year in enumerate(years)},
                **{f'vpd_summer_{year}': VPD_summer[i] for i, year in enumerate(years)},
                **{f'vpd_autumn_{year}': VPD_autumn[i] for i, year in enumerate(years)},
                **{f'vpd_winter_{year}': VPD_winter[i] for i, year in enumerate(years)},

                **{f'tm_annual_{year}': tm_annual[i] for i, year in enumerate(years)},
                **{f'tm_spring_{year}': tm_spring[i] for i, year in enumerate(years)},
                **{f'tm_summer_{year}': tm_summer[i] for i, year in enumerate(years)},
                **{f'tm_autumn_{year}': tm_autumn[i] for i, year in enumerate(years)},
                **{f'tm_summerhalf_{year}': tm_summerhalf[i] for i, year in enumerate(years)},
                **{f'tm_winterhalf_{year}': tm_winterhalf[i] for i, year in enumerate(years)},
                **{f'tm_chilling_{year}': tm_chilling[i] for i, year in enumerate(years)},

                **{f'spei_03_annual_spei_{year}': spei_03_annual_spei[i] for i, year in enumerate(years)},
                **{f'spei_03_spring_spei_{year}': spei_03_spring_spei[i] for i, year in enumerate(years)},
                **{f'spei_03_summer_spei_{year}': spei_03_summer_spei[i] for i, year in enumerate(years)},
                **{f'spei_03_autumn_spei_{year}': spei_03_autumn_spei[i] for i, year in enumerate(years)},
                **{f'spei_06_annual_spei_{year}': spei_06_annual_spei[i] for i, year in enumerate(years)},
                **{f'spei_06_summer_half_spei_{year}': spei_06_summer_half_spei[i] for i, year in enumerate(years)},
                **{f'spei_06_winter_half_spei_{year}': spei_06_winter_half_spei[i] for i, year in enumerate(years)},
            })

        # 将数据转换为DataFrame
        df = pd.DataFrame(data)

        # 将DataFrame写入CSV文件
        df.to_csv(os.path.join(os.path.split(mask_tif)[0],f'inform_sum_sa_EVI_SIF_{df_save_part}.csv'), index=False)
        return 'useful'
    def summarize_raster_by_mask_aggregation_basedon_existingnc(self,phenology_tifs,mask_tifs,gdd_paths,SPEI_paths,temperature_path,VPD_path,chillinday_paths):
        '''
        这个和上面那个summarize_raster_by_mask_aggregation方法一模一样，但是，不用重新聚合了，直接用之前保存的数据就可以了
        '''
        # 设置波段索引
        band_sos_number = 1  # SOS对应的波段
        band_eos_number = 3  # EOS对应的波段
        # sos_stack,eos_stack,phenology_tif_geography = self.stack_bands(phenology_tifs,band_sos_number,band_eos_number)
        phenology_data = self.stack_bands_tonc(phenology_tifs, band_sos_number, band_eos_number)
        # 获取GDD,SPEI,温度
        # gdd = self.get_GDD_current_region(gdd_paths, phenology_tifs[0])
        # spei_03 = self.get_SPEI_current_region(SPEI_paths[0], phenology_tifs[0], '03')
        # spei_06 = self.get_SPEI_current_region(SPEI_paths[1], phenology_tifs[0], '06')
        # temperature = self.get_Temperature_average_current_region(temperature_path, phenology_tifs[0])
        # VPD = self.get_VPD_average_current_region(VPD_path, phenology_tifs[0])
        chillinday = self.get_TMchillingdays_current_region(chillinday_paths, phenology_tifs[0])
        # for mask_tif in mask_tifs:
        mask_tif = mask_tifs[0]
        # if os.path.exists(os.path.join(os.path.split(mask_tif)[0],'inform_sum_sa.csv')):return None
        drought_year = re.search(r'_(\d{4})\.tif$', mask_tif).group(1)

        # 看聚合后的栅格中，每个栅格有多少原来的点（经过掩膜后过滤的点）参与了计算,同时绘制直方图
        mask_nc = self.tif_tonc(mask_tif)
        mask_num_nc = self.spatial_aggregation(mask_nc, xr.open_dataset(SPEI_paths[0]), 'sum')
        data_mask_num = mask_num_nc['number']['data'].values.flatten()
        df_mask_num = pd.DataFrame({'grid_id': range(len(data_mask_num)), 'data': data_mask_num})
        # # 根据四分位数确定mask数量的阈值
        # try:
        #     mask_num_threshold = np.percentile(data_mask_num[data_mask_num>0],25)
        # except:
        #     mask_num_threshold = 30.0
        # 直接定义阈值为1，后续采用加权回归模型
        mask_num_threshold = 61.0
        plt.figure(figsize=(10, 6))
        plt.bar(df_mask_num['grid_id'], df_mask_num['data'], color='skyblue')
        plt.title('Data Participation Count per Grid Cell')
        plt.xlabel('Grid ID')
        plt.ylabel('Data Participation Count')
        plt.grid(axis='y')
        plt.savefig(os.path.join(os.path.split(mask_tif)[0],f'{drought_year}_histogram.jpg'))
        mask_array = mask_nc['data'].data
        mask_array_phenology = np.repeat(np.expand_dims(mask_array.data,0),23,axis=0)
        phenology_sa = self.spatial_aggregation(
            phenology_data.where(mask_array_phenology == 1),
            xr.open_dataset(SPEI_paths[0]), 'mean')

        mask_array_others = np.repeat(np.expand_dims(mask_array.data, 0), 24, axis=0)     #这里面除了物候数据，其他数据年数都是2000年到2023年，因此统一用一个expandmask的形状就好了
        # gdd_sa = self.spatial_aggregation(gdd.where(mask_array_others == 1),xr.open_dataset(SPEI_paths[0]), 'mean')
        # # lat_mid = gdd.lat.mean().item()  # 纬度的中间点
        # # lon_mid = gdd.lon.mean().item()  # 经度的中间点
        # # ds_top_left = gdd.where((gdd.lat >= lat_mid) & (gdd.lon <= lon_mid), drop=True)
        # # # 2. 右上区域（纬度大于中间点， 经度大于中间点）
        # # ds_top_right = gdd.where((gdd.lat >= lat_mid) & (gdd.lon > lon_mid), drop=True)
        # # # 3. 左下区域（纬度小于中间点， 经度小于中间点）
        # # ds_bottom_left = gdd.where((gdd.lat < lat_mid) & (gdd.lon <= lon_mid), drop=True)
        # # # 4. 右下区域（纬度小于中间点， 经度大于中间点）
        # # ds_bottom_right = gdd.where((gdd.lat < lat_mid) & (gdd.lon > lon_mid), drop=True)
        # # ds_top_leftsa = self.spatial_aggregation(ds_top_left.where(mask_array_others == 1),
        # #                                          xr.open_dataset(SPEI_paths[0]), 'mean')
        # # ds_top_rightsa = self.spatial_aggregation(ds_top_right.where(mask_array_others == 1),
        # #                                           xr.open_dataset(SPEI_paths[0]), 'mean')
        # # ds_bottom_leftsa = self.spatial_aggregation(ds_bottom_left.where(mask_array_others == 1),
        # #                                             xr.open_dataset(SPEI_paths[0]), 'mean')
        # # ds_bottom_rightsa = self.spatial_aggregation(ds_bottom_right.where(mask_array_others == 1),
        # #                                              xr.open_dataset(SPEI_paths[0]), 'mean')
        # spei_03_sa = self.spatial_aggregation(spei_03.where(mask_array_others == 1),xr.open_dataset(SPEI_paths[0]), 'mean')
        # spei_06_sa = self.spatial_aggregation(spei_06.where(mask_array_others == 1),xr.open_dataset(SPEI_paths[0]), 'mean')
        # temperature_sa = self.spatial_aggregation(temperature.where(mask_array_others == 1),xr.open_dataset(SPEI_paths[0]), 'mean')
        # VPD_sa = self.spatial_aggregation(VPD.where(mask_array_others == 1), xr.open_dataset(SPEI_paths[0]), 'mean')
        chillinday_sa = self.spatial_aggregation(chillinday.where(mask_array_others == 1), xr.open_dataset(SPEI_paths[0]), 'mean')
        # 如果该点的有效数据数量小于阈值，就放弃
        rows, cols = np.where(mask_num_nc['number']['data']>mask_num_threshold)
        if len(rows) == 0 and len(cols) == 0: return None
        '''
        保存聚合后的masktif和所有的nc变量文件
        '''
        # xr.merge([gdd_sa, spei_03_sa.rename({'annual_spei': 'annual_spei_03'}),
        #           spei_06_sa.rename({'annual_spei': 'annual_spei_06'}), temperature_sa, VPD_sa,chillinday_sa]).to_netcdf(os.path.join(os.path.split(mask_tif)[0], f'info_sum_sa.nc'))
        infom_sum_sa = xr.open_dataset(os.path.join(os.path.split(mask_tif)[0], f'info_sum_sa.nc'))
        phenology_sa.to_netcdf(os.path.join(os.path.split(mask_tif)[0], f'phenology_sa.nc'))
        valid_data_mask = np.zeros(mask_num_nc['number']['data'].shape, dtype=np.uint8)
        # 根据有效数据的位置标记 1
        valid_data_mask[rows, cols] = 1
        # 定义输出tif文件路径
        output_tif = os.path.join(os.path.split(mask_tif)[0], f'valid_data_mask.tif')
        mask_sa_tif = mask_num_nc['number']['data'].copy()
        mask_sa_tif = mask_sa_tif.rename({'lon':'x','lat':'y'})
        mask_sa_tif.data = valid_data_mask
        mask_sa_tif.rio.write_crs("EPSG:4326", inplace=True).rio.to_raster(output_tif)
        mask_num_nc['number']['data'].copy().rename({'lon':'x','lat':'y'}).rio.write_crs("EPSG:4326", inplace=True).rio.to_raster(os.path.join(os.path.split(mask_tif)[0], f'valid_data_mask_number.tif'))
        mask_num_nc['ratio']['data'].copy().rename({'lon': 'x', 'lat': 'y'}).rio.write_crs("EPSG:4326",inplace=True).rio.to_raster(os.path.join(os.path.split(mask_tif)[0], f'valid_data_mask_ratio.tif'))

        '''
        写入csv
        '''
        # 创建一个空的列表来存储所有行的数据
        data= []
        # 定义年份范围
        years = list(range(2000, 2024))  # 2000年到2023年

        for i, (row, col) in tqdm(enumerate(zip(rows, cols))):

            sos_values = phenology_sa['SOS'][:, row, col].data  # SOS 时序数据
            eos_values = phenology_sa['EOS'][:, row, col].data  # EOS 时序数据

            gdd_fixed_sum = infom_sum_sa['fixed_sum'].data[:,row,col]
            VPD_MA_avg = infom_sum_sa['MA_avg'].data[:, row, col]

            tm_annual = infom_sum_sa['annual_avg'].data[:, row, col]
            tm_spring = infom_sum_sa['spring_avg'].data[:, row, col]
            tm_summer = infom_sum_sa['summer_avg'].data[:, row, col]
            tm_autumn = infom_sum_sa['autumn_avg'].data[:, row, col]
            tm_summerhalf = infom_sum_sa['summerhalf_avg'].data[:, row, col]
            tm_winterhalf = infom_sum_sa['winterhalf_avg'].data[:, row, col]
            tm_chilling = infom_sum_sa['chilling_avg'].data[:, row, col]
            chillinday_sa_value = chillinday_sa['chillingdays'].data[:, row, col]
            spei_03_annual_spei = infom_sum_sa['annual_spei_03'].data[:,row,col]
            spei_03_spring_spei = infom_sum_sa['spring_spei'].data[:, row, col]
            spei_03_summer_spei = infom_sum_sa['summer_spei'].data[:, row, col]
            spei_03_autumn_spei = infom_sum_sa['autumn_spei'].data[:, row, col]

            spei_06_annual_spei = infom_sum_sa['annual_spei_06'].data[:,row,col]
            spei_06_summer_half_spei = infom_sum_sa['summer_half_spei'].data[:, row, col]
            spei_06_winter_half_spei = infom_sum_sa['winter_half_spei'].data[:, row, col]
            # 将数据添加到列表中
            data.append({
                'row': row,
                'col': col,
                'weights':mask_num_nc['ratio']['data'].data[row, col],
                **{f'sos_{year}': sos_values[i] for i, year in enumerate(list(range(2001, 2024)) )},  # SOS 数据
                **{f'eos_{year}': eos_values[i] for i, year in enumerate(list(range(2001, 2024)) )},  # EOS 数据
                **{f'gdd_fixed_sum_{year}': gdd_fixed_sum[i] for i, year in enumerate(years)},
                **{f'VPD_MA_avg_{year}': VPD_MA_avg[i] for i, year in enumerate(years)},
                **{f'tm_annual_{year}': tm_annual[i] for i, year in enumerate(years)},
                **{f'tm_spring_{year}': tm_spring[i] for i, year in enumerate(years)},
                **{f'tm_summer_{year}': tm_summer[i] for i, year in enumerate(years)},
                **{f'tm_autumn_{year}': tm_autumn[i] for i, year in enumerate(years)},
                **{f'tm_summerhalf_{year}': tm_summerhalf[i] for i, year in enumerate(years)},
                **{f'tm_winterhalf_{year}': tm_winterhalf[i] for i, year in enumerate(years)},
                **{f'tm_chilling_{year}': tm_chilling[i] for i, year in enumerate(years)},
                **{f'chillingdays_{year}': chillinday_sa_value[i] for i, year in enumerate(years)},
                **{f'spei_03_annual_spei_{year}': spei_03_annual_spei[i] for i, year in enumerate(years)},
                **{f'spei_03_spring_spei_{year}': spei_03_spring_spei[i] for i, year in enumerate(years)},
                **{f'spei_03_summer_spei_{year}': spei_03_summer_spei[i] for i, year in enumerate(years)},
                **{f'spei_03_autumn_spei_{year}': spei_03_autumn_spei[i] for i, year in enumerate(years)},
                **{f'spei_06_annual_spei_{year}': spei_06_annual_spei[i] for i, year in enumerate(years)},
                **{f'spei_06_summer_half_spei_{year}': spei_06_summer_half_spei[i] for i, year in enumerate(years)},
                **{f'spei_06_winter_half_spei_{year}': spei_06_winter_half_spei[i] for i, year in enumerate(years)},
            })

        # 将数据转换为DataFrame
        df = pd.DataFrame(data)

        # 将DataFrame写入CSV文件
        df.to_csv(os.path.join(os.path.split(mask_tif)[0],'inform_sum_sa_maskthreshold61.csv'), index=False)
    def summarize_raster_by_mask_aggregation_addTMchillingdays(self,phenology_tifs,mask_tifs,SPEI_paths,chillinday_paths):
        # 设置波段索引
        band_sos_number = 1  # SOS对应的波段
        band_eos_number = 3  # EOS对应的波段
        # sos_stack,eos_stack,phenology_tif_geography = self.stack_bands(phenology_tifs,band_sos_number,band_eos_number)
        phenology_data = self.stack_bands_tonc(phenology_tifs, band_sos_number, band_eos_number)
        # 获取因子
        chillinday = self.get_TMchillingdays_current_region(chillinday_paths, phenology_tifs[0])
        mask_tif = mask_tifs[0]
        # 将数据转换为DataFrame
        df = pd.read_csv(os.path.join(os.path.split(mask_tif)[0],'inform_sum_sa.csv'))

        # 看聚合后的栅格中，每个栅格有多少原来的点（经过掩膜后过滤的点）参与了计算,同时绘制直方图
        mask_nc = self.tif_tonc(mask_tif)
        mask_num_nc = self.spatial_aggregation(mask_nc, xr.open_dataset(SPEI_paths[0]), 'sum')
        data_mask_num = mask_num_nc['data'].values.flatten()
        df_mask_num = pd.DataFrame({'grid_id': range(len(data_mask_num)), 'data': data_mask_num})
        # 根据四分位数确定mask数量的阈值
        try:
            mask_num_threshold = np.percentile(data_mask_num[data_mask_num>0],25)
        except:
            mask_num_threshold = 30.0
        mask_array = mask_nc['data'].data
        mask_array_others = np.repeat(np.expand_dims(mask_array.data, 0), 24, axis=0)      #这里面除了物候数据，其他数据年数都是2000年到2023年，因此统一用一个expandmask的形状就好了
        chillinday_sa = self.spatial_aggregation(chillinday.where(mask_array_others == 1),xr.open_dataset(SPEI_paths[0]), 'mean')
        # 如果该点的有效数据数量小于阈值，就放弃
        rows, cols = np.where(mask_num_nc['data']>mask_num_threshold)
        # 创建一个空的列表来存储所有行的数据
        data= []
        # 定义年份范围
        years = list(range(2000, 2024))  # 2000年到2023年
        for i, (row, col) in enumerate(zip(rows, cols)):
            chillinday_sa_value = chillinday_sa['chillingdays'].data[:, row, col]
            # 将数据添加到列表中
            data.append({
                'row': row,
                'col': col,
                **{f'chillingdays_{year}': chillinday_sa_value[i] for i, year in enumerate(years)}
            })
        factor_df = pd.DataFrame(data)
        df_cat = pd.concat([df,factor_df],axis=1)
        # 将DataFrame写入CSV文件
        df_cat.to_csv(os.path.join(os.path.split(mask_tif)[0],'inform_sum_sa.csv').replace('sa.csv','sa_chillingdays.csv'), index=False)
    def summarize_raster_by_mask_aggregation_addotherfactor(self,phenology_tifs,mask_tifs,SPEI_paths,factor_paths,factor_names):
        # 设置波段索引
        band_sos_number = 1  # SOS对应的波段
        band_eos_number = 3  # EOS对应的波段
        # sos_stack,eos_stack,phenology_tif_geography = self.stack_bands(phenology_tifs,band_sos_number,band_eos_number)
        phenology_data = self.stack_bands_tonc(phenology_tifs, band_sos_number, band_eos_number)
        for mask_tif in mask_tifs:
            # 将数据转换为DataFrame
            if os.path.exists(os.path.join(os.path.split(mask_tif)[0],'{}_inform_sum_sa_VPD.csv'.format(os.path.split(mask_tif)[-1].replace('.tif',''))).replace('sa_VPD.csv','sa_otherfactors.csv')): continue
            df = pd.read_csv(os.path.join(os.path.split(mask_tif)[0], '{}_inform_sum_sa.csv'.format(
                os.path.split(mask_tif)[-1].replace('.tif', ''))).replace('sa.csv', 'sa_VPD.csv'))
            df_cat = df.copy()
            for index, factor_path in enumerate(factor_paths):
                # 获取因子
                ofactor = self.get_otherfactor_average_current_region(factor_path, phenology_tifs[0],factor_names[index])

                drought_year =  re.search(r'_(\d{4})\.tif$', mask_tif).group(1)

                # 看聚合后的栅格中，每个栅格有多少原来的点（经过掩膜后过滤的点）参与了计算,同时绘制直方图
                mask_nc = self.tif_tonc(mask_tif)
                mask_num_nc = self.spatial_aggregation(mask_nc, xr.open_dataset(SPEI_paths[0]), 'sum')
                data_mask_num = mask_num_nc['data'].values.flatten()
                df_mask_num = pd.DataFrame({'grid_id': range(len(data_mask_num)), 'data': data_mask_num})
                # 根据四分位数确定mask数量的阈值
                try:
                    mask_num_threshold = np.percentile(data_mask_num[data_mask_num>0],25)
                except:
                    mask_num_threshold = 30.0
                mask_array = mask_nc['data'].data
                mask_array_others = np.repeat(np.expand_dims(mask_array.data, 0), 22, axis=0)     #这里面除了物候数据，其他数据年数都是2000年到2023年，因此统一用一个expandmask的形状就好了
                ofactor_sa = self.spatial_aggregation(ofactor.where(mask_array_others == 1),xr.open_dataset(SPEI_paths[0]), 'mean')
                # 如果该点的有效数据数量小于阈值，就放弃
                rows, cols = np.where(mask_num_nc['data']>mask_num_threshold)
                # 创建一个空的列表来存储所有行的数据
                data= []
                # 定义年份范围
                years = list(range(2000, 2024))  # 2000年到2023年

                for i, (row, col) in enumerate(zip(rows, cols)):

                    ofactor_annual = ofactor_sa['annual_avg'].data[:, row, col]
                    ofactor_spring = ofactor_sa['spring_avg'].data[:, row, col]
                    ofactor_summer = ofactor_sa['summer_avg'].data[:, row, col]
                    ofactor_autumn = ofactor_sa['autumn_avg'].data[:, row, col]
                    ofactor_summerhalf = ofactor_sa['summerhalf_avg'].data[:, row, col]
                    ofactor_winterhalf = ofactor_sa['winterhalf_avg'].data[:, row, col]

                    # 将数据添加到列表中
                    data.append({
                        'row': row,
                        'col': col,
                        **{f'{factor_names[index]}_annual_{year}': ofactor_annual[i] for i, year in enumerate(years)},
                        **{f'{factor_names[index]}_spring_{year}': ofactor_spring[i] for i, year in enumerate(years)},
                        **{f'{factor_names[index]}_summer_{year}': ofactor_summer[i] for i, year in enumerate(years)},
                        **{f'{factor_names[index]}_autumn_{year}': ofactor_autumn[i] for i, year in enumerate(years)},
                        **{f'{factor_names[index]}_summerhalf_{year}': ofactor_summerhalf[i] for i, year in enumerate(years)},
                        **{f'{factor_names[index]}_winterhalf_{year}': ofactor_winterhalf[i] for i, year in enumerate(years)},
                    })
                factor_df = pd.DataFrame(data)

                df_cat = pd.concat([df_cat,factor_df],axis=1)
                # 将DataFrame写入CSV文件
            df_cat.to_csv(os.path.join(os.path.split(mask_tif)[0],'{}_inform_sum_sa_VPD.csv'.format(os.path.split(mask_tif)[-1].replace('.tif',''))).replace('sa_VPD.csv','sa_otherfactors.csv'), index=False)
    def pure_data_df(self,filtered_df,drought_year,drought_timing,drought_year_spei,spei_scale):
        threshold_collections = {
            '03': {
                'spring': filtered_df[f'spei_03_spring_spei_{drought_year}'][
                    filtered_df[f'spei_03_spring_spei_{drought_year}'] > filtered_df[
                        f'spei_03_spring_spei_{drought_year}'].min()].quantile(0.25),
                'summer': filtered_df[f'spei_03_summer_spei_{drought_year}'][
                    filtered_df[f'spei_03_summer_spei_{drought_year}'] > filtered_df[
                        f'spei_03_summer_spei_{drought_year}'].min()].quantile(0.25),
                'autumn': filtered_df[f'spei_03_autumn_spei_{drought_year}'][
                    filtered_df[f'spei_03_autumn_spei_{drought_year}'] > filtered_df[
                        f'spei_03_autumn_spei_{drought_year}'].min()].quantile(0.25),
                'annual': filtered_df[f'spei_03_annual_spei_{drought_year}'][
                    filtered_df[f'spei_03_annual_spei_{drought_year}'] > filtered_df[
                        f'spei_03_annual_spei_{drought_year}'].min()].quantile(0.25)
            },
            '06': {
                'summerhalf': filtered_df[f'spei_06_summerhalf_spei_{drought_year}'][
                    filtered_df[f'spei_06_summerhalf_spei_{drought_year}'] > filtered_df[
                        f'spei_06_summerhalf_spei_{drought_year}'].min()].quantile(0.25),
                'winterhalf': filtered_df[f'spei_06_winterhalf_spei_{drought_year}'][
                    filtered_df[f'spei_06_winterhalf_spei_{drought_year}'] > filtered_df[
                        f'spei_06_winterhalf_spei_{drought_year}'].min()].quantile(0.25),
                'annual': filtered_df[f'spei_06_annual_spei_{drought_year}'][
                    filtered_df[f'spei_06_annual_spei_{drought_year}'] > filtered_df[
                        f'spei_06_annual_spei_{drought_year}'].min()].quantile(0.25)

            }
        }
        threshold_collections_static = {
            '03': {
                'spring': 0,
                'summer': 0,
                'autumn': 0,
                'annual': 0
            },
            '06': {
                'summerhalf': 0,
                'winterhalf': 0,
                'annual': 0

            }
        }
        if drought_timing == 'spring':
            drought_filtered_df = filtered_df[
                (drought_year_spei < threshold_collections_static[spei_scale][drought_timing]) & (
                            filtered_df[f'spei_{spei_scale}_summer_spei_{drought_year}'] >=
                            threshold_collections_static[spei_scale]['summer']) & (
                            filtered_df[f'spei_{spei_scale}_autumn_spei_{drought_year}'] >=
                            threshold_collections_static[spei_scale]['autumn'])]
            undrought_filtered_df = filtered_df[
                (drought_year_spei >= threshold_collections_static[spei_scale][drought_timing]) & (
                            filtered_df[f'spei_{spei_scale}_summer_spei_{drought_year}'] >=
                            threshold_collections_static[spei_scale]['summer']) & (
                            filtered_df[f'spei_{spei_scale}_autumn_spei_{drought_year}'] >=
                            threshold_collections_static[spei_scale]['autumn'])]
        if drought_timing == 'summer':
            drought_filtered_df = filtered_df[
                (drought_year_spei < threshold_collections_static[spei_scale][drought_timing]) & (
                        filtered_df[f'spei_{spei_scale}_spring_spei_{drought_year}'] >=
                        threshold_collections_static[spei_scale]['spring']) & (
                        filtered_df[f'spei_{spei_scale}_autumn_spei_{drought_year}'] >=
                        threshold_collections_static[spei_scale]['autumn'])]
            undrought_filtered_df = filtered_df[
                (drought_year_spei >= threshold_collections_static[spei_scale][drought_timing]) & (
                        filtered_df[f'spei_{spei_scale}_spring_spei_{drought_year}'] >=
                        threshold_collections_static[spei_scale]['spring']) & (
                        filtered_df[f'spei_{spei_scale}_autumn_spei_{drought_year}'] >=
                        threshold_collections_static[spei_scale]['autumn'])]
        if drought_timing == 'autumn':
            drought_filtered_df = filtered_df[
                (drought_year_spei < threshold_collections_static[spei_scale][drought_timing]) & (
                        filtered_df[f'spei_{spei_scale}_spring_spei_{drought_year}'] >=
                        threshold_collections_static[spei_scale]['spring']) & (
                        filtered_df[f'spei_{spei_scale}_summer_spei_{drought_year}'] >=
                        threshold_collections_static[spei_scale]['summer'])]
            undrought_filtered_df = filtered_df[
                (drought_year_spei >= threshold_collections_static[spei_scale][drought_timing]) & (
                        filtered_df[f'spei_{spei_scale}_spring_spei_{drought_year}'] >=
                        threshold_collections_static[spei_scale]['spring']) & (
                        filtered_df[f'spei_{spei_scale}_summer_spei_{drought_year}'] >=
                        threshold_collections_static[spei_scale]['summer'])]
        if drought_timing == 'annual' and spei_scale == '03':
            drought_filtered_df = filtered_df[
                (drought_year_spei < threshold_collections_static[spei_scale][drought_timing])]
            undrought_filtered_df = filtered_df[
                (drought_year_spei >= threshold_collections_static[spei_scale][drought_timing])]
        if drought_timing == 'summerhalf':
            drought_filtered_df = filtered_df[
                (drought_year_spei < threshold_collections_static[spei_scale][drought_timing]) & (
                        filtered_df[f'spei_{spei_scale}_winterhalf_spei_{drought_year}'] >=
                        threshold_collections_static[spei_scale]['winterhalf'])]
            undrought_filtered_df = filtered_df[
                (drought_year_spei >= threshold_collections_static[spei_scale][drought_timing]) & (
                        filtered_df[f'spei_{spei_scale}_winterhalf_spei_{drought_year}'] >=
                        threshold_collections_static[spei_scale]['winterhalf'])]
        if drought_timing == 'winterhalf':
            drought_filtered_df = filtered_df[
                (drought_year_spei < threshold_collections_static[spei_scale][drought_timing]) & (
                        filtered_df[f'spei_{spei_scale}_summerhalf_spei_{drought_year}'] >=
                        threshold_collections_static[spei_scale]['summerhalf'])]
            undrought_filtered_df = filtered_df[
                (drought_year_spei >= threshold_collections_static[spei_scale][drought_timing]) & (
                        filtered_df[f'spei_{spei_scale}_summerhalf_spei_{drought_year}'] >=
                        threshold_collections_static[spei_scale]['summerhalf'])]
        if drought_timing == 'annual' and spei_scale == '06':
            drought_filtered_df = filtered_df[
                (drought_year_spei < threshold_collections_static[spei_scale][drought_timing])]
            undrought_filtered_df = filtered_df[
                (drought_year_spei >= threshold_collections_static[spei_scale][drought_timing])]
        return drought_filtered_df,undrought_filtered_df

    # 定义一个函数来创建和拟合模型，并返回结果
    def run_glm(self,formula, data):
        model = sm.GLM.from_formula(formula, data=data, family=sm.families.Gaussian())
        results = model.fit(weights=data['weights'])
        return results

    # 定义一个函数来提取模型的系数和统计信息
    def extract_model_info(self,results, formula):
        params = results.params
        pvalues = results.pvalues
        std_errors = results.bse
        summary_df = pd.DataFrame({
            'Coefficient': params,
            'Std Error': std_errors,
            'P-value': pvalues
        })
        summary_df['Formula'] = formula
        return summary_df
    def statistical_variables_seprating_test(self,factors,response_var,data,interaction_term):
        # 存储所有模型的汇总信息
        model_summaries = []
        # 单因子模型
        for factor in factors:
            formula = f'{response_var} ~ {factor}'
            model_summaries.append(self.extract_model_info(self.run_glm(formula, data), formula))
        # 双因子模型组合
        combinations_2 = itertools.combinations(factors, 2)  # 生成双因子组合
        for combo in combinations_2:
            formula = f'{response_var} ~ {combo[0]} + {combo[1]}'
            model_summaries.append(self.extract_model_info(self.run_glm(formula, data), formula))
        # # 三因子模型组合
        # combinations_3 = itertools.combinations(factors, 3)  # 生成三因子组合
        # for combo in combinations_3:
        #     formula = f'{response_var_sos} ~ {combo[0]} + {combo[1]} + {combo[2]}'
        #     model_summaries.append(extract_model_info(run_glm(formula, filtered_df_stack), formula))

        # 完整模型
        all_factors = ' + '.join(factors)  # 将因子连接成字符串
        complete_formula = f'{response_var} ~ {all_factors} + {interaction_term}'  # 动态构建完整模型公式
        model_summaries.append(self.extract_model_info(self.run_glm(complete_formula, data), complete_formula))
        # 汇总所有模型信息到一个数据框
        summary_table = pd.concat(model_summaries, keys=[i for i in range(len(model_summaries))], names=['Model'])
        # 展示表格
        summary_table.reset_index(level=0, inplace=True)
        summary_table.to_csv('test.csv')
    def create_pred_df_for_marginaleffect(self,data, formula, control_var, marginal_var, control_value, marginal_range,marginal_item_name):
        '''
        动态创建预测数据集，用于计算边际效应时创建其他变量的数据，全部取中值
        :param data: 输入数据
        :param formula: 公式
        :param control_var: 控制变量
        :param marginal_var: 边际效应变量
        :param control_value: 控制变量的值
        :param marginal_range: 边际效应变量的范围
        :param marginal_item_name: 边际变量项的名字
        :return:
        '''

        # 解析公式，提取出所有变量
        formula_vars = [var.strip() for var in formula.split('~')[1].split('+')]
        # 移除交互项的变量，保留其他变量
        pred_dict = {control_var: control_value, marginal_var: marginal_range}
        for var in formula_vars:
            if var != marginal_item_name and var not in [control_var, marginal_var]:
                if '*' in var:
                    var_pre,var_post = var.split('*')[0],var.split('*')[1]
                    data[var.replace('*','_')] = data[var_pre]*data[var_post]
                    var = var.replace('*','_')
                # 动态取中值
                pred_dict[var] = data[var].median()
        # 创建预测数据集
        pred_df = pd.DataFrame(pred_dict)
        return pred_df
    # 简化公式的函数

    def simplify_formula(self,formula):
        variable_map = {
            'gdd_drought_year': 'G_d',
            'vpd_drought_year': 'V_d',
            'sos_drought_year': 'sos_d',
            'spei_spring_drought_year': 'Sp_d',
            'spei_summer_drought_year': 'Su_d',
            'spei_autumn_drought_year': 'Au_d',
            'gdd': 'G',
            'vpd': 'V',
            'spei_spring_lag': 'Sp',
            'spei_summer_lag': 'Su',
            'spei_autumn_lag': 'Au',
            'spei_winterhalf_lag': 'Wi',
            'G_fixed_sum_2020': 'G',  # 因为在简化公式的函数中，会先后遍历变量名，前面gdd已经换过了，所以这里用G_fixed.....
            'VPD_MA_avg_2020': 'V',
            'G_fixed_sum_2023': 'G',  # 因为在简化公式的函数中，会先后遍历变量名，前面gdd已经换过了，所以这里用G_fixed.....
            'VPD_MA_avg_2023': 'V',
            'spei_03_spring_spei_2018': 'Sp18',
            'spei_03_summer_spei_2018': 'Su18',
            'spei_03_autumn_spei_2018': 'Au18',
            'spei_06_winterhalf_spei_2018': 'Wi18',
            'spei_03_spring_spei_2019': 'Sp19',
            'spei_03_summer_spei_2019': 'Su19',
            'spei_03_autumn_spei_2019': 'Au19',
            'spei_06_winterhalf_spei_2019': 'Wi19',
            'spei_03_spring_spei_2022': 'Sp22',
            'spei_03_summer_spei_2022': 'Su22',
            'spei_03_autumn_spei_2022': 'Au22',
            'spei_06_winterhalf_spei_2022': 'Wi22',
            'spei_03_spring_spei_1819': 'Sp1819',
            'spei_03_summer_spei_1819': 'Su1819',
            'spei_03_autumn_spei_1819': 'Au1819',
            'spei_06_winterhalf_spei_1819': 'Wi1819',
            'compound_spei_1819': 'Co1819',
            'compound_spei_181922': 'Co181922',
        }
        # 替换变量名为缩写
        for long_var, short_var in variable_map.items():
            formula = formula.replace(long_var, short_var)
        # 简化交互项格式
        formula = formula.replace(':', ':')  # 保持交互项格式，如 G:Sp
        return formula

    def expand_formula(self, simplified_formula):
        variable_map = {
            'G': 'gdd',
            'V': 'vpd',
            'Sp': 'spei_spring_lag',
            'Su': 'spei_summer_lag',
            'Au': 'spei_autumn_lag',
            'Wi': 'spei_winterhalf_lag',
            'Sp18': 'spei_03_spring_spei_2018',
            'Su18': 'spei_03_summer_spei_2018',
            'Au18': 'spei_03_autumn_spei_2018',
            'Wi18': 'spei_06_winterhalf_spei_2018',
            'Sp19': 'spei_03_spring_spei_2019',
            'Su19': 'spei_03_summer_spei_2019',
            'Au19': 'spei_03_autumn_spei_2019',
            'Wi19': 'spei_06_winterhalf_spei_2019',
            'Sp22': 'spei_03_spring_spei_2022',
            'Su22': 'spei_03_summer_spei_2022',
            'Au22': 'spei_03_autumn_spei_2022',
            'Wi22': 'spei_06_winterhalf_spei_2022',
            'Co1819': 'compound_spei_1819',
            'Co181922': 'compound_spei_181922',
            'G_d': 'gdd_drought_year',
            'V_d': 'vpd_drought_year',
            'sos_d': 'sos_drought_year',
            'Sp_d': 'spei_spring_drought_year',
            'Su_d': 'spei_summer_drought_year',
            'Au_d': 'spei_autumn_drought_year',
        }
        # 按缩写的长度从长到短排序，防止短缩写优先替换而干扰长缩写
        for short_var, long_var in sorted(variable_map.items(), key=lambda x: -len(x[0])):
            simplified_formula = simplified_formula.replace(short_var, long_var)
        # 将交互项中的“:”替换回“*”格式
        expanded_formula = simplified_formula.replace(':', '*')
        return expanded_formula
    def marginal_effects_calculation(self,data,results,output_path,formula,control_var,marginal_var,marginal_item_name,response_var):
        '''
        边际效应计算，可视化，仅限单一边际效应，即只有一个交互项，或者有多个交互项，但只有一个交互项有需要计算边际效应的变量，若有多个需要修改
        :param data:
        :param results:
        :param output_path:
        :param formula:
        :param control_var:
        :param marginal_var:
        :param marginal_item_name:
        :param response_var:
        :return:
        '''
        control_var,marginal_var,marginal_item_name = control_var.strip(),marginal_var.strip(),marginal_item_name.strip()
        control_values = [data[control_var].quantile(0.25), data[control_var].median(),data[control_var].quantile(0.75)]
        # 绘制边际效应图
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'orange', 'red']  # 分别为低、中、高GDD条件的颜色
        # 对每control值，计算marginal var在不同条件下的边际效应
        for i, control_value in enumerate(control_values):
            beta_2 = results.params[marginal_var]
            beta_3 = results.params[marginal_item_name.replace('*',':')]
            marginal_effects = beta_2 + beta_3 * control_value

            # 计算不同marginal var值下的预测resoibse var值
            marginal_range = np.linspace(data[marginal_var].min(),data[marginal_var].max(), 100)

            # 创建数据集来计算预测值和置信区间
            pred_df = self.create_pred_df_for_marginaleffect(data,formula,control_var,marginal_var,control_value,marginal_range,marginal_item_name)

            # 获取预测值和置信区间
            predictions = results.get_prediction(pred_df)
            pred_mean = predictions.predicted_mean
            pred_ci = predictions.conf_int()
            # 绘制预测值曲线
            plt.plot(marginal_range, pred_mean, label= control_var+f' = {control_value:.2f} (Marginal Effect: {marginal_effects:.2f})',
                     color=colors[i])

            # 绘制置信区间，阴影更深
            plt.fill_between(marginal_range, pred_ci[:, 0], pred_ci[:, 1], color=colors[i], alpha=0.4)

        # 图形设置
        plt.xlabel(marginal_var)
        plt.ylabel(response_var)
        plt.legend()
        plt.grid(True)
        title_text = self.simplify_formula(formula)
        wrapped_title = "\n".join(textwrap.wrap(title_text, width=50))  # 50表示每行字符限制
        plt.title(wrapped_title)
        # plt.savefig(output_path, format='eps', dpi=600)
        # plt.savefig(output_path, format='pdf', dpi=600)
        try:
            plt.savefig(output_path.replace('~','-'), format='jpeg', dpi=300)
            plt.close()
        except:
            print('test:'+ output_path.replace('~','-'))
            # plt.savefig(output_path, format='jpeg', dpi=300)
            plt.close()

    def single_model_test(self,formula,filtered_df_stack,output_path_marginal_plot,response_var):
        model = glm(formula, data=filtered_df_stack, family=sm.families.Gaussian())
        results = model.fit(weights=filtered_df_stack['weights'])
        print(results.summary())
        interaction_term = [term for term in formula.split('+') if '*' in term]
        for interaction_term_item in interaction_term:
            control_var, marginal_var = interaction_term_item.split('*')
            # marginal_var,control_var = interaction_term_item.split('*')
            self.marginal_effects_calculation(filtered_df_stack, results, output_path_marginal_plot[:-4]+interaction_term_item.replace('*','_')+'.jpg',
                                              formula, control_var, marginal_var, interaction_term_item, response_var)

    def different_model_test(self,spei_vars,response_var_sos,gdd_var,vpd_var,filtered_df_stack,drought_year,modelling_object,model_results_save_path,sos_var_for_eosmodelling = None,tm_summer_var_for_eosmodelling=None,tm_summerhalf_var_for_eosmodelling=None):
        def generate_multiformula_set(spei_vars,modelling_object):
            # 创建存储公式的集合（用set来去重）
            formula_set = set()
            # 动态生成包含1到4个SPEI变量的公式，并为每个公式添加一种交互项
            for r in range(1, len(spei_vars) + 1):
                # 生成长度为r的所有SPEI变量组合
                for spei_combination in itertools.combinations(spei_vars, r):
                    # 基础公式
                    if modelling_object[0:1] == 'S':
                        base_formulas = [f'{response_var_sos} ~ {gdd_var} + {vpd_var}']
                    else:
                        base_formulas = [f'{response_var_sos} ~ {gdd_var} + {vpd_var} + {sos_var_for_eosmodelling} + {tm_summer_var_for_eosmodelling} + {tm_summer_var_for_eosmodelling}*{sos_var_for_eosmodelling}',
                             f'{response_var_sos} ~ {gdd_var} + {vpd_var} + {sos_var_for_eosmodelling} + {tm_summerhalf_var_for_eosmodelling} + {tm_summerhalf_var_for_eosmodelling}*{sos_var_for_eosmodelling}',
                             ]
                        # base_formulas = [f'{response_var_sos} ~ {gdd_var} + {vpd_var} + {sos_var_for_eosmodelling} + {tm_summer_var_for_eosmodelling}',
                        #      f'{response_var_sos} ~ {gdd_var} + {vpd_var} + {sos_var_for_eosmodelling} + {tm_summerhalf_var_for_eosmodelling}',
                        #      ]
                    # 添加SPEI变量项
                    spei_terms = ' + '.join(spei_combination)
                    # 对于每个公式，添加一种交互项 GDD * SPEI_x（任选一个）
                    for base_formula in base_formulas:
                        for spei in spei_combination:
                            interaction_term = f'{gdd_var}*{spei}'
                            # interaction_term = f'{spei}*{sos_var_for_eosmodelling}'
                            # 构建完整公式
                            full_formula = f'{base_formula} + {spei_terms} + {interaction_term}'
                            # if modelling_object[0:1] == 'E':full_formula = f'{base_formula} + {spei_terms}'
                            formula_set.add(full_formula)  # 将公式加入集合
            return formula_set

        formula_set = generate_multiformula_set(spei_vars,modelling_object)
        formula_df = pd.DataFrame({
            'Index': range(len(formula_set)),  # 生成索引
            'Formula': [self.simplify_formula(f) for f in formula_set]  # 集合转换为列表
        })
        formula_df.to_csv(model_results_save_path.replace('model_results.txt','formula_index.csv'), index=False)
        # 存储模型的结果
        model_results = []
        # 存储简化的公式标签
        formula_labels = {}
        # 打开文件以写入模型结果
        with open(model_results_save_path, 'w') as f:
            # 遍历每个公式并拟合模型
            for i,formula in enumerate(formula_set):
                print(f"Fitting model for formula: {formula}")
                model = glm(formula, data=filtered_df_stack, family=sm.families.Gaussian())
                results = model.fit(weights=filtered_df_stack['weights'])
                # 绘制边际效应
                # 提取交互项变量
                interaction_term = [term for term in formula.split('+') if '*' in term]
                if interaction_term:  # 如果有交互项
                    if modelling_object == 'SOS_Single_Drought':
                        gdd_var, spei_var = interaction_term[0].split('*')
                        # 绘制动态的边际效应图
                        # plot_marginal_effects(filtered_df_stack, results, gdd_var.strip(),
                        #                               spei_var.strip(), gdd_values,os.path.join(f'temp\statistical_results\Single_drought_metrics/{drought_year}\marginal effects',simplify_formula(formula).replace('*','_')+'.jpg'))
                        output_path_marginal_plot = os.path.join(f'temp\statistical_results(SOS)\Single_drought_metrics/{drought_year}\marginal effects',self.simplify_formula(formula).replace('*','_')+'.jpg')
                        # self.marginal_effects_calculation(filtered_df_stack,results,output_path_marginal_plot,formula,gdd_var,spei_var,interaction_term[0],'SOS')
                    if modelling_object == 'SOS_Multi_Drought':
                        gdd_var, spei_var = interaction_term[0].split('*')
                        output_path_marginal_plot = os.path.join(
                            f'temp\statistical_results(SOS)\Multi_drought_metrics/{drought_year}\marginal effects',
                            f'{i}.jpg')
                        self.marginal_effects_calculation(filtered_df_stack, results, output_path_marginal_plot,
                                                          formula, gdd_var, spei_var, interaction_term[0], 'SOS')
                    if modelling_object == 'SOS_Temporal_Series':
                        gdd_var, spei_var = interaction_term[0].split('*')
                        output_path_marginal_plot = os.path.join(
                            f'temp\statistical_results(SOS)\Temporal_Series/marginal effects',
                            self.simplify_formula(formula).replace('*', '_') + '.jpg')
                        self.marginal_effects_calculation(filtered_df_stack, results, output_path_marginal_plot,
                                                          formula, gdd_var, spei_var, interaction_term[0], 'SOS')
                    if modelling_object == 'EOS_Single_Drought':
                        for interaction_term_item in interaction_term:
                            control_var,marginal_var = interaction_term_item.split('*')
                            output_path_marginal_plot = os.path.join(f'temp\statistical_results(EOS)\Single_drought_metrics/{drought_year}\marginal effects',f'{i}.jpg')
                            # self.marginal_effects_calculation(filtered_df_stack, results, output_path_marginal_plot,
                            #                                   formula, control_var, marginal_var, interaction_term_item, 'EOS')
                    if modelling_object == 'EOS_Multi_Drought':
                        for interaction_term_item in interaction_term:
                            control_var,marginal_var = interaction_term_item.split('*')
                            output_path_marginal_plot = os.path.join(f'temp\statistical_results(EOS)\Multi_drought_metrics/{drought_year}\marginal effects',f'{i}.jpg')
                            self.marginal_effects_calculation(filtered_df_stack, results, output_path_marginal_plot,
                                                              formula, control_var, marginal_var, interaction_term_item, 'EOS')
                    if modelling_object == 'EOS_Temporal_Series':
                        for interaction_term_item in interaction_term:
                            control_var,marginal_var = interaction_term_item.split('*')
                            output_path_marginal_plot = os.path.join(f'temp\statistical_results(EOS)\Temporal_Series/marginal effects',(interaction_term_item + ' ' + self.simplify_formula(formula)).replace('*','_')+'.jpg')
                            # self.marginal_effects_calculation(filtered_df_stack, results, output_path_marginal_plot,
                            #                                   formula, control_var, marginal_var, interaction_term_item, 'EOS')

                # 获取模型结果的 summary
                summary_str = results.summary().as_text()
                aic = results.aic
                bic = results.bic
                pseudo_r_squared = results.pseudo_rsquared()
                # 保存结果
                model_results.append(
                    {'formula': formula, 'AIC': aic, 'BIC': bic, 'Pseudo R-squared': pseudo_r_squared, 'results': results})
                # 生成简化的公式标签
                simplified_label = self.simplify_formula(formula)  # 使用简化公式
                formula_labels[formula] = simplified_label  # 存储简化公式
                # 将公式和对应的 summary 写入文件
                f.write(f"Formula: {self.simplify_formula(formula)},[AIC:{aic} BIC:{bic}]\n")
                f.write(summary_str)
                f.write("\n\n" + "=" * 80 + "\n\n")  # 添加分隔符以区分不同模型的 summary

        # 生成AIC和BIC比较图
        aic_values = [model_result['AIC'] for model_result in model_results]
        bic_values = [model_result['BIC'] for model_result in model_results]
        pseudo_r_squared_values = [model_result['Pseudo R-squared'] for model_result in model_results]
        formulas = [model_result['formula'] for model_result in model_results]

        # 创建数据框用于可视化
        model_df = pd.DataFrame({
            'Formula': [formula_labels[formula] for formula in formulas],  # 使用简化的标签
            'AIC': aic_values,
            'BIC': bic_values,
            'Pseudo R-squared': pseudo_r_squared_values
        })
        model_df.to_csv(model_results_save_path.replace('.txt','metrics_ranking.csv'))

        file_save_path = {
            'EOS_Single_Drought':f'temp\statistical_results(EOS)\Single_drought_metrics/{drought_year}\marginal effects',
            'EOS_Multi_Drought':f'temp\statistical_results(EOS)\Multi_drought_metrics/{drought_year}\marginal effects',
            'EOS_Temporal_Series':f'temp\statistical_results(EOS)\Temporal_Series/marginal effects',
            'SOS_Single_Drought':f'temp\statistical_results(SOS)\Single_drought_metrics/{drought_year}\marginal effects',
            'SOS_Multi_Drought': f'temp\statistical_results(SOS)\Multi_drought_metrics/{drought_year}\marginal effects'
        }
        top_num = 5
        response_var_name = re.search(r"\((.*?)\)", file_save_path[modelling_object]).group(1)
        # 获取 AIC 排名前五的公式并执行 test 函数
        top_aic = model_df.nsmallest(top_num, 'AIC')
        for _, row in enumerate(top_aic.iterrows()):
            self.single_model_test(self.expand_formula(row[1]['Formula']),filtered_df_stack,os.path.join(file_save_path[modelling_object],f'AIC{_}.jpg'),response_var_name)

        # 获取 BIC 排名前五的公式并执行 test 函数
        top_bic = model_df.nsmallest(top_num, 'BIC')
        for _, row in enumerate(top_bic.iterrows()):
            self.single_model_test(self.expand_formula(row[1]['Formula']),filtered_df_stack,os.path.join(file_save_path[modelling_object],f'BIC{_}.jpg'),response_var_name)
        # 获取 Pseudo R-squared 排名前五的公式（这里是从大到小）
        top_r2 = model_df.nlargest(top_num, 'Pseudo R-squared')
        for _, row in enumerate(top_r2.iterrows()):
            self.single_model_test(self.expand_formula(row[1]['Formula']),filtered_df_stack,os.path.join(file_save_path[modelling_object],f'Pseudo R-squared{_}.jpg'),response_var_name)
        # # 设置绘图风格
        # plt.figure(figsize=(30, 6*(int(len(formula_set)/32))))
        # # 绘制AIC值的条形图
        # plt.subplot(1, 3, 1)
        # sns.barplot(x='AIC', y='Formula', data=model_df.sort_values('AIC'), palette='Blues')
        # for index, value in enumerate(model_df.sort_values('AIC')['AIC']):
        #     plt.text(value, index, f'{value:.2f}', color='black', va="center")
        # plt.title('Model AIC Comparison')
        # plt.xlabel('AIC')
        # plt.ylabel('Model Formula')
        # # 绘制BIC值的条形图
        # plt.subplot(1, 3, 2)
        # sns.barplot(x='BIC', y='Formula', data=model_df.sort_values('BIC'), palette='Reds')
        # for index, value in enumerate(model_df.sort_values('BIC')['BIC']):
        #     plt.text(value, index, f'{value:.2f}', color='black', va="center")
        # plt.title('Model BIC Comparison')
        # plt.xlabel('BIC')
        # plt.ylabel('Model Formula')
        # # 绘制Pseudo R-squared值的条形图
        # plt.subplot(1, 3, 3)
        # sns.barplot(x='Pseudo R-squared', y='Formula', data=model_df.sort_values('Pseudo R-squared', ascending=False),
        #             palette='Greens')
        # for index, value in enumerate(model_df.sort_values('Pseudo R-squared', ascending=False)['Pseudo R-squared']):
        #     plt.text(value, index, f'{value:.2f}', color='black', va="center")
        # plt.title('Model Pseudo R-squared Comparison')
        # plt.xlabel('Pseudo R-squared')
        # plt.ylabel('Model Formula')
        # plt.tight_layout()
        # plt.savefig(model_results_save_path.replace('.txt', 'metrics_ranking.pdf'), dpi=300)
        # plt.savefig(model_results_save_path.replace('.txt','metrics_ranking.jpg'),dpi=300)
    def different_model_test_for_multidrought(self,base_formula,filtered_df_stack,model_results_save_path,gdd_replacement):
        def extract_unique_spei_vars(formula):
            """
            从公式中提取按年份分组的 SPEI 变量，每个年份只包含独特变量。
            返回一个按年份分组的字典。
            """
            # 使用正则表达式提取变量并按年份分组
            year_vars = {}
            matches = re.findall(r'(spei_\d{2}_[a-z]+_spei_\d{4})', formula)
            for var in matches:
                year = re.search(r'(\d{4})', var).group(1)

                # 避免每年重复变量，只保留每年唯一的变量
                if year not in year_vars:
                    year_vars[year] = set()  # 使用集合确保唯一性
                year_vars[year].add(var)
            # 将集合转换为列表
            for year in year_vars:
                year_vars[year] = list(year_vars[year])
            return year_vars
        def generate_full_year_interaction(base_formula):
            """
            动态生成所有年份的 SPEI 变量的全年份交互项组合，并生成带有交互项的公式。
            """
            # 提取按年份分组的 SPEI 变量
            year_vars = extract_unique_spei_vars(base_formula)
            # 获取所有年份的 SPEI 变量列表
            spei_combinations = [year_vars[year] for year in sorted(year_vars.keys())]
            # 生成全年份交互项
            formula_set = set()
            for vars_tuple in itertools.product(*spei_combinations):
                # 创建交互项，例如 spei_03_spring_spei_2018*spei_03_autumn_spei_2019*spei_06_winterhalf_spei_2022
                interaction_term = '*'.join(vars_tuple)
                # 替换初始的 gdd 交互项为新生成的交互项
                modified_formula = re.sub(r'gdd\*spei_\d{2}_[a-z]+_spei_\d{4}', f'gdd*{interaction_term}', base_formula)
                # 去掉包含 "compound" 的项
                modified_formula = re.sub(r'\+?\s*compound_spei_\d+', '', modified_formula)
                # 添加新的交互项
                modified_formula += f' + {interaction_term}'
                # 添加到公式集合
                formula_set.add(modified_formula)

            return formula_set

        formula_set = generate_full_year_interaction(base_formula)
        # 存储模型的结果
        model_results = []
        # 存储简化的公式标签
        formula_labels = {}
        # 打开文件以写入模型结果
        with open(model_results_save_path, 'w') as f:
            # 遍历每个公式并拟合模型
            for i,formula in enumerate(formula_set):
                formula = formula.replace('gdd',gdd_replacement)
                print(f"Fitting model for formula: {formula}")
                model = glm(formula, data=filtered_df_stack, family=sm.families.Gaussian())
                results = model.fit(weights=filtered_df_stack['weights'])
                # 获取模型结果的 summary
                summary_str = results.summary().as_text()
                aic = results.aic
                bic = results.bic
                pseudo_r_squared = results.pseudo_rsquared()
                # 保存结果
                model_results.append(
                    {'formula': formula, 'AIC': aic, 'BIC': bic, 'Pseudo R-squared': pseudo_r_squared, 'results': results})
                # 生成简化的公式标签
                simplified_label = self.simplify_formula(formula)  # 使用简化公式
                formula_labels[formula] = simplified_label  # 存储简化公式
                # 将公式和对应的 summary 写入文件
                f.write(f"Formula: {self.simplify_formula(formula)},[AIC:{aic} BIC:{bic}]\n")
                f.write(summary_str)
                f.write("\n\n" + "=" * 80 + "\n\n")  # 添加分隔符以区分不同模型的 summary
        # 生成AIC和BIC比较图
        aic_values = [model_result['AIC'] for model_result in model_results]
        bic_values = [model_result['BIC'] for model_result in model_results]
        pseudo_r_squared_values = [model_result['Pseudo R-squared'] for model_result in model_results]
        formulas = [model_result['formula'] for model_result in model_results]
        # 创建数据框用于可视化
        model_df = pd.DataFrame({
            'Formula': [formula_labels[formula] for formula in formulas],  # 使用简化的标签
            'AIC': aic_values,
            'BIC': bic_values,
            'Pseudo R-squared': pseudo_r_squared_values
        })
        model_df.to_csv(model_results_save_path.replace('.txt','metrics_ranking.csv'))
        # # 设置绘图风格
        plt.figure(figsize=(32,10))
        # 绘制AIC值的条形图
        plt.subplot(1, 3, 1)
        sns.barplot(x='AIC', y='Formula', data=model_df.sort_values('AIC'), palette='Blues')
        for index, value in enumerate(model_df.sort_values('AIC')['AIC']):
            plt.text(value, index, f'{value:.2f}', color='black', va="center")
        plt.title('Model AIC Comparison')
        plt.xlabel('AIC')
        plt.ylabel('Model Formula')
        # 绘制BIC值的条形图
        plt.subplot(1, 3, 2)
        sns.barplot(x='BIC', y='Formula', data=model_df.sort_values('BIC'), palette='Reds')
        for index, value in enumerate(model_df.sort_values('BIC')['BIC']):
            plt.text(value, index, f'{value:.2f}', color='black', va="center")
        plt.title('Model BIC Comparison')
        plt.xlabel('BIC')
        plt.ylabel('Model Formula')
        # 绘制Pseudo R-squared值的条形图
        plt.subplot(1, 3, 3)
        sns.barplot(x='Pseudo R-squared', y='Formula', data=model_df.sort_values('Pseudo R-squared', ascending=False),
                    palette='Greens')
        for index, value in enumerate(model_df.sort_values('Pseudo R-squared', ascending=False)['Pseudo R-squared']):
            plt.text(value, index, f'{value:.2f}', color='black', va="center")
        plt.title('Model Pseudo R-squared Comparison')
        plt.xlabel('Pseudo R-squared')
        plt.ylabel('Model Formula')
        plt.tight_layout()
        plt.savefig(model_results_save_path.replace('.txt', 'metrics_ranking.pdf'), dpi=300)
        plt.savefig(model_results_save_path.replace('.txt','metrics_ranking.jpg'),dpi=300)

    def statistical_modelling(self,data_path,drought_year,legacy_end_year):
        df = pd.read_csv(data_path).dropna()
        # 过滤异常值
        sos_columns = [col for col in df.columns if 'sos' in col]
        eos_columns = [col for col in df.columns if 'eos' in col]
        sos_values = df[sos_columns].values
        eos_values = df[eos_columns].values
        # Calculate invalid SOS/EOS pairs
        invalid_sos_eos_counts = np.sum((sos_values >= eos_values) | (sos_values >= 180), axis=1)
        # Calculate invalid data points
        combined_values = np.concatenate([sos_values, eos_values], axis=1)
        invalid_data_points_counts = np.sum((combined_values == -32768) | (combined_values == 261), axis=1)
        # Filter the rows
        valid_rows = (invalid_sos_eos_counts == 0) & (invalid_data_points_counts == 0)
        filtered_df = df[valid_rows].copy()
        filtered_df[f'sos_baseline_prehalf'] = filtered_df[sos_columns[:len(sos_columns)//2]].mean(axis=1)
        filtered_df[f'eos_baseline_prehalf'] = filtered_df[eos_columns[:len(eos_columns)//2]].mean(axis=1)
        filtered_df[f'sos_baseline_posthalf'] = filtered_df[sos_columns[len(sos_columns)//2:]].mean(axis=1)
        filtered_df[f'eos_baseline_posthalf'] = filtered_df[eos_columns[len(eos_columns)//2:]].mean(axis=1)
        filtered_df['sos_baseline_whole'] = filtered_df[sos_columns].mean(axis=1)
        filtered_df['eos_baseline_whole'] = filtered_df[eos_columns].mean(axis=1)
        filtered_df['spei_03_spring_spei_1819'] = filtered_df['spei_03_spring_spei_2018'] + filtered_df['spei_03_spring_spei_2019']
        filtered_df['spei_03_summer_spei_1819'] = filtered_df['spei_03_summer_spei_2018'] + filtered_df['spei_03_summer_spei_2019']
        filtered_df['spei_03_autumn_spei_1819'] = filtered_df['spei_03_autumn_spei_2018'] + filtered_df['spei_03_autumn_spei_2019']
        filtered_df['spei_06_winterhalf_spei_1819'] = filtered_df['spei_06_winter_half_spei_2018'] + filtered_df['spei_06_winter_half_spei_2019']

        # 处理一下列名的差别
        for item in [col for col in df.columns if 'summer_half' in col]:
            filtered_df.rename(columns={item: item.replace('summer_half','summerhalf')}, inplace=True)
        for item in [col for col in df.columns if 'winter_half' in col]:
            filtered_df.rename(columns={item: item.replace('winter_half','winterhalf')}, inplace=True)

        # 构建复合干旱指标
        filtered_df['compound_spei_1819'] = 0
        filtered_df['compound_spei_181922'] = 0
        for item in ['spring', 'summer', 'autumn', 'winterhalf']:
            speiscale = '03'
            if item == 'winterhalf': speiscale = '06'
            filtered_df[f'compound_spei_1819'] = filtered_df[f'compound_spei_1819'] + filtered_df[f'spei_{speiscale}_{item}_spei_2018'] + filtered_df[f'spei_{speiscale}_{item}_spei_2019']
        for item in ['spring', 'summer','autumn', 'winterhalf']:
            speiscale = '03'
            if item == 'winterhalf': speiscale = '06'
            filtered_df[f'compound_spei_181922'] = filtered_df[f'compound_spei_181922'] + filtered_df[f'spei_{speiscale}_{item}_spei_2018'] + filtered_df[f'spei_{speiscale}_{item}_spei_2019'] + filtered_df[f'spei_{speiscale}_{item}_spei_2022']

        # 标准化
        exclude_columns = ['Unnamed: 0','row', 'col','row.1', 'col.1','weights','country',
                           'sos_baseline_prehalf','eos_baseline_prehalf','sos_baseline_posthalf','eos_baseline_posthalf',
                           'sos_baseline_whole','eos_baseline_whole']
        exclude_columns += sos_columns
        exclude_columns += eos_columns
        columns_to_standardize = [col for col in filtered_df.columns if col not in exclude_columns]
        filtered_df_normalization = filtered_df.copy()
        for col in columns_to_standardize:
            filtered_df_normalization[col] = (filtered_df_normalization[col] - filtered_df_normalization[col].mean()) / filtered_df_normalization[col].std()

        filtered_df_stack = self.stack_data(filtered_df_normalization,drought_year,legacy_end_year)
        '''分析GDD和TM之间的相关性'''
        # corr = filtered_df_stack['gdd'].corr(filtered_df_stack['tm_chilling'])
        # print(f'GDD和TM之间的相关性: {corr}')
        # # 可视化散点图和拟合线
        # sns.lmplot(x='gdd', y='tm_chilling', data=filtered_df_stack)
        # plt.title(f'GDD和TM的相关性：{corr:.2f}')
        # plt.show()

        response_var_sos = f'sos'
        gdd_var = f'gdd'
        vpd_var = f'vpd'
        spei_drought_spring_var = 'spei_spring_lag'
        spei_drought_summer_var = 'spei_summer_lag'
        spei_drought_autumn_var = 'spei_autumn_lag'
        spei_drought_winterhalf_var = 'spei_winterhalf_lag'
        spei_drought_spring_var2 = 'spei_spring_lag2'
        spei_drought_summer_var2 = 'spei_summer_lag2'
        spei_drought_autumn_var2 = 'spei_autumn_lag2'
        spei_drought_winterhalf_var2 = 'spei_winterhalf_lag2'
        tm_chillingdays_var= f'chillingdays'
        # response_var_sos = f'sos_{drought_year+1}'
        # gdd_var = f'gdd_fixed_sum_{drought_year+1}'
        # vpd_var = f'VPD_MA_avg_{drought_year+1}'
        # spei_drought_spring_var = f'spei_03_spring_spei_{drought_year}'
        # spei_drought_summer_var = f'spei_03_summer_spei_{drought_year}'
        # spei_drought_autumn_var = f'spei_03_autumn_spei_{drought_year}'
        # spei_drought_winterhalf_var = f'spei_06_winterhalf_spei_{drought_year}'
        # tm_chilling_var= f'tm_chilling_{drought_year}'
        formula = (f'{response_var_sos} ~  {gdd_var} + {vpd_var} + {spei_drought_spring_var}+ {spei_drought_summer_var} '
                   f'+ {spei_drought_autumn_var} + {spei_drought_winterhalf_var} + {gdd_var}*{spei_drought_summer_var}')
        '''权重归一化
        # weights = filtered_df_stack['weights']
        # # Reshape weights to a 2D array for MinMaxScaler (if weights is a pandas Series)
        # if isinstance(weights, pd.Series):
        #     weights = weights.values.reshape(-1, 1)
        # # Normalize weights using MinMaxScaler
        # scaler = MinMaxScaler()
        # normalized_weights = scaler.fit_transform(weights).flatten()  # Flatten to a 1D array
        '''
        model = glm(formula, data=filtered_df_stack, family=sm.families.Gaussian())
        # results = model.fit()
        results = model.fit(weights=filtered_df_stack['weights'])
        print(results.summary())  # 保存结果
        print(results.params)  # 提取模型的系数
        spei_vars = [spei_drought_spring_var, spei_drought_summer_var, spei_drought_autumn_var,spei_drought_winterhalf_var]
        # spei_vars = [spei_drought_spring_var, spei_drought_summer_var, spei_drought_autumn_var,spei_drought_winterhalf_var,spei_drought_spring_var2, spei_drought_summer_var2, spei_drought_autumn_var2,spei_drought_winterhalf_var2]
        self.different_model_test(spei_vars,response_var_sos,gdd_var, vpd_var,
                                  filtered_df_stack,drought_year,'SOS_Single_Drought',
                                  model_results_save_path=f'temp\statistical_results(SOS)\Single_drought_metrics/{drought_year}/model_results.txt')
        '''# 计算边际效应marginal effects'''
        # gdd_values = [filtered_df_stack[gdd_var].quantile(0.25), filtered_df_stack[gdd_var].median(),
        #               filtered_df_stack[gdd_var].quantile(0.75)]
        # interaction_items = {
        #     'control':{'variable name':gdd_var,'show name':'GDD'},
        #     'marginal':{'variable name':spei_drought_summer_var,'show name':'SPEI Summer'}
        # }
        # spei_range = np.linspace(filtered_df_stack[interaction_items['marginal']['variable name']].min(),
        #                                 filtered_df_stack[interaction_items['marginal']['variable name']].max(), 100)
        # pred_df = pd.DataFrame({
        #     'spei_range':spei_range,        #因为后面的都是一个标量值（即单个值），而 pandas 要求在这种情况下必须提供一个 index，麻烦，所以加一个等长的spei_range,实际上也是一样的
        #     vpd_var: filtered_df_stack[vpd_var].median(),
        #     spei_drought_spring_var: filtered_df_stack[spei_drought_spring_var].median(),
        #     spei_drought_autumn_var: filtered_df_stack[spei_drought_autumn_var].median(),
        #     spei_drought_winterhalf_var: filtered_df_stack[spei_drought_winterhalf_var].median()
        # })
        # output_path = r'test.jpg'
        # self.marginal_effects_calculation(filtered_df_stack,results,gdd_values,interaction_items,pred_df,output_path)
        # # 选择GDD值的分位数
        # gdd_values = [filtered_df_stack[gdd_var].quantile(0.25), filtered_df_stack[gdd_var].median(),
        #               filtered_df_stack[gdd_var].quantile(0.75)]
        #
        # # 绘制边际效应图
        # plt.figure(figsize=(10, 6))
        # colors = ['blue', 'orange', 'red']  # 分别为低、中、高GDD条件的颜色
        # # 对每个GDD值，计算SPEI_summer在不同条件下的边际效应
        # for i, gdd in enumerate(gdd_values):
        #     beta_2 = results.params[spei_drought_summer_var]
        #     beta_3 = results.params[f'{gdd_var}:{spei_drought_summer_var}']
        #     marginal_effects = beta_2 + beta_3 * gdd
        #
        #     # 计算不同SPEI_summer值下的预测SOS值
        #     spei_summer_range = np.linspace(filtered_df_stack[spei_drought_summer_var].min(),
        #                                     filtered_df_stack[spei_drought_summer_var].max(), 100)
        #     # 创建数据集来计算预测值和置信区间
        #     pred_df = pd.DataFrame({
        #         gdd_var: gdd,
        #         spei_drought_summer_var: spei_summer_range,
        #         vpd_var: filtered_df_stack[vpd_var].median(),
        #         tm_chillingdays_var: filtered_df_stack[tm_chillingdays_var].median(),
        #         spei_drought_spring_var: filtered_df_stack[spei_drought_spring_var].median(),
        #         spei_drought_autumn_var: filtered_df_stack[spei_drought_autumn_var].median(),
        #         spei_drought_winterhalf_var: filtered_df_stack[spei_drought_winterhalf_var].median()
        #     })
        #
        #     # 获取预测值和置信区间
        #     predictions = results.get_prediction(pred_df)
        #     pred_mean = predictions.predicted_mean
        #     pred_ci = predictions.conf_int()
        #     # 绘制预测值曲线
        #     plt.plot(spei_summer_range, pred_mean, label=f'GDD = {gdd:.2f} (Marginal Effect: {marginal_effects:.2f})',
        #              color=colors[i])
        #
        #     # 绘制置信区间，阴影更深
        #     plt.fill_between(spei_summer_range, pred_ci[:, 0], pred_ci[:, 1], color=colors[i], alpha=0.4)
        #
        # # 图形设置
        # plt.xlabel('SPEI Summer')
        # plt.ylabel('SOS')
        # plt.title('Marginal Effect of SPEI Summer on SOS under Different GDD Conditions')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(os.path.join(f'temp/statistical_results/{drought_year}','marginal_effects.jpg'))
        '''分离因子测试'''
        # # 因子列表
        # factors = [
        #     gdd_var,
        #     vpd_var,
        #     tm_chillingdays_var,
        #     spei_drought_spring_var,
        #     spei_drought_summer_var,
        #     spei_drought_autumn_var,
        #     spei_drought_winterhalf_var
        # ]
        # self.statistical_variables_seprating_test(factors,response_var_sos,filtered_df_stack,f'{gdd_var} * {spei_drought_summer_var}')
        '''复合影响'''
        response_var_sos2020 = f'sos_2020'
        gdd_var2020 = f'gdd_fixed_sum_2020'
        vpd_var2020 = f'VPD_MA_avg_2020'
        chillingdays_var2019 = f'chillingdays_2019'
        tm_summer_var2020 = f'tm_summer_2020'
        tm_summerhalf_var2020 = f'tm_summerhalf_2020'
        response_var_sos2023 = f'sos_2023'
        gdd_var2023 = f'gdd_fixed_sum_2023'
        vpd_var2023 = f'VPD_MA_avg_2023'
        chillingdays_var2022 = f'chillingdays_2022'
        tm_summer_var2023 = f'tm_summer_2023'
        tm_summerhalf_var2023 = f'tm_summerhalf_2023'
        # SPEI变量定义
        spei_drought_vars_2018 = [
            'spei_03_spring_spei_2018',
            'spei_03_summer_spei_2018',
            'spei_03_autumn_spei_2018',
            'spei_06_winterhalf_spei_2018'
        ]

        spei_drought_vars_2019 = [
            'spei_03_spring_spei_2019',
            'spei_03_summer_spei_2019',
            'spei_03_autumn_spei_2019',
            'spei_06_winterhalf_spei_2019'
        ]

        spei_drought_vars_1819 = [
            'spei_03_spring_spei_1819',
            'spei_03_summer_spei_1819',
            'spei_03_autumn_spei_1819',
            'spei_06_winterhalf_spei_1819'
        ]

        spei_drought_vars_2022 = [
            'spei_03_spring_spei_2022',
            'spei_03_summer_spei_2022',
            'spei_03_autumn_spei_2022',
            'spei_06_winterhalf_spei_2022'
        ]

        spei_compound_1819 = 'compound_spei_1819'
        spei_compound_181922 = 'compound_spei_181922'
        spei_vars = spei_drought_vars_2018+spei_drought_vars_2019+[spei_compound_1819]
        # self.different_model_test(spei_vars,response_var_sos2020,gdd_var2020, vpd_var2020,
        #                           filtered_df_normalization,1819,'SOS_Multi_Drought',
        #                           model_results_save_path=f'temp\statistical_results(SOS)\Multi_drought_metrics/1819/model_results.txt')
        spei_vars = spei_drought_vars_2018+spei_drought_vars_2019+spei_drought_vars_2022+[spei_compound_181922]

        self.different_model_test(spei_vars,response_var_sos2023,gdd_var2023, vpd_var2023,
                                  filtered_df_normalization,181922,'SOS_Multi_Drought',
                                  model_results_save_path=f'temp\statistical_results(SOS)\Multi_drought_metrics/181922/model_results.txt')

        # 复合干旱深入，compoun_spei改为spei18*spei19*spei22
        # 1819干旱
        base_formula = r'sos_2020 ~ G + V + Sp18 + Au18 + Wi18 + Sp19 + Au19 + Co1819 + G*Sp19'
        base_formula = self.expand_formula(base_formula).replace('vpd','VPD_MA_avg_2020')
        gdd_replacement = 'gdd_fixed_sum_2020'
        # self.different_model_test_for_multidrought(base_formula,filtered_df_normalization
        #     ,f'temp\statistical_results(SOS)\Multi_drought_metrics/1819/deeper/model_results.txt',gdd_replacement)
        #     181922干旱
        base_formula = r'sos_2023 ~ G + V + Sp18 + Su18 + Au18 + Wi18 + Sp19 + Au19 + Wi19 + Sp22 + Su22 + Au22 + G*Sp19'
        base_formula = self.expand_formula(base_formula).replace('vpd','VPD_MA_avg_2023')
        gdd_replacement = 'gdd_fixed_sum_2023'
        # self.different_model_test_for_multidrought(base_formula,filtered_df_normalization
        #     ,f'temp\statistical_results(SOS)\Multi_drought_metrics/181922/deeper/model_results.txt',gdd_replacement)


        '''SOS时序建模'''
        spei_vars = [spei_drought_spring_var, spei_drought_summer_var, spei_drought_autumn_var,
                     spei_drought_winterhalf_var]
        filtered_df_stack_time_series = self.stack_data(filtered_df_normalization, 2003, 2023)
        # self.different_model_test(spei_vars, response_var_sos, gdd_var, vpd_var,
        #                           filtered_df_stack_time_series, 9999, 'SOS_Temporal_Series',
        #                           model_results_save_path=f'temp\statistical_results(SOS)\Temporal_Series/model_results.txt')

        '''EOS建模'''
        response_var_eos_d = 'eos_drought_year'
        tm_summer_var_d = 'tm_summer_drought_year'
        tm_summerhalf_var_d = 'tm_summerhalf_drought_year'
        sos_var_d = 'sos_drought_year'
        gdd_var_d = f'gdd_drought_year'
        vpd_var_d = f'vpd_drought_year'
        spei_drought_spring_var_d = 'spei_spring_drought_year'
        spei_drought_summer_var_d = 'spei_summer_drought_year'
        spei_drought_autumn_var_d = 'spei_autumn_drought_year'

        spei_spring,spei_summer,spei_autumn = 'spei_spring','spei_summer','spei_autumn'

        response_var_eos = 'eos'
        tm_summer_var = 'tm_summer'
        tm_summerhalf_var = 'tm_summerhalf'
        sos_var = 'sos'
        filtered_df_normalization_eos = filtered_df_normalization.copy()
        for col in [item for item in filtered_df_normalization_eos.columns if 'sos' in item]:
            filtered_df_normalization_eos[col] = (filtered_df_normalization_eos[col] - filtered_df_normalization_eos[col].mean()) / filtered_df_normalization_eos[col].std()
        filtered_df_stack_eos = self.stack_data(filtered_df_normalization_eos, drought_year, legacy_end_year)
        spei_vars = [spei_drought_spring_var, spei_drought_summer_var, spei_drought_autumn_var,
                     spei_drought_winterhalf_var, spei_drought_spring_var2, spei_drought_summer_var2,
                     spei_drought_autumn_var2, spei_drought_winterhalf_var2]
        # self.different_model_test(spei_vars,response_var_eos,gdd_var, vpd_var,
        #                           filtered_df_stack_eos,drought_year,'EOS_Single_Drought',
        #                          f'temp\statistical_results(EOS)\Single_drought_metrics/{drought_year}/model_results.txt',
        #                           sos_var,tm_summer_var,tm_summerhalf_var)

        # eos干旱年建模
        spei_vars_d = [spei_drought_spring_var_d, spei_drought_summer_var_d, spei_drought_autumn_var_d]
        # self.different_model_test(spei_vars_d,response_var_eos_d,gdd_var_d, vpd_var_d,
        #                           filtered_df_stack_eos,drought_year,'EOS_Single_Drought',
        #                          f'temp\statistical_results(EOS)\Single_drought_metrics/{drought_year}/model_results.txt',
        #                           sos_var_d,tm_summer_var_d,tm_summerhalf_var_d)


        # 复合干旱EOS建模
        sos_var2020 = 'sos_2020'
        sos_var2023 = 'sos_2023'
        response_var_eos2020 = 'eos_2020'
        response_var_eos2023 = 'eos_2023'
        spei_vars = spei_drought_vars_2018+spei_drought_vars_2019+[spei_compound_1819]
        # self.different_model_test(spei_vars,response_var_eos2020,gdd_var2020, vpd_var2020,
        #                           filtered_df_normalization_eos,1819,'EOS_Multi_Drought',
        #                           f'temp\statistical_results(EOS)\Multi_drought_metrics/1819/model_results.txt',
        #                           sos_var2020,tm_summer_var2020,tm_summerhalf_var2020)
        spei_vars = spei_drought_vars_2018+spei_drought_vars_2019+spei_drought_vars_2022+[spei_compound_181922]
        # self.different_model_test(spei_vars,response_var_eos2023,gdd_var2023, vpd_var2023,
        #                           filtered_df_normalization_eos,181922,'EOS_Multi_Drought',
        #                           f'temp\statistical_results(EOS)\Multi_drought_metrics/181922/model_results.txt',
        #                           sos_var2023,tm_summer_var2023,tm_summerhalf_var2023)


        # 时序EOS建模
        filtered_df_stack_eos_time_series = self.stack_data(filtered_df_normalization_eos, 2002, 2023)
        # self.different_model_test(spei_vars, response_var_sos, gdd_var, vpd_var,
        #                           filtered_df_stack_eos_time_series, 9999, 'SOS_Temporal_Series',
        #                           f'temp\statistical_results(SOS)\Temporal_Series/model_results.txt',
        #                           sos_var,tm_summer_var,tm_summerhalf_var)


        # 时序EOS干旱年建模
        spei_vars_d = [spei_spring, spei_summer, spei_autumn]
        self.different_model_test(spei_vars_d,response_var_eos,gdd_var, vpd_var,
                                  filtered_df_stack_eos_time_series,drought_year,'EOS_Temporal_Series',
                                 f'temp\statistical_results(EOS)\Temporal_Series/model_results.txt',
                                  sos_var,tm_summer_var,tm_summerhalf_var)

    def statistical_results_analysis(self,data_path,top_number):

        ranking_data = pd.read_csv(data_path)
        # get top formulas of different metrics
        top_aic = ranking_data.nsmallest(top_number, 'AIC')
        top_bic = ranking_data.nsmallest(top_number, 'BIC')
        top_r2 = ranking_data.nlargest(top_number, 'Pseudo R-squared')

        # 从公式中提取变量，排除 G 和 V
        def extract_variables(formula):
            variables = formula.split("~")[1].replace("G", "").replace("V", "").replace("+", "").replace("*","").split()
            return [v.strip() for v in variables if v.strip()]

        def extract_variables_without_interaction_item(formula):
            terms = formula.split("~")[1].split("+")
            # 只保留不包含 '*' 的单一变量项，并去除 G 和 V
            variables = [term.strip() for term in terms if '*' not in term and term.strip() not in ["G", "V"]]
            return variables
        # 提取单一的spei变量
        def extract_variables_set():
            return set([var for formula in ranking_data['Formula'] for var in extract_variables(formula)])

        def visualize_counter_data(counters, output_path,labels=['AIC', 'BIC','Pseudo R-squared']):
            data = pd.DataFrame(counters)
            data = data.T  # 转置，使变量在索引（行）上
            data.columns = labels  # 设置列标签
            # 配置图表外观
            sns.set(style="whitegrid")
            plt.figure(figsize=(10, 6), dpi=300)  # 高分辨率图表
            # 绘制柱状图
            data.plot(kind='bar', width=0.8, edgecolor='black', colormap='coolwarm', ax=plt.gca())
            # 设置标签、标题、字体大小等
            plt.title('Frequency of Variables across Different Rankings', fontsize=16, fontweight='bold')
            plt.xlabel('Variables', fontsize=14, fontweight='bold')
            plt.ylabel('Frequency', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(title='Ranking', title_fontsize='13', fontsize=12)
            # 调整布局
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        aic_variables = Counter([var for formula in top_aic['Formula'] for var in extract_variables(formula)])
        bic_variables = Counter([var for formula in top_bic['Formula'] for var in extract_variables(formula)])
        pseudo_r2_variables = Counter([var for formula in top_r2['Formula'] for var in extract_variables(formula)])
        visualize_counter_data([aic_variables, bic_variables, pseudo_r2_variables],os.path.join(os.path.split(data_path)[0],'all_ranking_plot.jpg'))
        spei_vars = extract_variables_set()
        for r in range(1, len(spei_vars) + 1):
            filtered_df = ranking_data[
                ranking_data['Formula'].apply(lambda x: len(extract_variables_without_interaction_item(x)) == r)]
            formula_subset_df = filtered_df[['Formula', 'AIC', 'BIC', 'Pseudo R-squared']]
            top_aic_subset = formula_subset_df.nsmallest(min(top_number,formula_subset_df.shape[0]), 'AIC')
            top_bic_subset = formula_subset_df.nsmallest(min(top_number,formula_subset_df.shape[0]), 'BIC')
            top_r2_subset = formula_subset_df.nlargest(min(top_number,formula_subset_df.shape[0]), 'Pseudo R-squared')
            aic_variables_subset = Counter([var for formula in top_aic_subset['Formula'] for var in extract_variables(formula)])
            bic_variables_subset = Counter([var for formula in top_bic_subset['Formula'] for var in extract_variables(formula)])
            pseudo_r2_variables_subset = Counter([var for formula in top_r2_subset['Formula'] for var in extract_variables(formula)])
            visualize_counter_data([aic_variables_subset, bic_variables_subset, pseudo_r2_variables_subset],
                                   os.path.join(os.path.split(data_path)[0], f'subset_ranking_plot{r}.jpg'))


    def statistical_modelling_pure(self,data_path,drought_year,legacy_end_year):
        '''
        上面的统计模型中变换干旱区域非干旱区域选择方法
        上面的是单一选择，这个是复合选择，以春季干旱为例
        上面的会选择春季spei小于阈值的地方
        这个会选择春季spei小于阈值，夏季，秋季大于阈值，这个阈值是各自季节对应的阈值，全年的画两个是一样的
        :param data_path:
        :param drought_year:
        :return:
        '''
        df = pd.read_csv(data_path).dropna()
        # 过滤异常值
        sos_columns = [col for col in df.columns if 'sos' in col]
        eos_columns = [col for col in df.columns if 'eos' in col]
        sos_values = df[sos_columns].values
        eos_values = df[eos_columns].values
        # Calculate invalid SOS/EOS pairs
        invalid_sos_eos_counts = np.sum(sos_values >= eos_values, axis=1)
        # Calculate invalid data points
        combined_values = np.concatenate([sos_values, eos_values], axis=1)
        invalid_data_points_counts = np.sum((combined_values == -32768) | (combined_values == 261), axis=1)
        # Filter the rows
        valid_rows = (invalid_sos_eos_counts == 0) & (invalid_data_points_counts == 0)
        filtered_df = df[valid_rows].copy()
        # 处理一下列名的差别
        for item in [col for col in df.columns if 'summer_half' in col]:
            filtered_df.rename(columns={item: item.replace('summer_half','summerhalf')}, inplace=True)
        for item in [col for col in df.columns if 'winter_half' in col]:
            filtered_df.rename(columns={item: item.replace('winter_half','winterhalf')}, inplace=True)
        # 对GDD和SPEI进行纵向标准化或者横向标准化，目前先进行纵向标准化
        # 待分析因素字典
        factors_waiting = {'03':['spring','summer','autumn','annual',],
                           '06':['summerhalf','winterhalf','annual',]}

        threshold_collections = {
            '03':{
                'spring':filtered_df[f'spei_03_spring_spei_{drought_year}'][filtered_df[f'spei_03_spring_spei_{drought_year}']>filtered_df[f'spei_03_spring_spei_{drought_year}'].min()].quantile(0.25),
                'summer':filtered_df[f'spei_03_summer_spei_{drought_year}'][filtered_df[f'spei_03_summer_spei_{drought_year}']>filtered_df[f'spei_03_summer_spei_{drought_year}'].min()].quantile(0.25),
                'autumn':filtered_df[f'spei_03_autumn_spei_{drought_year}'][filtered_df[f'spei_03_autumn_spei_{drought_year}']>filtered_df[f'spei_03_autumn_spei_{drought_year}'].min()].quantile(0.25),
                'annual':filtered_df[f'spei_03_annual_spei_{drought_year}'][filtered_df[f'spei_03_annual_spei_{drought_year}']>filtered_df[f'spei_03_annual_spei_{drought_year}'].min()].quantile(0.25)
            },
            '06':{
                'summerhalf':filtered_df[f'spei_06_summerhalf_spei_{drought_year}'][filtered_df[f'spei_06_summerhalf_spei_{drought_year}']>filtered_df[f'spei_06_summerhalf_spei_{drought_year}'].min()].quantile(0.25),
                'winterhalf': filtered_df[f'spei_06_winterhalf_spei_{drought_year}'][filtered_df[f'spei_06_winterhalf_spei_{drought_year}']>filtered_df[f'spei_06_winterhalf_spei_{drought_year}'].min()].quantile(0.25),
                'annual':filtered_df[f'spei_06_annual_spei_{drought_year}'][filtered_df[f'spei_06_annual_spei_{drought_year}']>filtered_df[f'spei_06_annual_spei_{drought_year}'].min()].quantile(0.25)

            }
        }
        for spei_scale in factors_waiting.keys():
            drought_timings = factors_waiting[spei_scale]
            for drought_timing in drought_timings:
                gdd_columns = [col for col in df.columns if 'gdd_fixed_sum' in col]
                spei_columns = [col for col in df.columns if 'spei_{}_{}'.format(spei_scale,drought_timing) in col]
                gdd_values = filtered_df[gdd_columns].values
                spei_values = filtered_df[spei_columns].values

                drought_year_spei = filtered_df[f'spei_{spei_scale}_{drought_timing}_spei_{drought_year}']
                drought_filtered_df,undrought_filtered_df = self.pure_data_df(filtered_df,drought_year,drought_timing,drought_year_spei,spei_scale)
                results_drought_area = {}
                results_undrought_area = {}
                random_effects_var = 'country'

                # 创建空字典来存储每年的系数
                drought_coef = {}
                undrought_coef = {}
                response_vars = ['sos', 'eos']
                tm_vars = ['gdd', 'tm']
                for response_item in response_vars:
                    for tm_item in tm_vars:
                        for year in range(drought_year+1,legacy_end_year):
                            response_var_sos = f'sos_{year}'
                            response_var_eos = f'eos_{year}'
                            gdd_var = f'gdd_fixed_sum_{year}'
                            vpd_var = f'VPD_{drought_timing}_{year}'
                            tm_var= f'tm_{drought_timing}_{year}'
                            spei_drought_var = f'spei_{spei_scale}_{drought_timing}_spei_{drought_year}'
                            formula_collections_sos = {
                                'gdd': f'{response_var_sos} ~ {spei_drought_var} + {gdd_var} + {vpd_var} + {spei_drought_var}*{vpd_var}'
                                ,
                                'tm':f'{response_var_sos} ~ {spei_drought_var} + {tm_var} + {vpd_var} + {spei_drought_var}*{vpd_var}'

                            }
                            formula_collections_eos = {
                                'gdd': f'{response_var_eos} ~ {spei_drought_var} + {gdd_var}+ {vpd_var} + {spei_drought_var}*{vpd_var}'
                                ,
                                'tm':f'{response_var_eos} ~ {spei_drought_var} + {tm_var}+ {vpd_var} + {spei_drought_var}*{vpd_var}'
                            }
                            if response_item == 'sos': formula = formula_collections_sos[tm_item]
                            if response_item == 'eos': formula = formula_collections_eos[tm_item]
                            print(response_var_sos,f'spei_{spei_scale}_{drought_timing}_spei_{drought_year}',gdd_var,tm_var,formula)
                            #
                            # model_drought = glm(formula,data = drought_filtered_df,family = sm.families.Gaussian())
                            # results_drought = model_drought.fit()
                            # results_drought_area[year] = results_drought.summary()  # 保存结果
                            # drought_coef[year] = results_drought.params  # 提取模型的系数
                            # 混合模型公式
                            # 构建混合效应模型
                            model_drought = MixedLM.from_formula(formula,
                                                         data=drought_filtered_df,
                                                         groups=drought_filtered_df[random_effects_var])

                            results_drought = model_drought.fit()
                            results_drought_area[year] = results_drought.summary()
                            drought_coef[year] = results_drought.params  # 提取模型的系数


                        # for year, result in results_drought_area.items():
                        #     print(f"Year {year}:")
                        #     print(result)
                        #     print("\n" + "="*60 + "\n")


                        for year in range(drought_year+1,legacy_end_year):
                            response_var_sos = f'sos_{year}'
                            response_var_eos = f'eos_{year}'
                            gdd_var = f'gdd_fixed_sum_{year}'
                            vpd_var = f'VPD_{drought_timing}_{year}'
                            tm_var = f'tm_{drought_timing}_{year}'
                            spei_drought_var = f'spei_{spei_scale}_{drought_timing}_spei_{drought_year}'
                            formula_collections_sos = {
                                'gdd': f'{response_var_sos} ~ {spei_drought_var} + {gdd_var} + {vpd_var} + {spei_drought_var}*{vpd_var}'
                                ,
                                'tm':f'{response_var_sos} ~ {spei_drought_var} + {tm_var} + {vpd_var} + {spei_drought_var}*{vpd_var}'

                            }
                            formula_collections_eos = {
                                'gdd': f'{response_var_eos} ~ {spei_drought_var} + {gdd_var} + {vpd_var} + {spei_drought_var}*{vpd_var}'
                                ,
                                'tm':f'{response_var_eos} ~ {spei_drought_var} + {tm_var} + {vpd_var} + {spei_drought_var}*{vpd_var}'
                            }
                            if response_item == 'sos': formula = formula_collections_sos[tm_item]
                            if response_item == 'eos': formula = formula_collections_eos[tm_item]
                            print(response_var_sos,f'spei_{spei_scale}_{drought_timing}_spei_{drought_year}',gdd_var,tm_var,formula)

                            # model_undrought = glm(formula,data = undrought_filtered_df,family = sm.families.Gaussian())
                            # results_undrought = model_undrought.fit()
                            # results_undrought_area[year] = results_undrought.summary()  # 保存结果
                            # undrought_coef[year] = results_undrought.params  # 提取模型的系数
                            # 构建混合效应模型
                            model_undrought = MixedLM.from_formula(formula,
                                                         data=undrought_filtered_df,
                                                         groups=undrought_filtered_df[random_effects_var])

                            results_undrought = model_undrought.fit()
                            results_undrought_area[year] = results_undrought.summary()
                            undrought_coef[year] = results_undrought.params  # 提取模型的系数
                        # for year, result in results_undrought_area.items():
                        #     print(f"Year {year}:")
                        #     print(result)
                        #     print("\n" + "="*60 + "\n")
                        # 保存干旱区域的结果到txt文件
                        root_path = f'temp/statistical_results/{response_item}/spei_{spei_scale}/{tm_item}'
                        if not os.path.exists(root_path):os.makedirs(root_path)
                        with open(f'temp/statistical_results/{response_item}/spei_{spei_scale}/{tm_item}/spei{spei_scale}_time{drought_timing}_drought_area_results.txt', 'w') as f:
                            for year, result in results_drought_area.items():
                                f.write(f"Year {year}:\n")
                                f.write(result.as_text())
                                f.write("\n" + "="*60 + "\n")

                        # 保存非干旱区域的结果到txt文件
                        with open(f'temp/statistical_results/{response_item}/spei_{spei_scale}/{tm_item}/spei{spei_scale}_time{drought_timing}_undrought_area_results.txt', 'w') as f:
                            for year, result in results_undrought_area.items():
                                f.write(f"Year {year}:\n")
                                f.write(result.as_text())
                                f.write("\n" + "="*60 + "\n")

                        # 转换为DataFrame以便绘图，仅保留'spei_03_spring_spei_2003'变量
                        drought_coef_df = pd.DataFrame(drought_coef).T
                        undrought_coef_df = pd.DataFrame(undrought_coef).T

                        # 绘制干旱区域和非干旱区域的'spei_03_spring_spei_2003'系数变化
                        plt.figure(figsize=(12, 6))

                        # 干旱区域曲线
                        plt.plot(drought_coef_df.index, drought_coef_df[f'spei_{spei_scale}_{drought_timing}_spei_{drought_year}'],
                                 label=f'Drought spei_{spei_scale}_{drought_timing}_spei_{drought_year}', linewidth=2.5, color='red')

                        # 非干旱区域曲线
                        plt.plot(undrought_coef_df.index, undrought_coef_df[f'spei_{spei_scale}_{drought_timing}_spei_{drought_year}'],
                                 label=f'Non-Drought spei_{spei_scale}_{drought_timing}_spei_{drought_year}', linewidth=2.5, color='blue')

                        plt.xlabel('Year')
                        plt.ylabel('Coefficient')
                        plt.title('Coefficient Changes Over Time (Drought vs Non-Drought Areas)')
                        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # 图例放置在图表旁边
                        plt.grid(True)
                        plt.tight_layout()  # 自动调整子图参数，适应图例等
                        plt.savefig(f'temp/statistical_results/{response_item}/spei_{spei_scale}/{tm_item}/spei{spei_scale}_time{drought_timing}_curve.jpg')
                        # drought_coef = {}
                        # undrought_coef = {}
                        # results_drought_area = {}
                        # results_undrought_area = {}
    def calculate_sos_recovery_year(self,data, drought_year,baseline_years,legacy_end_year_max):
        """
        计算每个像素点的SOS恢复年份，并返回所有像素点恢复年份的均值。

        参数:
        - data: 包含SOS数据的DataFrame，每一行是一个像素点，列名格式为 'sos_年份'
        - drought_year: 干旱发生的年份
        - baseline_years: 用于计算基线水平的年份列

        返回:
        - average_recovery_year: 所有像素点恢复年份的均值
        """
        # 获取基线年份的平均值，作为基线水平
        baseline_values = data[baseline_years].mean(axis=1)

        # 计算基线水平的 95% 范围
        lower_bound = baseline_values * 0.95
        upper_bound = baseline_values * 1.05

        # 定义要检查的年份列，从干旱年份后开始
        post_drought_years = [col for col in baseline_years if int(col.split('_')[1]) > drought_year]

        # 用于保存每一行（像素点）恢复的年份
        recovery_years = []

        # 遍历每个像素点（每一行数据）
        for i, row in data.iterrows():
            baseline = baseline_values[i]
            lower = lower_bound[i]
            upper = upper_bound[i]

            # 在干旱年之后，找到第一个恢复到95%范围内的年份
            recovery_year = np.nan  # 默认设置为 NaN，表示未恢复
            for year_col in post_drought_years:
                sos_value = row[year_col]
                if lower <= sos_value <= upper:
                    recovery_year = int(year_col.split('_')[1])  # 获取年份
                    break

            recovery_years.append(recovery_year)

        return min(int(np.mean(np.array(recovery_years)[~np.isnan(recovery_years)])),legacy_end_year_max)

    def calculate_sos_recovery_year_pacf(self,data, drought_year,legacy_end_year_max,spei_scale,drought_timing):
        """
        根据pacf计算遗留效应结束的年

        """
        sos_columns = [col for col in data.columns if 'sos' in col and 'baseline' not in col]
        eos_columns = [col for col in data.columns if 'eos' in col and 'baseline' not in col]
        sos_data = data[sos_columns]

        # 去趋势处理
        sos_detrended = sos_data.apply(lambda x: signal.detrend(x), axis=1)
        detrended_df = pd.DataFrame(sos_detrended.tolist())
        detrended_df.columns = [f'detrended_sos_{i}' for i in range(2001, 2024)]
        speis = data[[f'spei_{spei_scale}_{drought_timing}_spei_{i}' for i in range(2001,2024)]]
        speis_detrend = speis.apply(lambda x: signal.detrend(x), axis=1)
        speis_detrend_df = pd.DataFrame(speis_detrend.tolist())
        speis_detrend_df.columns = [f'spei_{spei_scale}_{drought_timing}_spei_{i}' for i in range(2001, 2024)]
        # 初始化一个空列表，用于存储相关系数
        correlations = []
        p_values = []

        for year in range(drought_year +1, legacy_end_year_max):
            corr, p_value = pearsonr(speis_detrend_df[f'spei_{spei_scale}_{drought_timing}_spei_{drought_year}'], detrended_df[f'detrended_sos_{year}'])
            correlations.append(corr)
            p_values.append(p_value)
        # 输出每年的相关性系数和对应的p值
        for year, corr, p_value in zip(range(drought_year +1, legacy_end_year_max), correlations, p_values):
            print(f"Year: {year}, Correlation: {corr:.4f}, p-value: {p_value:.4f}")

        # 绘制相关性随年份的变化趋势
        # years = [year for year in range(drought_year +1, legacy_end_year_max)]  # 提取年份
        # plt.plot(years, correlations, marker='o')
        # plt.axhline(0, color='r', linestyle='--')  # 绘制y=0的参考线
        # plt.title('Correlation between 2003 SPEI and Subsequent SOS Values')
        # plt.xlabel('Year')
        # plt.ylabel('Correlation')
        # plt.show()
        # return None
    def random_forest(self,target,features,df,u_df,year,drought_timing,spei_scale,drought_year,if_drought,if_visualization):

        # 定义保存图片路径
        if if_drought:output_path = os.path.join(r'temp\random_forest/',f'drought{drought_year}/spei_{spei_scale}/{drought_timing}/drought_area')
        else:output_path = os.path.join(r'temp\random_forest/',f'drought{drought_year}/spei_{spei_scale}/{drought_timing}/undrought_area')
        if not os.path.exists(output_path):os.makedirs(output_path)
        df_new = df.copy()
        X = df_new[features]
        y = df_new[target]
        df_new[f'spei_drought_lag'] = X[f'spei_{spei_scale}_{drought_timing}_spei_{drought_year}'] * (year - drought_year)
        features_new = features.copy()
        features_new[0] = 'spei_drought_lag'
        X = df_new[features_new]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 创建和训练随机森林模型
        rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
        rf_model.fit(X_train, y_train)
        # 预测和评估模型
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # 绘制散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue', edgecolor='k', s=100)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)  # 添加 y=x 的参考线

        # 添加 MSE 和 R2 标签
        plt.text(0.05, 0.95, f'MSE: {mse:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.05, 0.90, f'R²: {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        # 添加轴标签和标题
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path,f'Scatter_plot_{year}.jpg'), bbox_inches='tight', dpi=300)
        plt.close()  # 关闭图形以释放内存
        '''# # 根据非干旱区域地区当基线值计算SHAP'''
        # # 选择自定义的背景数据
        # background_data = u_df[features]
        # background_data_reduced = shap.kmeans(background_data,k=100)
        # # 初始化 KernelExplainer
        # kernel_explainer = shap.KernelExplainer(rf_model.predict, background_data_reduced)
        # # 计算 SHAP 值
        # shap_values = kernel_explainer.shap_values(X_test)
        # # shap全局特征重要性排序
        # shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
        # # mean_abs_shap_values = np.abs(shap_df).mean()
        # # # 将特征和它们的重要性值放入 DataFrame
        # # feature_importance = pd.DataFrame({
        # #     'Feature': X_test.columns,
        # #     'Mean Absolute SHAP Value': (mean_abs_shap_values)
        # # })
        # # # 按照重要性排序
        # # feature_importance = feature_importance.sort_values(by='Mean Absolute SHAP Value', ascending=False)

        '''# 计算SHAP值（原始方法，全局基线值）'''
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test)
        shap_interaction_values = explainer.shap_interaction_values(X_test)
        # shap全局特征重要性排序
        shap_df = pd.DataFrame(shap_values, columns=X_test.columns)
        # mean_abs_shap_values = np.abs(shap_df).mean()
        # # 将特征和它们的重要性值放入 DataFrame
        # feature_importance = pd.DataFrame({
        #     'Feature': X_test.columns,
        #     'Mean Absolute SHAP Value': (mean_abs_shap_values)
        # })
        # # 按照重要性排序
        # feature_importance = feature_importance.sort_values(by='Mean Absolute SHAP Value', ascending=False)

        # 量化每对特征之间的交互作用
        # interaction_importance = []
        # for i in range(len(X_test.columns)):
        #     for j in range(i + 1, len(X_test.columns)):
        #         interaction_strength = np.abs(shap_interaction_values[:, i, j]).mean()
        #         interaction_importance.append((X_test.columns[i], X_test.columns[j], interaction_strength))
        # # 转换为 DataFrame
        # interaction_importance_df = pd.DataFrame(interaction_importance, columns=['Feature 1', 'Feature 2',
        #                                                                           'Mean Absolute Interaction SHAP Value'])
        #
        # # 按照交互作用强度排序
        # interaction_importance_df = interaction_importance_df.sort_values(by='Mean Absolute Interaction SHAP Value',
        #                                                                   ascending=False)

        # SHAP可视化
        # 保存 SHAP 汇总图为图片
        if if_visualization:
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = 14  # Increase font size
            plt.rcParams['font.weight'] = 'bold'  # Make font bold
            plt.rcParams['text.color'] = 'black'  # Darken font color

            plt.figure()  # 创建一个新的图形对象
            shap.summary_plot(shap_values, X_test, show=False,cmap=plt.get_cmap("RdBu").reversed())  # show=False 防止直接显示图形
            # 保存图像
            plt.savefig(os.path.join(output_path,f'shap_summary_plot_{year}.pdf'), bbox_inches='tight', dpi=300)
            plt.savefig(os.path.join(output_path,f'shap_summary_plot_{year}.jpg'), bbox_inches='tight', dpi=300)
            # plt.savefig('shap_summary_plot.eps', bbox_inches='tight', dpi=300)
            plt.close()  # 关闭图形以释放内存

            # SHAP Dependence Plot for each feature
            for index,feature in enumerate(features_new):
                if index==0:continue
                plt.rcParams['font.family'] = 'Arial'
                plt.figure()
                shap.dependence_plot(features_new[0], shap_values, X_test,interaction_index=features_new[index],show=False,cmap=plt.get_cmap("RdBu").reversed())
                plt.savefig(os.path.join(output_path,f'shap_dependence_plot_{year}_{feature}.pdf'), bbox_inches='tight', dpi=300)
                plt.savefig(os.path.join(output_path,f'shap_dependence_plot_{year}_{feature}.jpg'), bbox_inches='tight', dpi=300)
                # plt.savefig(f'shap_dependence_plot_{feature}.eps', bbox_inches='tight', dpi=300)
                plt.close()

        return mse,r2,np.abs(shap_df),y_pred
    def random_forest_spatial(self,target,features,df,drought_test_number,year,drought_timing,spei_scale,drought_year):
        df_new = df.copy()
        X = df_new[features]
        y = df_new[target]
        df_new[f'spei_drought_lag'] = X[f'spei_{spei_scale}_{drought_timing}_spei_{drought_year}'] * (year - drought_year)
        features_new = features.copy()
        features_new[0] = 'spei_drought_lag'
        X = df_new[features_new]
        X_train, X_test, y_train, y_test = X.head(X.shape[0]-drought_test_number),X.tail(drought_test_number),y.head(y.shape[0]-drought_test_number),y.tail(drought_test_number)
        # 创建和训练随机森林模型
        rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
        rf_model.fit(X_train, y_train)
        # 预测和评估模型
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 计算SHAP值（原始方法，全局基线值）
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_test)
        return shap_values[:,0]
    def random_forest_co_spatial(self, target, features, df_new,drought_test_number):

        df_new = df_new.copy()
        X = df_new[features]
        y = df_new[target]
        X_train, X_test, y_train, y_test = X.head(X.shape[0]-drought_test_number),X.tail(drought_test_number),y.head(y.shape[0]-drought_test_number),y.tail(drought_test_number)
        features_new = features.copy()[0:-1]
        # 创建和训练随机森林模型
        rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
        rf_model.fit(X_train[features_new], y_train)
        # 预测和评估模型
        y_pred = rf_model.predict(X_test[features_new])

        return y_test, y_pred,r2_score(y_test, y_pred)
    def spatial_cross_shap(self,df,countries,target,features,year,drought_timing,spei_scale,drought_year):

        # 空间K交叉获得所有数据点的shap
        spatial_data_all = pd.DataFrame()
        for country in countries:
            drought_train = df[df['country'] != country]
            drought_test = df[df['country'] == country]
            drought_test_number = drought_test.shape[0]
            drought_fusion = pd.concat((drought_train,drought_test))#这个合并是有必要的，因为要是的train和test的两个df按照上下的顺序，这样子才能根据shape的大小获取下方的test
            drought_test_shap = self.random_forest_spatial(target,features,drought_fusion.copy(),drought_test_number,year,drought_timing,spei_scale,drought_year)
            spatial_data = drought_test[['row','col','country']]
            spatial_data[f'legacy_{year}'] = drought_test_shap
            spatial_data_all = pd.concat((spatial_data_all,spatial_data))
        return spatial_data_all

    def rf_modelling(self,data_path,drought_year,legacy_end_year):
        df = pd.read_csv(data_path).dropna()
        # 过滤异常值
        sos_columns = [col for col in df.columns if 'sos' in col]
        eos_columns = [col for col in df.columns if 'eos' in col]
        sos_values = df[sos_columns].values
        eos_values = df[eos_columns].values
        # Calculate invalid SOS/EOS pairs
        invalid_sos_eos_counts = np.sum(sos_values >= eos_values, axis=1)
        # Calculate invalid data points
        combined_values = np.concatenate([sos_values, eos_values], axis=1)
        invalid_data_points_counts = np.sum((combined_values == -32768) | (combined_values == 261), axis=1)
        # Filter the rows
        valid_rows = (invalid_sos_eos_counts == 0) & (invalid_data_points_counts == 0)
        filtered_df = df[valid_rows].copy()
        filtered_df[f'sos_baseline'] = filtered_df[sos_columns].mean(axis=1)
        filtered_df[f'eos_baseline'] = filtered_df[eos_columns].mean(axis=1)
        # 处理一下列名的差别
        for item in [col for col in df.columns if 'summer_half' in col]:
            filtered_df.rename(columns={item: item.replace('summer_half','summerhalf')}, inplace=True)
        for item in [col for col in df.columns if 'winter_half' in col]:
            filtered_df.rename(columns={item: item.replace('winter_half','winterhalf')}, inplace=True)
        # 待分析因素字典
        factors_waiting = {'03':['spring','summer','autumn','annual',],
                           '06':['summerhalf','winterhalf','annual',]}
        # 构建交互值
        legacy_VPD_compund = {}
        for spei_scale in factors_waiting.keys():
            # if spei_scale == '06':continue
            drought_timings = factors_waiting[spei_scale]
            for drought_timing in drought_timings:
                for year in list(range(2000,2024)):
                    legacy_VPD_compund[f'{drought_year}SPEI{spei_scale}_VPD_compund_{drought_timing}_{year}'] = filtered_df[f'spei_{spei_scale}_{drought_timing}_spei_{drought_year}'] * filtered_df[f'VPD_{drought_timing}_{year}']

        filtered_df = pd.concat([filtered_df, pd.DataFrame(legacy_VPD_compund)], axis=1)

        for spei_scale in factors_waiting.keys():
            drought_timings = factors_waiting[spei_scale]
            for drought_timing in drought_timings:
                # 干旱和非干旱区域划分
                drought_year_spei = filtered_df[f'spei_{spei_scale}_{drought_timing}_spei_{drought_year}']
                drought_filtered_df = filtered_df[drought_year_spei<=drought_year_spei[drought_year_spei>drought_year_spei.min()].quantile(0.25)]
                undrought_filtered_df = filtered_df[drought_year_spei > drought_year_spei[drought_year_spei>drought_year_spei.min()].quantile(0.25)]
                # drought_filtered_df, undrought_filtered_df = self.pure_data_df(filtered_df, drought_year,drought_timing, drought_year_spei,spei_scale)
                result_featureimportance_d = pd.DataFrame()
                result_featureimportance_ud = pd.DataFrame()
                countries_d = drought_filtered_df['country'].unique()
                countries_ud = undrought_filtered_df['country'].unique()
                for year in range(drought_year+1, legacy_end_year):
                    target_sos = f'sos_{year}'
                    target_eos = f'eos_{year}'
                    # 干旱指数建模
                    features = [f'spei_{spei_scale}_{drought_timing}_spei_{drought_year}',
                                f'gdd_fixed_sum_{year}',
                                f'VPD_{drought_timing}_{year}',]
                                # f'{drought_year}SPEI{spei_scale}_VPD_compund_{drought_timing}_{year}']
                    # 空间K交叉获得所有数据点的shap
                    spatial_data_all_d = self.spatial_cross_shap(drought_filtered_df,countries_d,target_sos,features,year,drought_timing,spei_scale,drought_year)
                    spatial_data_all_ud = self.spatial_cross_shap(undrought_filtered_df, countries_ud, target_sos, features,
                                                                 year, drought_timing, spei_scale, drought_year)
                    mse_d,r2_d,feature_importance_d,y_pred_d = self.random_forest(target_sos,features,drought_filtered_df.copy(),undrought_filtered_df.copy(),year,drought_timing,spei_scale,drought_year,if_drought = True,if_visualization=True)
                    mse_ud,r2_ud,feature_importance_ud,y_pred_ud = self.random_forest(target_sos, features, undrought_filtered_df.copy(),undrought_filtered_df.copy(),year,drought_timing,spei_scale,drought_year,if_drought = False,if_visualization=True)
                    spatial_data_all_d.to_csv(os.path.join(r'temp\random_forest/',f'drought{drought_year}/spei_{spei_scale}/{drought_timing}/drought_area_legacy{year}.csv'))
                    spatial_data_all_ud.to_csv(os.path.join(r'temp\random_forest/',
                                                           f'drought{drought_year}/spei_{spei_scale}/{drought_timing}/undrought_area_legacy{year}.csv'))
                    result_featureimportance_d = pd.concat([result_featureimportance_d,feature_importance_d],axis = 1)
                    result_featureimportance_ud = pd.concat([result_featureimportance_ud,feature_importance_ud],axis = 1)
                #
                #
                result_featureimportance_d.to_csv(os.path.join(r'temp\random_forest/',f'drought{drought_year}/spei_{spei_scale}/{drought_timing}/drought_area/results_featureimportance_d.csv'))
                result_featureimportance_ud.to_csv(os.path.join(r'temp\random_forest/',
                                                               f'drought{drought_year}/spei_{spei_scale}/{drought_timing}/undrought_area/results_featureimportance_ud.csv'))
    def random_forest_co_Kfold_shap(self,X,y,features_new,baseline_features,weights):
        spatial_data_all = pd.DataFrame()
        shap_data_all = {
            'shap':np.empty((0,len(features_new))),
            'x_val':pd.DataFrame()}     #汇总所有的shap值和对应的X_test
        # 定义 K 折交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        # 创建随机森林模型
        rf_model = RandomForestRegressor(n_estimators=500, n_jobs=10)
        # 使用交叉验证评估 R²
        r2_scores = []
        # 权重
        weights = weights.values.reshape(-1, 1)  # Use .values to avoid issues if 'weights' is a pandas Series
        # Normalize weights to the range [0, 1]
        scaler = MinMaxScaler()
        weights_normalized = scaler.fit_transform(weights).flatten()

        for train_index, val_index in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_index][features_new], X.iloc[val_index][features_new]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
            val_fold_baseline = X.iloc[val_index][baseline_features]
            weights_normalized_k = weights_normalized[train_index]
            # 训练随机森林模型
            rf_model.fit(X_train_fold, y_train_fold,sample_weight=weights_normalized_k)

            # 在验证集上进行预测
            y_pred = rf_model.predict(X_val_fold)

            # 计算验证集的 R² 分数
            r2 = r2_score(y_val_fold, y_pred)
            r2_scores.append(r2)

            # 计算验证集的 SHAP 值
            explainer = shap.TreeExplainer(rf_model)  # 创建SHAP解释器
            shap_values = explainer.shap_values(pd.DataFrame(X_val_fold, columns=features_new))  # 计算验证集的SHAP值
            data_shap = X.iloc[val_index][['row','col','country','year']]
            data_shap[f'legacy_value_spring'] = shap_values[:,0] / (np.sum(np.abs(shap_values),axis=1))
            data_shap[f'legacy_value_summer'] = shap_values[:, 1] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'legacy_value_autumn'] = shap_values[:, 2] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'legacy_value_winterhalf'] = shap_values[:, 3] / (np.sum(np.abs(shap_values), axis=1))
            # 保存每次验证集的 SHAP 值
            spatial_data_all = pd.concat((spatial_data_all,data_shap))
            shap_data_all['shap'] = np.vstack((shap_data_all['shap'],shap_values))
            shap_data_all['x_val'] = pd.concat((shap_data_all['x_val'],X_val_fold))

        print(f'K-fold R_square:{np.mean(r2_scores)}')
        return spatial_data_all,shap_data_all
    def random_forest_co(self, target, features, df_new, drought_year,legacy_end_year):

        # 定义保存图片路径
        output_path = os.path.join(r'temp\random_forest_co/',f'drought{drought_year}_maskweights_withphenologybaseline_onlybroadleaf')
        if not os.path.exists(output_path): os.makedirs(output_path)
        df_new = df_new.copy()
        X = df_new[features]
        y = df_new[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        features_new = features.copy()[0:-9]
        if int(drought_year)<2010:baseline_features = ['sos_baseline_prehalf','eos_baseline_prehalf']
        else:baseline_features = ['sos_baseline_posthalf','eos_baseline_posthalf']
        # 设立权重
        weights = X_train['weights']
        # Reshape weights to a 2D array for MinMaxScaler
        weights = weights.values.reshape(-1, 1)  # Use .values to avoid issues if 'weights' is a pandas Series
        # Normalize weights to the range [0, 1]
        scaler = MinMaxScaler()
        weights_normalized = scaler.fit_transform(weights).flatten()
        # 创建和训练随机森林模型
        rf_model = RandomForestRegressor(n_estimators=500, random_state=42,n_jobs=10)
        rf_model.fit(X_train[features_new], y_train,sample_weight=weights_normalized)
        # 预测和评估模型
        y_pred = rf_model.predict(X_test[features_new])

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # 分位数随机森林回归
        common_params = dict(
            n_estimators=500,  # 使用500颗树
            max_depth=5,
            min_samples_leaf=4,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        qrf = RandomForestQuantileRegressor(**common_params, q=[0.05, 0.5, 0.95])
        qrf.fit(X_train[features_new], y_train)
        # 在测试集上预测分位数
        quantile_predictions = qrf.predict(X_test[features_new])
        # 标准随机森林模型预测均值
        y_pred_mean = rf_model.predict(X_test[features_new])
        # 提取分位数预测值
        y_lower = quantile_predictions[0]  # 5%分位数
        y_med = quantile_predictions[1]  # 50%分位数（中位数）
        y_upper = quantile_predictions[2]  # 95%分位数
        # 打印模型评估指标
        mse = mean_squared_error(y_test, y_pred_mean)
        r2 = r2_score(y_test, y_pred_mean)
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")
        # 绘制分位数预测结果
        fig = plt.figure(figsize=(100, 10))
        # 预测值和真实值
        plt.plot(range(len(y_test)), y_test, 'b.', markersize=10, label='Test observations')
        plt.plot(range(len(y_test)), y_pred_mean, 'r-', label='Predicted mean', color="orange")
        plt.plot(range(len(y_test)), y_med, 'r-', label='Predicted median', color="green")
        plt.plot(range(len(y_test)), y_upper, 'g', label='Predicted 95th percentile')
        plt.plot(range(len(y_test)), y_lower, 'grey', label='Predicted 5th percentile')
        # 绘图细节
        plt.xlabel('Sample Index')
        plt.ylabel('Predicted/True Values')
        plt.legend(loc='upper right')
        plt.title("Quantile Random Forest Prediction")
        plt.savefig('test.jpg')
        # 绘制散点图
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, color='blue', edgecolor='k', s=100)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', lw=2)  # 添加 y=x 的参考线

        # 添加 MSE 和 R2 标签
        plt.text(0.05, 0.90, f'R-square: {r2:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        # 添加轴标签和标题
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted Values')

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path,f'scatter.jpg'))
        plt.close()  # 关闭图形以释放内存
        legacy_values,shap_data_all = self.random_forest_co_Kfold_shap(X, y, features_new,baseline_features,X['weights'])
        shap_df_allyears = pd.DataFrame()
        for year in range(drought_year+1,legacy_end_year):
            legacy_values[legacy_values['year'] == year].to_csv(os.path.join(output_path,f'legacy_{year}.csv'))
            '''# 计算SHAP值（原始方法，全局基线值）'''
            # explainer = shap.TreeExplainer(rf_model)
            # shap_values = explainer.shap_values(X_test[X_test['year']==year][features_new])
            # shap_interaction_values = explainer.shap_interaction_values(X_test[X_test['year']==year][features_new])
            # shap全局特征重要性排序
            shap_values = shap_data_all['shap']
            shap_values_x = shap_data_all['x_val']
            shap_df = pd.DataFrame(shap_values, columns=features_new)
            shap_df['year'] = year
            shap_df_allyears = pd.concat((shap_df_allyears, shap_df))
            # SHAP可视化
            # 保存 SHAP 汇总图为图片
            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = 14  # Increase font size
            plt.rcParams['font.weight'] = 'bold'  # Make font bold
            plt.rcParams['text.color'] = 'black'  # Darken font color

            plt.figure()  # 创建一个新的图形对象
            shap.summary_plot(shap_values, shap_values_x, show=False,
                              cmap=plt.get_cmap("RdBu").reversed())  # show=False 防止直接显示图形
            # 保存图像
            # plt.savefig(os.path.join(output_path, f'shap_summary_plot_{year}.pdf'), bbox_inches='tight', dpi=300)
            plt.savefig(os.path.join(output_path, f'shap_summary_plot_{year}.jpg'), bbox_inches='tight', dpi=300)
            # plt.savefig('shap_summary_plot.eps', bbox_inches='tight', dpi=300)
            plt.close()  # 关闭图形以释放内存

            # SHAP Dependence Plot for each feature
            for i, feature1 in enumerate(features_new):
                for j, feature2 in enumerate(features_new):
                    if i==j : continue
                    plt.rcParams['font.family'] = 'Arial'
                    plt.figure()
                    shap.dependence_plot(feature1, shap_values, shap_values_x, interaction_index=feature2,
                                         show=False, cmap=plt.get_cmap("RdBu").reversed())
                    # plt.savefig(os.path.join(output_path, f'shap_dependence_plot_{year}_{feature}.pdf'),
                    #             bbox_inches='tight', dpi=300)
                    plt.savefig(os.path.join(output_path, f'shap_dependence_plot_{drought_year}_{feature1}_vs_{feature2}.jpg'),
                                bbox_inches='tight', dpi=300)
                    # plt.savefig(f'shap_dependence_plot_{feature}.eps', bbox_inches='tight', dpi=300)
                    plt.close()
        #
        return shap_df_allyears

    def stack_data(self,input_data,drought_year,legacy_end_year):
        new_data = []
        for index, row in input_data.iterrows():
            for year in range(drought_year+1, legacy_end_year):
                new_row = {}
                new_row["gridid"] = f"{row['row']}_{row['col']}_{row['country']}"
                new_row["year"] = year
                new_row['row'] = row['row']
                new_row['col'] = row['col']
                new_row['weights'] = row['weights']
                new_row["country"] = row['country']
                new_row["sos"] = row[f"sos_{year}"]
                new_row["eos"] = row[f"eos_{year}"]
                new_row["spei_spring_lag"] = row[f'spei_03_spring_spei_{year - 1}']
                new_row["spei_summer_lag"] = row[f'spei_03_summer_spei_{year - 1}']
                new_row["spei_autumn_lag"] = row[f'spei_03_autumn_spei_{year - 1}']
                new_row["spei_winterhalf_lag"] = row[f'spei_06_winterhalf_spei_{year - 1}']
                new_row["spei_spring_lag2"] = row[f'spei_03_spring_spei_{year - 2}']
                new_row["spei_summer_lag2"] = row[f'spei_03_summer_spei_{year - 2}']
                new_row["spei_autumn_lag2"] = row[f'spei_03_autumn_spei_{year - 2}']
                new_row["spei_winterhalf_lag2"] = row[f'spei_06_winterhalf_spei_{year - 2}']
                new_row[f"tm_chilling"] = row[f'tm_chilling_{year - 1}']
                new_row[f"chillingdays"] = row[f'chillingdays_{year - 1}']
                new_row[f"tm_summer"] = row[f'tm_summer_{year}']
                new_row[f"tm_summerhalf"] = row[f'tm_summerhalf_{year}']
                new_row["gdd"] = row[f'gdd_fixed_sum_{year}']
                new_row["vpd"] = row[f'VPD_MA_avg_{year}']
                new_row[f'sos_baseline_prehalf'] = row[f'sos_baseline_prehalf']
                new_row[f'eos_baseline_prehalf'] = row[f'eos_baseline_prehalf']
                new_row[f'sos_baseline_posthalf'] = row[f'sos_baseline_posthalf']
                new_row[f'eos_baseline_posthalf'] = row[f'eos_baseline_posthalf']
                new_row["sos_drought_year"] = row[f"sos_{drought_year}"]
                new_row["eos_drought_year"] = row[f"eos_{drought_year}"]
                new_row["gdd_drought_year"] = row[f'gdd_fixed_sum_{drought_year}']
                new_row["vpd_drought_year"] = row[f'VPD_MA_avg_{drought_year}']
                new_row["spei_spring_drought_year"] = row[f'spei_03_spring_spei_{drought_year}']
                new_row["spei_summer_drought_year"] = row[f'spei_03_summer_spei_{drought_year}']
                new_row["spei_autumn_drought_year"] = row[f'spei_03_autumn_spei_{drought_year}']
                new_row[f"tm_summer_drought_year"] = row[f'tm_summer_{drought_year}']
                new_row[f"tm_summerhalf_drought_year"] = row[f'tm_summerhalf_{drought_year}']

                new_row["spei_spring"] = row[f'spei_03_spring_spei_{year}']
                new_row["spei_summer"] = row[f'spei_03_summer_spei_{year}']
                new_row["spei_autumn"] = row[f'spei_03_autumn_spei_{year}']
                new_data.append(new_row)
        new_df = pd.DataFrame(new_data)
        return new_df


    def dtrend_zscore(self,df,cols):
        for col in cols:
            df[col + '_detrended'] = signal.detrend(df[col])

        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[[col + '_detrended' for col in cols]])
        return df
    def rf_modelling_co(self,data_path,drought_year,legacy_end_year):
        '''
        随机森林建模，把所有年份汇聚到一块建模
        :param data_path:
        :param drought_year:
        :param legacy_end_year:
        :return:
        '''
        legacy_end_year_max = legacy_end_year
        df = pd.read_csv(data_path).dropna()
        # 过滤异常值
        sos_columns = [col for col in df.columns if 'sos' in col]
        eos_columns = [col for col in df.columns if 'eos' in col]
        sos_values = df[sos_columns].values
        eos_values = df[eos_columns].values
        # Calculate invalid SOS/EOS pairs
        invalid_sos_eos_counts = np.sum(sos_values >= eos_values, axis=1)
        # Calculate invalid data points
        combined_values = np.concatenate([sos_values, eos_values], axis=1)
        invalid_data_points_counts = np.sum((combined_values == -32768) | (combined_values == 261), axis=1)
        # Filter the rows
        valid_rows = (invalid_sos_eos_counts == 0) & (invalid_data_points_counts == 0)
        filtered_df = df[valid_rows].copy()
        filtered_df[f'sos_baseline_prehalf'] = filtered_df[sos_columns[:len(sos_columns)//2]].mean(axis=1)
        filtered_df[f'eos_baseline_prehalf'] = filtered_df[eos_columns[:len(eos_columns)//2]].mean(axis=1)
        filtered_df[f'sos_baseline_posthalf'] = filtered_df[sos_columns[len(sos_columns)//2:]].mean(axis=1)
        filtered_df[f'eos_baseline_posthalf'] = filtered_df[eos_columns[len(eos_columns)//2:]].mean(axis=1)
        # 处理一下列名的差别
        for item in [col for col in df.columns if 'summer_half' in col]:
            filtered_df.rename(columns={item: item.replace('summer_half','summerhalf')}, inplace=True)
        for item in [col for col in df.columns if 'winter_half' in col]:
            filtered_df.rename(columns={item: item.replace('winter_half','winterhalf')}, inplace=True)

        # 构建复合干旱指标
        filtered_df['compound_spei_1819'] = 0
        filtered_df['compound_spei_181922'] = 0
        for item in ['spring','summer', 'autumn', 'winterhalf']:
            speiscale = '03'
            if item == 'winterhalf': speiscale = '06'
            filtered_df[f'compound_spei_1819'] = filtered_df[f'compound_spei_1819'] + filtered_df[f'spei_{speiscale}_{item}_spei_2018'] + filtered_df[f'spei_{speiscale}_{item}_spei_2019']
        for item in ['spring','summer', 'autumn', 'winterhalf']:
            speiscale = '03'
            if item == 'winterhalf': speiscale = '06'
            filtered_df[f'compound_spei_181922'] = filtered_df[f'compound_spei_181922'] + filtered_df[f'spei_{speiscale}_{item}_spei_2018'] + filtered_df[f'spei_{speiscale}_{item}_spei_2019'] + filtered_df[f'spei_{speiscale}_{item}_spei_2022']

        # 标准化
        exclude_columns = ['Unnamed: 0','row', 'col','country','weights', 'sos_baseline_prehalf','eos_baseline_prehalf','sos_baseline_posthalf','eos_baseline_posthalf',]
        exclude_columns += sos_columns
        exclude_columns += eos_columns
        columns_to_standardize = [col for col in filtered_df.columns if col not in exclude_columns]
        filtered_df_normalization = filtered_df.copy()
        for col in columns_to_standardize:
            filtered_df_normalization[col] = (filtered_df_normalization[col] - filtered_df_normalization[col].mean()) / \
                                             filtered_df_normalization[col].std()

        filtered_df_stack = self.stack_data(filtered_df_normalization,drought_year,legacy_end_year)
        target_sos = 'sos'
        target_eos = 'eos'
        features = [f'spei_spring_lag',
                    f'spei_summer_lag',
                    f'spei_autumn_lag',
                    f'spei_winterhalf_lag',
                    f'gdd',
                    f'vpd',
                    'chillingdays',
                    'year',
                    'country',
                    'row',
                    'col',
                    'weights',
                    'sos_baseline_prehalf',
                    'eos_baseline_prehalf',
                    'sos_baseline_posthalf',
                    'eos_baseline_posthalf']
        feature_importance_d = self.random_forest_co(target_sos,features,filtered_df_stack,drought_year,legacy_end_year)
        feature_importance_d.to_csv(os.path.join(r'temp\random_forest_co/',f'drought{drought_year}/results_featureimportance_d.csv'))

    def rf_modelling_co_for_compounddrought(self,data_path):

        df = pd.read_csv(data_path).dropna()
        # 过滤异常值
        sos_columns = [col for col in df.columns if 'sos' in col]
        eos_columns = [col for col in df.columns if 'eos' in col]
        sos_values = df[sos_columns].values
        eos_values = df[eos_columns].values
        # Calculate invalid SOS/EOS pairs
        invalid_sos_eos_counts = np.sum(sos_values >= eos_values, axis=1)
        # Calculate invalid data points
        combined_values = np.concatenate([sos_values, eos_values], axis=1)
        invalid_data_points_counts = np.sum((combined_values == -32768) | (combined_values == 261), axis=1)
        # Filter the rows
        valid_rows = (invalid_sos_eos_counts == 0) & (invalid_data_points_counts == 0)
        filtered_df = df[valid_rows].copy()
        filtered_df[f'sos_baseline_prehalf'] = filtered_df[sos_columns[:len(sos_columns)//2]].mean(axis=1)
        filtered_df[f'eos_baseline_prehalf'] = filtered_df[eos_columns[:len(eos_columns)//2]].mean(axis=1)
        filtered_df[f'sos_baseline_posthalf'] = filtered_df[sos_columns[len(sos_columns)//2:]].mean(axis=1)
        filtered_df[f'eos_baseline_posthalf'] = filtered_df[eos_columns[len(eos_columns)//2:]].mean(axis=1)
        # 处理一下列名的差别
        for item in [col for col in df.columns if 'summer_half' in col]:
            filtered_df.rename(columns={item: item.replace('summer_half','summerhalf')}, inplace=True)
        for item in [col for col in df.columns if 'winter_half' in col]:
            filtered_df.rename(columns={item: item.replace('winter_half','winterhalf')}, inplace=True)
        # 待分析因素字典
        factors_waiting = {'03':['spring','summer','autumn','annual',],
                           '06':['summerhalf','winterhalf','annual',]}
        # 构建复合干旱指标
        filtered_df['compound_spei_1819'] = 0
        filtered_df['compound_spei_181922'] = 0
        for item in ['spring','summer', 'autumn', 'winterhalf']:
            speiscale = '03'
            if item == 'winterhalf': speiscale = '06'
            filtered_df[f'compound_spei_1819'] = filtered_df[f'compound_spei_1819'] + filtered_df[f'spei_{speiscale}_{item}_spei_2018'] + filtered_df[f'spei_{speiscale}_{item}_spei_2019']
        for item in ['spring','summer', 'autumn', 'winterhalf']:
            speiscale = '03'
            if item == 'winterhalf': speiscale = '06'
            filtered_df[f'compound_spei_181922'] = filtered_df[f'compound_spei_181922'] + filtered_df[f'spei_{speiscale}_{item}_spei_2018'] + filtered_df[f'spei_{speiscale}_{item}_spei_2019'] + filtered_df[f'spei_{speiscale}_{item}_spei_2022']

        # 标准化
        exclude_columns = ['Unnamed: 0','row', 'col','country', 'sos_baseline_prehalf','eos_baseline_prehalf','sos_baseline_posthalf','eos_baseline_posthalf',]
        exclude_columns += sos_columns
        exclude_columns += eos_columns
        columns_to_standardize = [col for col in filtered_df.columns if col not in exclude_columns]
        filtered_df_normalization = filtered_df.copy()
        for col in columns_to_standardize:
            filtered_df_normalization[col] = (filtered_df_normalization[col] - filtered_df_normalization[col].mean()) / \
                                             filtered_df_normalization[col].std()


        target_sos_2020 = 'sos_2020'
        target_sos_2023 = 'sos_2023'
        features_2020 = ['compound_spei_1819',
                         'gdd_fixed_sum_2020',
                         'VPD_MA_avg_2020',
                         'tm_chilling_2019',
                         'spei_03_spring_spei_2018',
                         'spei_03_spring_spei_2019',
                         'spei_03_summer_spei_2018',
                         'spei_03_summer_spei_2019',
                         'spei_03_autumn_spei_2018',
                         'spei_03_autumn_spei_2019',
                         'spei_06_winterhalf_spei_2018',
                         'spei_06_winterhalf_spei_2019',
                        'year',
                        'country',
                        'row',
                        'col',
                        'sos_baseline_prehalf',
                        'eos_baseline_prehalf',
                        'sos_baseline_posthalf',
                        'eos_baseline_posthalf']
        features_2023 = ['compound_spei_181922',
                         'gdd_fixed_sum_2023',
                         'VPD_MA_avg_2023',
                         'tm_chilling_2022',
                         'spei_03_spring_spei_2018',
                         'spei_03_spring_spei_2019',
                         'spei_03_summer_spei_2018',
                         'spei_03_summer_spei_2019',
                         'spei_03_autumn_spei_2018',
                         'spei_03_autumn_spei_2019',
                         'spei_06_winterhalf_spei_2018',
                         'spei_06_winterhalf_spei_2019',
                        'year',
                        'country',
                        'row',
                        'col',
                        'sos_baseline_prehalf',
                        'eos_baseline_prehalf',
                        'sos_baseline_posthalf',
                        'eos_baseline_posthalf']
        feature_importance_d = self.random_forest_co(target_sos,features,filtered_df_stack,drought_year,legacy_end_year)
        feature_importance_d.to_csv(os.path.join(r'temp\random_forest_co/',f'drought{drought_year}/results_featureimportance_d.csv'))
    def visualization_tools1(self,df,years,spei_columns,drought_timing):
        df = df.abs()
        for index, year in enumerate(years):
            df_year = df[df['year']==year].copy()
            spei_year = df_year[spei_columns[index]]
            features_year = df_year[[spei_columns[index], f'vpd', f'gdd','spei_current']]
            df[f'drought_lag{year}'] = (spei_year / features_year.sum(axis=1))*100
            df[f'GDD_{year}'] = (df_year[f'gdd']/features_year.sum(axis=1))*100
            df[f'VPD_{year}'] = (df_year[f'vpd'] / features_year.sum(axis=1)) * 100
            df[f'SPEI_current_{year}'] = (df_year[f'spei_current'] / features_year.sum(axis=1)) * 100
        return df
    def rf_results_visualization(self,drought_data_path,undrought_data_path,drought_year,legacy_end_year,drought_timing):
        output_path = os.path.split(os.path.split(drought_data_path)[0])[0]
        drought_df = pd.read_csv(drought_data_path)
        undrought_df = pd.read_csv(undrought_data_path)
        years = np.arange(drought_year+1, drought_year+9)
        if drought_year == 2003:
            spei_columns = ['spei_drought_lag'] + [f'spei_drought_lag.{i}' for i in range(1, 11)]
            drought_df_process = self.visualization_tools1(drought_df.copy(),years,spei_columns,drought_timing)
            undrought_df_process = self.visualization_tools1(undrought_df.copy(),years,spei_columns,drought_timing)
            spei_lag_columns = [f'drought_lag{i}' for i in range(2004, 2012)]
            spei_data_d = drought_df_process[spei_lag_columns]
            means_d = spei_data_d.mean()
            sem_d = spei_data_d.sem()
            spei_data_ud = undrought_df_process[spei_lag_columns]
            means_ud = spei_data_ud.mean()
            sem_ud = spei_data_ud.sem()
        if drought_year == 2015:
            spei_columns = ['spei_drought_lag'] + [f'spei_drought_lag.{i}' for i in range(1, 2)]
            drought_df_process = self.visualization_tools1(drought_df.copy(),years,spei_columns,drought_timing)
            undrought_df_process = self.visualization_tools1(undrought_df.copy(),years,spei_columns,drought_timing)
            spei_lag_columns = [f'drought_lag{i}' for i in range(2016, 2018)]
            spei_data_d = drought_df_process[spei_lag_columns]
            means_d = spei_data_d.mean()
            sem_d = spei_data_d.sem()
            spei_data_ud = undrought_df_process[spei_lag_columns]
            means_ud = spei_data_ud.mean()
            sem_ud = spei_data_ud.sem()

        # Plotting
        plt.rcParams['font.family'] = 'Arial'
        plt.figure(figsize=(6, 3))
        plt.plot(years-drought_year, means_d, marker='o', linestyle='-', color='#6A3D00', label='Mean Legacy in Drought Area', markersize=4, linewidth=1, alpha=0.8)
        plt.fill_between(years-drought_year, means_d - sem_d, means_d + sem_d, color='#858786', alpha=0.2, edgecolor='none', label='Uncertainty Range')
        plt.plot(years-drought_year, means_ud, marker='o', linestyle='-', color='#004C42', label='Mean SPEI Drought Lag in Undrought Area', markersize=4, linewidth=1, alpha=0.8)
        plt.fill_between(years-drought_year, means_ud - sem_ud, means_ud + sem_ud, color='#858786', alpha=0.2, edgecolor='none', label='Uncertainty Range')

        # Labels and title
        plt.xlabel('Post Drought Year', fontsize=10)
        ylabel_text = textwrap.fill('Relative Contribution of Drought Legacy (%)', width=20)
        plt.ylabel(ylabel_text, fontsize=10, labelpad=10)
        # plt.savefig(os.path.join(output_path,'legacy_contribution_trend.jpg'), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(output_path,'legacy_contribution_trend2.jpg'), bbox_inches='tight', dpi=1200)
        # plt.savefig(os.path.join(output_path,'legacy_contribution_trend.jpg'), dpi=300)
    def rf_results_visualization_box(self,drought_data_path,undrought_data_path,drought_year,legacy_end_year,drought_timing):

        # ''' 先画了gfoe部分绘图的代码，需要修改'''
        output_path = os.path.split(os.path.split(drought_data_path)[0])[0]
        drought_df = pd.read_csv(drought_data_path)
        undrought_df = pd.read_csv(undrought_data_path)
        years = np.arange(drought_year+1, legacy_end_year)
        if drought_year == 2003:
            spei_columns = ['spei_drought_lag'] + [f'spei_drought_lag.{i}' for i in range(1, 11)]
            drought_df_process = self.visualization_tools1(drought_df.copy(),years,spei_columns,drought_timing)
            undrought_df_process = self.visualization_tools1(undrought_df.copy(),years,spei_columns,drought_timing)
            drought_df_process['Region'] = 'Drought'
            undrought_df_process['Region'] = 'Nondrought'
            palette = {'Drought': '#D9534F', 'Nondrought': '#4C72B0'}
            # 合并数据框
            combined_df = pd.concat([drought_df_process, undrought_df_process])

            # 确保 Region 列是分类数据
            combined_df['Region'] = combined_df['Region'].astype('category')
            plt.rcParams['font.family'] = 'Arial'
            # plt.rcParams['font.weight'] = 'bold'  # 字体加黑
            # 设置图形的尺寸
            plt.figure(figsize=(10.5,3), dpi=1200)
            # 设置字体
            sns.set_context("notebook",
                            rc={"font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18, "xtick.labelsize": 16,
                                "ytick.labelsize": 16,})

            # 绘制 GDD 的箱型图，去掉异常点
            plt.subplot(1, 3, 1)
            sns.boxplot(x='Region', y='GDD_2004', data=combined_df, showfliers=False,palette=palette,width=0.6)
            plt.ylabel('Contribution of GDD', )
            plt.xticks(ticks=[0, 1], labels=['Drought', 'Non-drought'], fontsize=14,)
            plt.yticks(fontsize=14,)
            # 绘制 SPEI 的箱型图，去掉异常点
            plt.subplot(1, 3, 2)
            sns.boxplot(x='Region', y='drought_lag2004', data=combined_df, showfliers=False,palette=palette,width=0.6)
            plt.ylabel('Contribution of\n drought legacy',)
            plt.xticks(ticks=[0, 1], labels=['Drought', 'Non-drought'], fontsize=14, )
            plt.yticks(fontsize=14, )
            # 绘制 VPD 的箱型图，去掉异常点
            plt.subplot(1, 3, 3)
            sns.boxplot(x='Region', y='VPD_2004', data=combined_df, showfliers=False,palette=palette,width=0.6)
            plt.ylabel('Contribution of VPD',)
            plt.xticks(ticks=[0, 1], labels=['Drought', 'Non-drought'], fontsize=14, )
            plt.yticks(fontsize=14, )
            # 显示图形
            plt.tight_layout(pad=1.5)
            plt.savefig('test.jpg',dpi=1200)
        if drought_year == 2015:
            spei_columns = ['spei_drought_lag'] + [f'spei_drought_lag.{i}' for i in range(1, 2)]
            drought_df_process = self.visualization_tools1(drought_df.copy(),years,spei_columns,drought_timing)
            undrought_df_process = self.visualization_tools1(undrought_df.copy(),years,spei_columns,drought_timing)
            spei_lag_columns = [f'drought_lag{i}' for i in range(2016, 2018)]
            spei_data_d = drought_df_process[spei_lag_columns]
            means_d = spei_data_d.mean()
            sem_d = spei_data_d.sem()
            spei_data_ud = undrought_df_process[spei_lag_columns]
            means_ud = spei_data_ud.mean()
            sem_ud = spei_data_ud.sem()

        # Plotting
        plt.rcParams['font.family'] = 'Arial'
        plt.figure(figsize=(6, 3))
        plt.plot(years-drought_year, means_d, marker='o', linestyle='-', color='#B7222F', label='Mean Legacy in Drought Area', markersize=4, linewidth=1, alpha=0.8)
        plt.fill_between(years-drought_year, means_d - sem_d, means_d + sem_d, color='#858786', alpha=0.2, edgecolor='none', label='Uncertainty Range')
        plt.plot(years-drought_year, means_ud, marker='o', linestyle='-', color='#114680', label='Mean SPEI Drought Lag in Undrought Area', markersize=4, linewidth=1, alpha=0.8)
        plt.fill_between(years-drought_year, means_ud - sem_ud, means_ud + sem_ud, color='#858786', alpha=0.2, edgecolor='none', label='Uncertainty Range')

        # Labels and title
        plt.xlabel('Post Drought Year', fontsize=10)
        ylabel_text = textwrap.fill('Relative Contribution of Drought Legacy (%)', width=20)
        plt.ylabel(ylabel_text, fontsize=10, labelpad=10)
        # plt.savefig(os.path.join(output_path,'legacy_contribution_trend.jpg'), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(output_path,'legacy_contribution_trend2.jpg'), bbox_inches='tight', dpi=300)
        # plt.savefig(os.path.join(output_path,'legacy_contribution_trend.jpg'), dpi=300)
    def rf_results_visualization_box_co(self, drought_data_path, undrought_data_path, drought_year, legacy_end_year,
                                     drought_timing):

        output_path = os.path.split(os.path.split(drought_data_path)[0])[0]
        drought_df = pd.read_csv(drought_data_path)
        undrought_df = pd.read_csv(undrought_data_path)
        years = np.arange(drought_year + 1, drought_df['year'].unique().max()+1)
        for year in years:
            spei_columns = ['spei_drought_lag']
            drought_df_process = self.visualization_tools1(drought_df.copy(), years, spei_columns, drought_timing)
            undrought_df_process = self.visualization_tools1(undrought_df.copy(), years, spei_columns,
                                                             drought_timing)
            drought_df_process['Region'] = 'Drought'
            undrought_df_process['Region'] = 'Nondrought'
            palette = {'Drought': '#D9534F', 'Nondrought': '#4C72B0'}
            # 合并数据框
            combined_df = pd.concat([drought_df_process, undrought_df_process])

            # 确保 Region 列是分类数据
            combined_df['Region'] = combined_df['Region'].astype('category')
            plt.rcParams['font.family'] = 'Arial'
            # plt.rcParams['font.weight'] = 'bold'  # 字体加黑
            # 设置图形的尺寸
            plt.figure(figsize=(18, 3), dpi=30)
            # 设置字体
            sns.set_context("notebook",
                            rc={"font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18, "xtick.labelsize": 16,
                                "ytick.labelsize": 16, })

            # 绘制 GDD 的箱型图，去掉异常点
            plt.subplot(1, 4, 1)
            sns.boxplot(x='Region', y=f'GDD_{year}', data=combined_df, showfliers=False, palette=palette, width=0.6)
            plt.ylabel('Contribution of GDD', )
            plt.xticks(ticks=[0, 1], labels=['Drought', 'Non-drought'], fontsize=14, )
            plt.yticks(fontsize=14, )
            # 绘制 SPEI 的箱型图，去掉异常点
            plt.subplot(1, 4, 2)
            sns.boxplot(x='Region', y=f'drought_lag{year}', data=combined_df, showfliers=False, palette=palette,
                        width=0.6)
            plt.ylabel('Contribution of\n drought legacy', )
            plt.xticks(ticks=[0, 1], labels=['Drought', 'Non-drought'], fontsize=14, )
            plt.yticks(fontsize=14, )
            # 绘制 VPD 的箱型图，去掉异常点
            plt.subplot(1, 4, 3)
            sns.boxplot(x='Region', y=f'VPD_{year}', data=combined_df, showfliers=False, palette=palette, width=0.6)
            plt.ylabel('Contribution of VPD', )
            plt.xticks(ticks=[0, 1], labels=['Drought', 'Non-drought'], fontsize=14, )
            plt.yticks(fontsize=14, )
            # 绘制 SPEI_current 的箱型图，去掉异常点
            plt.subplot(1, 4, 4)
            sns.boxplot(x='Region', y=f'SPEI_current_{year}', data=combined_df, showfliers=False, palette=palette, width=0.6)
            plt.ylabel(f'Contribution of SPEI_current_{year}', )
            plt.xticks(ticks=[0, 1], labels=['Drought', 'Non-drought'], fontsize=14, )
            plt.yticks(fontsize=14, )
            # 显示图形
            plt.tight_layout(pad=1.5)
            plt.savefig(os.path.join(os.path.split(drought_data_path)[0],f'{year}.jpg'), dpi=300)

    def generate_spatial_basedata(self,phenology_path,SPEI_path):
        phenology_nc = xr.open_dataset(phenology_path)
        base_data = self.spatial_aggregation(phenology_nc, xr.open_dataset(SPEI_path), 'mean')
        base_data['Legacy'] = (('time', 'lat', 'lon'), np.full_like(base_data['SOS'].values, -9999))
        legacy_data = base_data['Legacy'].sel(time='2001-01-01')
        legacy_data.attrs = base_data.attrs
        legacy_data = legacy_data.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
        legacy_data.rio.to_raster(os.path.join(os.path.split(phenology_path)[0],'legacy.tif'))
    def rf_results_legacy_spatial_visualization(self,data_path,drought_year,legacy_end_year,drought_timing):
        years = np.arange(drought_year+1, legacy_end_year)
        for year in years:
            legacy_data = pd.read_csv(os.path.join(data_path,f'legacy_{year}.csv'))
            countries = legacy_data['country'].unique().tolist()
            for country_name in countries:
                country_legacy_data = legacy_data[legacy_data['country'] == country_name]
                phenology_path = os.path.join(f'D:\Data Collection\RS\Disturbance/7080016/{country_name}','phenology.nc')
                self.generate_spatial_basedata(phenology_path,r'data collection\SPEI/spei03.nc')
                country_basetif = gdal.Open(os.path.join(os.path.split(phenology_path)[0],'legacy.tif'))
                country_basetif_data = country_basetif.GetRasterBand(1).ReadAsArray()
                for index, row in country_legacy_data.iterrows():
                    country_basetif_data[row['row'],row['col']] = row[f'legacy_value_{drought_timing}']
                driver = gdal.GetDriverByName('GTiff')

                # Create a new GeoTIFF file with the modified data
                output_path = os.path.join(r'temp\random_forest_co\spatial_visualization', f'{country_name}_{year}_{drought_timing}legacy.tif')
                out_tif = driver.Create(output_path, country_basetif.RasterXSize, country_basetif.RasterYSize, 1,
                                        gdal.GDT_Float32)
                # Set the GeoTIFF's spatial reference system
                out_tif.SetGeoTransform(country_basetif.GetGeoTransform())
                out_tif.SetProjection(country_basetif.GetProjection())
                # Write the modified data to the new file
                out_band = out_tif.GetRasterBand(1)
                out_band.WriteArray(country_basetif_data)
                out_band.SetNoDataValue(-9999)
                # Flush data to disk and close the file
                out_band.FlushCache()
                out_tif = None
                print(f'Modified GeoTIFF saved as {output_path}')

    def rf_results_spatial_visualization_merge(self,tif_paths,output_path):
        # 用于合并tif文件的命令
        vrt_options = gdal.BuildVRTOptions(addAlpha=True)  # 去掉resampleAlg参数
        vrt = gdal.BuildVRT('/vsimem/merged.vrt', tif_paths, options=vrt_options)

        # 将虚拟VRT文件转换为实际的tif文件
        gdal.Translate(output_path, vrt)
    def merge_tifs(self,tif_paths,output_path):
        # 读取所有 TIFF 文件
        src_files_to_mosaic = []
        for fp in tif_paths:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)

        # 使用 rasterio 的 merge 函数合并文件
        mosaic, out_trans = merge(src_files_to_mosaic, nodata=None)

        # 获取第一个栅格文件的元数据作为基础
        out_meta = src_files_to_mosaic[0].meta.copy()

        # 更新输出元数据，以匹配合并后的数据集
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans
        })

        # 输出合并后的文件
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        # 关闭所有输入文件
        for src in src_files_to_mosaic:
            src.close()

        print(f"合并后的文件已保存到: {output_path}")
    def paper_drawing_study_area(self,mask_tifs,SPEI_paths,phenology_tifs,country_name):
        spei_03 = self.get_SPEI_current_region(SPEI_paths[0], phenology_tifs[0], '03')
        spei_06 = self.get_SPEI_current_region(SPEI_paths[1], phenology_tifs[0], '06')
        for mask_tif in mask_tifs:
            # 看聚合后的栅格中，每个栅格有多少原来的点（经过掩膜后过滤的点）参与了计算,同时绘制直方图
            mask_nc = self.tif_tonc(mask_tif)
            mask_num_nc = self.spatial_aggregation(mask_nc, xr.open_dataset(SPEI_paths[0]), 'sum')
            data_mask_num = mask_num_nc['data'].values.flatten()
            # 根据四分位数确定mask数量的阈值
            try:
                mask_num_threshold = np.percentile(data_mask_num[data_mask_num > 0], 25)
            except:
                mask_num_threshold = 30.0
            mask_array = mask_nc['data'].data
            mask_array_others = np.repeat(np.expand_dims(mask_array.data, 0), 22, axis=0)     #这里面除了物候数据，其他数据年数都是2000年到2023年，因此统一用一个expandmask的形状就好了
            spei_03_sa = self.spatial_aggregation(spei_03.where(mask_array_others == 1),xr.open_dataset(SPEI_paths[0]), 'mean')
            # spei_06_sa = self.spatial_aggregation(spei_06.where(mask_array_others == 1),xr.open_dataset(SPEI_paths[0]), 'mean')

            data_drawing =spei_03_sa['annual_spei'].sel(year=2003)
            drawing_mask = mask_num_nc['data'] > mask_num_threshold
            new_data = np.where(drawing_mask,data_drawing,-9999)
            new_data_da = xr.DataArray(new_data, dims=data_drawing.dims, coords=data_drawing.coords)
            data_drawing['spei_mask'] = new_data_da
            new_data_array = xr.DataArray(
                new_data,
                coords=data_drawing.coords,
                dims=data_drawing.dims,
                attrs=data_drawing.attrs
            )
            transform = from_origin(
                west=new_data_array.lon.min().item() - 0.25,  # Upper-left longitude
                north=new_data_array.lat.max().item() + 0.25,  # Upper-left latitude
                xsize=(new_data_array.lon[1] - new_data_array.lon[0]).item(),  # Longitude resolution
                ysize=(new_data_array.lat[0] - new_data_array.lat[1]).item()   # Latitude resolution
            )

            # Set the CRS for WGS84
            crs = 'EPSG:4326'
            # Define the output file name
            output_tif = os.path.join(r'C:\CMZ\PhD\2024\conference\Gfoe\study_area_tifs/{}.tif'.format(country_name))

            # Open a new raster file and write data
            with rasterio.open(
                output_tif,
                'w',
                driver='GTiff',
                height=new_data_array.shape[0],
                width=new_data_array.shape[1],
                count=1,  # Number of bands
                dtype=new_data_array.dtype,
                crs=crs,
                transform=transform,
                nodata=-9999  # Specify the nodata value
            ) as dst:
                dst.write(new_data_array.values, 1)  # Write the data to the first band

class HANTS:

    def __init__(self):
        '''
        HANTS algorithm for time series smoothing
        '''
        pass

    def makediag3d(self,M):
        b = np.zeros((M.shape[0], M.shape[1] * M.shape[1]))
        b[:, ::M.shape[1] + 1] = M
        return b.reshape((M.shape[0], M.shape[1], M.shape[1]))

    def get_starter_matrix(self,base_period_len, sample_count, frequencies_considered_count):
        nr = min(2 * frequencies_considered_count + 1,
                 sample_count)  # number of 2*+1 frequencies, or number of input images
        mat = np.zeros(shape=(nr, sample_count))
        mat[0, :] = 1
        ang = 2 * np.pi * np.arange(base_period_len) / base_period_len
        cs = np.cos(ang)
        sn = np.sin(ang)
        # create some standard sinus and cosinus functions and put in matrix
        i = np.arange(1, frequencies_considered_count + 1)
        ts = np.arange(sample_count)
        for column in range(sample_count):
            index = np.mod(i * ts[column], base_period_len)
            # index looks like 000, 123, 246, etc, until it wraps around (for len(i)==3)
            mat[2 * i - 1, column] = cs.take(index)
            mat[2 * i, column] = sn.take(index)
        return mat

    def hants(self,sample_count, inputs,
              frequencies_considered_count=3,
              outliers_to_reject='Hi',
              low=0., high=255,
              fit_error_tolerance=5.,
              delta=0.1):
        """
        Function to apply the Harmonic analysis of time series applied to arrays
        sample_count    = nr. of images (total number of actual samples of the time series)
        base_period_len    = length of the base period, measured in virtual samples
                (days, dekads, months, etc.)
        frequencies_considered_count    = number of frequencies to be considered above the zero frequency
        inputs     = array of input sample values (e.g. NDVI values)
        ts    = array of size sample_count of time sample indicators
                (indicates virtual sample number relative to the base period);
                numbers in array ts maybe greater than base_period_len
                If no aux file is used (no time samples), we assume ts(i)= i,
                where i=1, ..., sample_count
        outliers_to_reject  = 2-character string indicating rejection of high or low outliers
                select from 'Hi', 'Lo' or 'None'
        low   = valid range minimum
        high  = valid range maximum (values outside the valid range are rejeced
                right away)
        fit_error_tolerance   = fit error tolerance (points deviating more than fit_error_tolerance from curve
                fit are rejected)
        dod   = degree of overdeterminedness (iteration stops if number of
                points reaches the minimum required for curve fitting, plus
                dod). This is a safety measure
        delta = small positive number (e.g. 0.1) to suppress high amplitudes
        """
        # define some parameters
        base_period_len = sample_count  #

        # check which setting to set for outlier filtering
        if outliers_to_reject == 'Hi':
            sHiLo = -1
        elif outliers_to_reject == 'Lo':
            sHiLo = 1
        else:
            sHiLo = 0

        nr = min(2 * frequencies_considered_count + 1,
                 sample_count)  # number of 2*+1 frequencies, or number of input images

        # create empty arrays to fill
        outputs = np.zeros(shape=(inputs.shape[0], sample_count))

        mat = self.get_starter_matrix(base_period_len, sample_count, frequencies_considered_count)

        # repeat the mat array over the number of arrays in inputs
        # and create arrays with ones with shape inputs where high and low values are set to 0
        mat = np.tile(mat[None].T, (1, inputs.shape[0])).T
        p = np.ones_like(inputs)
        p[(low >= inputs) | (inputs > high)] = 0
        nout = np.sum(p == 0, axis=-1)  # count the outliers for each timeseries

        # prepare for while loop
        ready = np.zeros((inputs.shape[0]), dtype=bool)  # all timeseries set to false

        dod = 1  # (2*frequencies_considered_count-1)  # Um, no it isn't :/
        noutmax = sample_count - nr - dod
        for _ in range(sample_count):
            if ready.all():
                break
            # print '--------*-*-*-*',it.value, '*-*-*-*--------'
            # multiply outliers with timeseries
            za = np.einsum('ijk,ik->ij', mat, p * inputs)

            # multiply mat with the multiplication of multiply diagonal of p with transpose of mat
            diag = self.makediag3d(p)
            A = np.einsum('ajk,aki->aji', mat, np.einsum('aij,jka->ajk', diag, mat.T))
            # add delta to suppress high amplitudes but not for [0,0]
            A = A + np.tile(np.diag(np.ones(nr))[None].T, (1, inputs.shape[0])).T * delta
            A[:, 0, 0] = A[:, 0, 0] - delta

            # solve linear matrix equation and define reconstructed timeseries
            zr = np.linalg.solve(A, za)
            outputs = np.einsum('ijk,kj->ki', mat.T, zr)

            # calculate error and sort err by index
            err = p * (sHiLo * (outputs - inputs))
            rankVec = np.argsort(err, axis=1, )

            # select maximum error and compute new ready status
            maxerr = np.diag(err.take(rankVec[:, sample_count - 1], axis=-1))
            ready = (maxerr <= fit_error_tolerance) | (nout == noutmax)

            # if ready is still false
            if not ready.all():
                j = rankVec.take(sample_count - 1, axis=-1)

                p.T[j.T, np.indices(j.shape)] = p.T[j.T, np.indices(j.shape)] * ready.astype(
                    int)  # *check
                nout += 1

        return outputs


class Phenology:

    def __init__(self):
        '''
        # step1: get the 365-day NDVI time series using the function self.hants_smooth
        # step2: get phenology information using the function self.pick_early_peak_late_dormant_period
        '''
        pass

    def hants_smooth(self,VI_bi_week):
        '''
        :param VI_bi_week: bi-weekly VI values
        :return: 365-day NDVI time series
        '''
        VI_bi_week = np.array(VI_bi_week)
        std = np.nanstd(VI_bi_week)
        std = float(std)
        if std == 0:
            return None
        xnew, ynew = self.__interp(VI_bi_week)
        ynew = np.array([ynew])
        results = HANTS().hants(sample_count=365, inputs=ynew, low=-10000, high=10000,
                        fit_error_tolerance=std)
        result = results[0]

        return result
    def double_logistic(self,t, v1, v2, m1, n1, m2, n2):
        return v1 + v2 * (1 / (1 + np.exp(-m1 * (t - n1))) - 1 / (1 + np.exp(-m2 * (t - n2))))
    def asymmetric_gaussian(self,t, a1, mu1, sigma1, a2, mu2, sigma2):
        """Asymmetric Gaussian function with two components."""
        return (a1 * np.exp(-((t - mu1) ** 2) / (2 * sigma1 ** 2)) +
                a2 * np.exp(-((t - mu2) ** 2) / (2 * sigma2 ** 2)))
    def reproject_modis(self,input_file, output_file, target_projection='EPSG:4326'):
        subprocess.run([
            'gdalwarp',
            '-t_srs', target_projection,
            input_file, output_file
        ])
        print(f"Reprojected file saved as {output_file}")
    def get_series_mean_initial_parameters(self, plot,position,plot_year):
        '''
        帮助get_curve_initial_parameters_n1n2这个函数获取初始soseos均值（在元数据不可用情况下）
        :param plot:
        :param position:
        :return:
        '''
        # years = np.arange(2001,2022)
        # sos_series,eos_series = [],[]
        # for year in years:
        #    reprojected_path = os.path.join(r'D:\Data Collection\RS\MODIS\phenology/{}'.format(plot),
        #                                  '{}_01_01_days_interpolation_reprojection.tif'.format(year))
        #    if not os.path.exists(reprojected_path):
        #       self.reproject_modis(os.path.join(r'D:\Data Collection\RS\MODIS\phenology/{}'.format(plot),
        #                                         '{}_01_01_days_interpolation.tif'.format(year)), reprojected_path)
        #    # modis_phenology_500m_reprojected = gdal.Open(reprojected_path)
        #    #
        #    # modis_phenology_500m_reprojected_sos = modis_phenology_500m_reprojected.GetRasterBand(1).ReadAsArray()
        #    # modis_phenology_500m_reprojected_eos = modis_phenology_500m_reprojected.GetRasterBand(3).ReadAsArray()
        #    test = rioxarray.open_rasterio(reprojected_path)
        #
        #    test_sos = test.sel(band=1).values
        #    test_eos = test.sel(band=3).values
        #
        #    # print(str(modis_phenology_500m_reprojected_eos.shape == original_indices_data.ReadAsArray().shape) + '*'*20)
        #    sos, eos = test_sos[position[0], position[1]], \
        #    test_eos[position[0], position[1]]
        #    if sos >= eos:continue
        #    sos_series.append(sos)
        #    eos_series.append(eos)
        modis_phenology_500m_reprojected = xarray.open_dataset(os.path.join(r'D:\Data Collection\RS\MODIS\phenology/{}'.format(plot),'days_interpolation_reprojection_summary.nc'))
        modis_phenology_500m_reprojected_sos = modis_phenology_500m_reprojected['sos'][:,position[0],position[1]].values
        modis_phenology_500m_reprojected_eos = modis_phenology_500m_reprojected['eos'][:,position[0],position[1]].values
        valid_indices = modis_phenology_500m_reprojected_sos < modis_phenology_500m_reprojected_eos
        filtered_sos_values = modis_phenology_500m_reprojected_sos[valid_indices]
        filtered_eos_values = modis_phenology_500m_reprojected_eos[valid_indices]
        if filtered_sos_values.shape[0] != 0 and filtered_eos_values.shape[0] != 0 : return int(np.mean(filtered_sos_values)),int(np.mean(filtered_eos_values))
        else:
           # reprojected_path = os.path.join(r'D:\Data Collection\RS\MODIS\phenology/{}'.format(plot),
           #                               '{}_01_01_days_interpolation_reprojection.tif'.format(int(plot_year)))
           # # modis_phenology_500m_reprojected = gdal.Open(reprojected_path)
           # # modis_phenology_500m_reprojected_sos = modis_phenology_500m_reprojected.GetRasterBand(1).ReadAsArray()
           # # modis_phenology_500m_reprojected_eos = modis_phenology_500m_reprojected.GetRasterBand(3).ReadAsArray()
           # test = rioxarray.open_rasterio(reprojected_path)
           #
           # test_sos = test.sel(band=1).values
           # test_eos = test.sel(band=3).values

           radius = 100  #选取对应点周围一百个像素点的范围

           # 确定子区域的边界
           xmin = max(0, position[0] - radius)
           xmax = min(modis_phenology_500m_reprojected['sos'].shape[1] - 1, position[0] + radius)
           ymin = max(0, position[1] - radius)
           ymax = min(modis_phenology_500m_reprojected['sos'].shape[2] - 1, position[1] + radius)
           # test_sos = test_sos[xmin:xmax + 1, ymin:ymax + 1]
           # test_eos = test_eos[xmin:xmax + 1, ymin:ymax + 1]
           modis_phenology_500m_reprojected_sos = modis_phenology_500m_reprojected['sos'][min(20,int(plot_year)-2001),xmin:xmax+1,ymin:ymax+1].values    #因为plotyear是从2001年开始，但是nc是从0开始，所以减去2001
           modis_phenology_500m_reprojected_eos = modis_phenology_500m_reprojected['eos'][min(20, int(plot_year) - 2001),xmin:xmax+1, ymin:ymax+1].values

           try:
               return int(modis_phenology_500m_reprojected_sos[modis_phenology_500m_reprojected_sos!=261].mean()),int(modis_phenology_500m_reprojected_eos[modis_phenology_500m_reprojected_eos!=261].mean())
           except:
               return 140,280
    def get_curve_initial_parameters_n1n2(self,plot,year,position):
        '''
        获取曲线拟合的初始参数，即MODIS500M分辨率物候产品的参数
        :param plot:
        :param position:
        :return:
        '''

        year_phenology = year
        # modis_phenology_500m = gdal.Open(os.path.join(r'D:\Data Collection\RS\MODIS\phenology/{}'.format(plot),'{}_01_01_days_interpolation.tif'.format(year[:4])))
        original_indices_data = gdal.Open(os.path.join(r'D:\Data Collection\RS\MODIS\EVI/{}'.format(plot),'{}_reprojection.tif'.format(year)))
        # 重投影并保存到新文件
        if int(year[:4]) > 2021: year_phenology = year_phenology.replace(year[:4],'2021')

        # reprojected_path = os.path.join(r'D:\Data Collection\RS\MODIS\phenology/{}'.format(plot),'{}_01_01_days_interpolation_reprojection.tif'.format(year_phenology[:4]))
        # test = gdal.Open(reprojected_path)
        # try:
        #     test_sos = test.GetRasterBand(1).ReadAsArray()
        #     test_eos = test.GetRasterBand(3).ReadAsArray()
        # except:
        #     print(reprojected_path)
        modis_phenology_500m_reprojected = xarray.open_dataset(os.path.join(r'D:\Data Collection\RS\MODIS\phenology/{}'.format(plot),'days_interpolation_reprojection_summary.nc'))
        modis_phenology_500m_reprojected_sos = modis_phenology_500m_reprojected['sos'][int(year_phenology[:4])-2001,position[0],position[1]].values
        modis_phenology_500m_reprojected_eos = modis_phenology_500m_reprojected['eos'][int(year_phenology[:4])-2001,position[0],position[1]].values
        # print(str(modis_phenology_500m_reprojected_eos.shape == original_indices_data.ReadAsArray().shape) + '*'*20)
        # sos,eos = test_sos[position[0],position[1]],test_eos[position[0],position[1]]
        sos, eos = int(modis_phenology_500m_reprojected_sos), int(modis_phenology_500m_reprojected_eos)
        if sos == eos or sos > eos:return self.get_series_mean_initial_parameters(plot,position,year[:4])
        else:return [sos,eos]
    def curve_fitting(self,VI_bi_week,i):

        smoothed_VI = self.hants_smooth(VI_bi_week)
        t = np.arange(1,366)
        # smoothed_VI = VI_bi_week
        # t = np.arange(0, len(evi) * 16, 16) + 1
        v1_init = np.percentile(smoothed_VI, 5)
        v2_init = np.percentile(smoothed_VI, 95) - np.percentile(smoothed_VI, 5)
        m1_init, m2_init = 0.05, 0.05
        n1_init, n2_init = 140, 280

        initial_parameters = [v1_init, v2_init, m1_init, m2_init, n1_init, n2_init]

        params, covariance = curve_fit(self.asymmetric_gaussian, t, smoothed_VI, p0=initial_parameters,maxfev=500000)
        fitted_evi = self.asymmetric_gaussian(t, *params)

        # 计算振幅段的阈值
        amplitude_threshold = 0.5 * (v2_init - v1_init)

        # 找到SOS的日期
        sos_index = np.argmax(fitted_evi > v1_init + amplitude_threshold)

        # 找到EOS的日期
        eos_index = len(t) - np.argmax(fitted_evi[::-1] > v1_init + amplitude_threshold) - 1

        SOS_date = t[sos_index]
        EOS_date = t[eos_index]
        if SOS_date == EOS_date or SOS_date > EOS_date or SOS_date + 30 > EOS_date or (
                SOS_date < 20 or EOS_date > 340):
            # 找到SOS的日期
            sos_index = np.argmax(smoothed_VI > v1_init + amplitude_threshold)

            # 找到EOS的日期
            eos_index = len(t) - np.argmax(smoothed_VI[::-1] > v1_init + amplitude_threshold) - 1

            SOS_date = t[sos_index]
            EOS_date = t[eos_index]

        if SOS_date == EOS_date or SOS_date > EOS_date or SOS_date + 30 > EOS_date or (
                SOS_date < 10 or EOS_date > 350):
            # Calculate the curvature of the fitted curve
            curve_curvature = self.curvature(t, fitted_evi)
            curvature_rate_of_change = np.gradient(curve_curvature)

            # Identify the local maxima in the rate of change of curvature
            local_maxima = argrelextrema(curvature_rate_of_change, np.greater)[0]
            if len(local_maxima) > 0:
                SOS_date = t[local_maxima[0]]
                EOS_date = t[local_maxima[-1]]
            else:
                SOS_date, EOS_date = None, None
        if SOS_date == EOS_date or SOS_date > EOS_date or SOS_date + 30 > EOS_date or (
                SOS_date < 10 and EOS_date > 360): SOS_date, EOS_date = None, None
        if SOS_date != None and EOS_date != None:
            plt.figure(figsize=(10, 6))
            plt.plot(np.arange(0, 23 * 16, 16)+1, VI_bi_week, 'o', label='Original Data Points')
            plt.plot(t, smoothed_VI, '-', label='HANTS Smoothed Data')
            plt.plot(t, fitted_evi, '--', label='Fitted Curve')
            plt.axvline(SOS_date, color='g', linestyle='--', label='SOS (Start of Season)')
            plt.axvline(EOS_date, color='r', linestyle='--', label='EOS (End of Season)')
            params_text = (f"Initial Parameters:\n"
                           f"v1: {v1_init:.2f}, v2: {v2_init:.2f}\n"
                           f"m1: {m1_init:.2f}, m2: {m2_init:.2f}\n"
                           f"n1: {n1_init:.2f}, n2: {n2_init:.2f}")
            plt.text(0.02, 0.02, params_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

            plt.xlabel('Day of Year')
            plt.ylabel('Vegetation Index')
            plt.title(f'Curve Fitting')
            plt.legend()
            plt.grid(True)
            plt.savefig(r'temp/test/with hants/gaussian/{}.jpg'.format(str(i)))
    def curvature(self,x, y):
        """Calculate the curvature of the curve."""
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(ddx * dy - dx * ddy) / (dx ** 2 + dy ** 2) ** 1.5
        return curvature
    def curve_fitting_table(self, row,year_columns):

        VI_bi_week = row[year_columns]/10000
        plot = int(row['plot'])
        year = year_columns[0]
        position = [int(row['row']),int(row['col'])]

        smoothed_VI = self.hants_smooth(VI_bi_week)
        t = np.arange(1, 366)
        v1_init = np.percentile(smoothed_VI, 5)
        v2_init = np.percentile(smoothed_VI, 95) - np.percentile(smoothed_VI, 5)
        m1_init, m2_init = 0.05, 0.05
        # n1_init, n2_init = 140, 280
        n1_init_n2_init = self.get_curve_initial_parameters_n1n2(plot, year,position)
        n1_init, n2_init = n1_init_n2_init[0], n1_init_n2_init[1]
        initial_parameters = [v1_init, v2_init, m1_init, m2_init, n1_init, n2_init]
        try:
            params, covariance = curve_fit(self.asymmetric_gaussian, t, smoothed_VI, p0=initial_parameters,
                                           maxfev=10000)
            fitted_evi = self.asymmetric_gaussian(t, *params)
        except:
            fitted_evi = smoothed_VI
        # 计算振幅段的阈值
        amplitude_threshold = 0.5 * (max(fitted_evi) - min(fitted_evi))

        # 找到SOS的日期
        sos_index = np.argmax(fitted_evi > v1_init + amplitude_threshold)

        # 找到EOS的日期
        eos_index = len(t) - np.argmax(fitted_evi[::-1] > v1_init + amplitude_threshold) - 1

        SOS_date = t[sos_index]
        EOS_date = t[eos_index]
        if SOS_date == EOS_date or SOS_date > EOS_date or SOS_date + 30 > EOS_date or (
                SOS_date < 20 or EOS_date > 340):
            # 找到SOS的日期
            sos_index = np.argmax(smoothed_VI > v1_init + amplitude_threshold)

            # 找到EOS的日期
            eos_index = len(t) - np.argmax(smoothed_VI[::-1] > v1_init + amplitude_threshold) - 1

            SOS_date = t[sos_index]
            EOS_date = t[eos_index]

        if SOS_date == EOS_date or SOS_date > EOS_date or SOS_date + 30 > EOS_date or (
                SOS_date < 10 or EOS_date > 350):
            # Calculate the curvature of the fitted curve
            curve_curvature = self.curvature(t, fitted_evi)
            curvature_rate_of_change = np.gradient(curve_curvature)

            # Identify the local maxima in the rate of change of curvature
            local_maxima = argrelextrema(curvature_rate_of_change, np.greater)[0]
            if len(local_maxima) > 0:
                SOS_date = t[local_maxima[0]]
                EOS_date = t[local_maxima[-1]]
            else:
                SOS_date, EOS_date = int(n1_init), int(n2_init)
        if SOS_date == EOS_date or SOS_date > EOS_date or SOS_date + 30 > EOS_date or (
                SOS_date < 5 or EOS_date > 360 or EOS_date < 50) :
            SOS_date, EOS_date = None,None
        # if SOS_date != None and EOS_date != None and random.random() < 0.001:
        #     plt.figure(figsize=(10, 6))
        #     plt.plot(np.arange(0, 23 * 16, 16) + 1, VI_bi_week, 'o', label='Original Data Points')
        #     plt.plot(t, smoothed_VI, '-', label='HANTS Smoothed Data')
        #     plt.plot(t, fitted_evi, '--', label='Fitted Curve')
        #     plt.axvline(SOS_date, color='g', linestyle='--', label='SOS (Start of Season)')
        #     plt.axvline(EOS_date, color='r', linestyle='--', label='EOS (End of Season)')
        #     params_text = (f"Initial Parameters:\n"
        #                    f"v1: {v1_init:.2f}, v2: {v2_init:.2f}\n"
        #                    f"m1: {m1_init:.2f}, m2: {m2_init:.2f}\n"
        #                    f"n1: {n1_init:.2f}, n2: {n2_init:.2f}"
        #                    f"n1: {SOS_date:.2f}, n2: {EOS_date:.2f}")
        #     plt.text(0.02, 0.02, params_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
        #
        #     plt.xlabel('Day of Year')
        #     plt.ylabel('Vegetation Index')
        #     plt.title(f'Curve Fitting for Plot {plot}, Year {year}')
        #     plt.legend()
        #     plt.grid(True)
        #     plt.savefig(r'temp/test/Plot_{}_{}_{}_Year_{}.jpg'.format(str(plot), str(row['row']), str(row['col']),
        #                                                                            str(year[:4])))

        return [SOS_date,EOS_date]
    def pick_early_peak_late_dormant_period(self,NDVI_daily,threshold=0.3):
        '''
        :param NDVI_daily: 365-day NDVI time series
        :param threshold: SOS and EOS threshold of minimum NDVI plus the 30% of the seasonal amplitude for multiyear NDVI
        :return: details of phenology
        '''
        peak = np.argmax(NDVI_daily)
        if peak == 0 or peak == (len(NDVI_daily)-1):
            raise
        try:
            early_start = self.__search_SOS(NDVI_daily, peak, threshold)
            late_end = self.__search_EOS(NDVI_daily, peak, threshold)
        except:
            early_start = np.nan
            late_end = np.nan
        # method 1
        # early_end, late_start = self.__slope_early_late(vals,early_start,late_end,peak) # unstable
        # method 2
        early_end, late_start = self.__median_early_late(NDVI_daily,early_start,late_end,peak) # choose the median value before and after the peak

        early_length = early_end - early_start
        mid_length = late_start - early_end
        late_length = late_end - late_start
        dormant_length = 365 - (late_end - early_start)

        result = {
            'early_length':early_length,
            'mid_length':mid_length,
            'late_length':late_length,
            'dormant_length':dormant_length,
            'early_start':early_start,
            'early_start_mon':self.__doy_to_month(early_start),
            'early_end':early_end,
            'early_end_mon':self.__doy_to_month(early_end),
            'peak':peak,
            'peak_mon':self.__doy_to_month(peak),
            'late_start':late_start,
            'late_start_mon':self.__doy_to_month(late_start),
            'late_end':late_end,
            'late_end_mon':self.__doy_to_month(late_end),
            'growing_season':list(range(early_start,late_end)),
            'growing_season_mon':[self.__doy_to_month(i) for i in range(early_start,late_end)],
            'dormant_season':[i for i in range(0,early_start)]+[i for i in range(late_end,365)],
            'dormant_season_mon':[self.__doy_to_month(i) for i in range(0,early_start)]+[self.__doy_to_month(i) for i in range(late_end,365)],
        }
        return result
    def __doy_to_month(self,doy):
        '''
        :param doy: day of year
        :return: month
        '''
        base = datetime.datetime(2000,1,1)
        time_delta = datetime.timedelta(int(doy))
        date = base + time_delta
        month = date.month
        day = date.day
        if day > 15:
            month = month + 1
        if month >= 12:
            month = 12
        return month
    def __interp(self, vals):
        '''
        :param vals: bi-weekly NDVI values
        :return: 365-day NDVI time series with linear interpolation
        '''
        inx = list(range(len(vals)))
        iny = vals
        x_new = np.linspace(min(inx), max(inx), 365)
        func = interpolate.interp1d(inx, iny)
        y_new = func(x_new)
        return x_new, y_new
    def __search_SOS(self, vals, maxind, threshold_i):
        '''
        :param vals: 365-day NDVI time series
        :param maxind: the index of the peak value
        :param threshold_i: threshold of minimum NDVI plus the 30% of the seasonal amplitude for multiyear NDVI
        :return: the index of the Start of Season (SOS)
        '''
        left_vals = vals[:maxind]
        left_min = np.min(left_vals)
        max_v = vals[maxind]
        if left_min < 2000: # for NDVI, 2000 is equivalent to 0.2
            left_min = 2000
        threshold = (max_v - left_min) * threshold_i + left_min

        ind = 999999
        for step in range(365):
            ind = maxind - step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold:
                break

        return ind
    def __search_EOS(self, vals, maxind, threshold_i):
        '''
        :param vals: 365-day NDVI time series
        :param maxind: the index of the peak value
        :param threshold_i: threshold of minimum NDVI plus the 30% of the seasonal amplitude for multiyear NDVI
        :return: the index of the End of Season (EOS)
        '''
        right_vals = vals[maxind:]
        right_min = np.min(right_vals)
        max_v = vals[maxind]
        if right_min < 2000: # for NDVI, 2000 is equivalent to 0.2
            right_min = 2000
        threshold = (max_v - right_min) * threshold_i + right_min

        ind = 999999
        for step in range(365):
            ind = maxind + step
            if ind >= 365:
                break
            val_s = vals[ind]
            if val_s <= threshold: # stop search when the value is lower than threshold
                break
        return ind
    def __slope_early_late(self,vals,sos,eos,peak):
        slope_left = []
        for i in range(sos,peak):
            if i-1 < 0:
                slope_i = vals[1]-vals[0]
            else:
                slope_i = vals[i]-vals[i-1]
            slope_left.append(slope_i)

        slope_right = []
        for i in range(peak,eos):
            if i-1 < 0:
                slope_i = vals[1]-vals[0]
            else:
                slope_i = vals[i]-vals[i-1]
            slope_right.append(slope_i)

        max_ind = np.argmax(slope_left) + sos
        min_ind = np.argmin(slope_right) + peak

        return max_ind, min_ind
    def __median_early_late(self,vals,sos,eos,peak):
        '''
        :param vals: 365-day NDVI time series
        :param sos: the index of the Start of Season (SOS)
        :param eos: the index of the End of Season (EOS)
        :param peak: the index of the peak index
        :return: the index of the early end and late start
        '''
        median_left = int((peak-sos)/2.)
        median_right = int((eos - peak)/2.)
        max_ind = median_left + sos
        min_ind = median_right + peak
        return max_ind, min_ind


if __name__ == '__main__':

    '''# 干旱区域物候指标提取'''
    legacy = legacy_effects()
    '''
    2024-08-13,之前是对整个裁剪区域内进行分析，现在对不同国家进行分析，最后汇总
    添加数据是土地利用数据，干扰数据
    '''
    # 根据土地利用数据，干扰数据，DEM数据筛选标准如下：
    '''
    1.土地利用中选择 Broad-leaved forest(23)  Mixed forest(25)
    2. 高程低于800米
    扰动数据做记录
    土地利用阈值设为比例超过百分八十的点
    扰动设为扰动少于百分之20
    '''
    # country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
    #                 os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    #
    # DEM_path = r'D:\Data Collection\DEM/Europe_DEM.tif'
    # drought_years = [2003,2015,2018,2019,2022]
    # phenology_band = 1          #1是SOS， 3是EOS
    # for country_tif in tqdm(country_tifs):
    #    if 'ukraine' in country_tif or 'belarus' in country_tif: continue
    #    # if 'germany' in country_tif:
    #    country_name = os.path.split(country_tif)[-1]
    #    print(country_name)
    #    # for drought_year in drought_years:
    #    evi_path = os.path.join(country_tif,country_name+'_2001_EVI.tif')
    #    landcover_path = os.path.join(country_tif,country_name+'_land_cover.tif')
    #    disturbance_path = os.path.join(country_tif,'disturbance_year_1986-2020_{}_reprojection.tif'.format(country_name))
    #    mask_out_path = {'landcover_mask':os.path.join(country_tif,r'Wu Yong/low_elevation_everygreen/mask_landcover.tif'),
    #                     'DEM_mask': os.path.join(country_tif, r'Wu Yong/low_elevation_everygreen/mask_DEM.tif'),
    #                     f'combined_mask':os.path.join(country_tif,r'Wu Yong/low_elevation_everygreen/mask_combined.tif')}
    #    if not os.path.exists(os.path.split(mask_out_path['landcover_mask'])[0]):os.makedirs(os.path.split(mask_out_path['landcover_mask'])[0])
    #    legacy.generate_mask(evi_path, phenology_band,landcover_path, disturbance_path,DEM_path,mask_out_path)

    '''
    基于筛除扰动后的数据和森林区域进行干旱遗留效应的统计建模
    1. 获取对应点的SPEI,GDD,VPD,TM
    2. 统计建模
    '''
    '''# # 1. 获取对应点的SPEI数据和GDD数据和温度'''
    country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
                    os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    gdd_path = r'D:\Data Collection\Temperature'
    SPEI_paths = [r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei03.nc',
                 r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei06.nc']
    tem_path = r'D:\Data Collection\Temperature\T2M\T2M_2000to2023.nc'
    VPD_path = r'D:\Data Collection\Temperature\VPD/VPD_2000to2023.nc'
    TP_path = r'D:\Data Collection\other_factors/total_precipitation.nc'
    TP_path_2224 = r'D:\Data Collection\other_factors/total_precipitation_22_24.nc'
    SIF_path = r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\2 forest sensitivity to water\data collection/SIF_tempory.nc'
    chillingdays_data_path = r'D:\Data Collection\Temperature\chillingdays'
    chillingdays_paths = glob.glob(os.path.join(chillingdays_data_path,'*.nc'))
    drought_years = [2003, 2015, 2018,2019,2022]
    norway_index = None
    # for i, country_tif in enumerate(country_tifs):
    #     if 'ukraine' in country_tif:
    #         norway_index = i
    #         break
    block_size = 2
    # country_tif = r'D:\\Data Collection\\RS\\Disturbance\\7080016\\germany'
    # country_name = os.path.split(country_tif)[-1]
    # # if i <= norway_index:
    # #     print(country_name+'跳过')
    # #     continue
    # print(country_name)
    # evi_tifs = glob.glob(os.path.join(country_tif, '*EVI*.tif'))
    # evi_tifs = sorted(evi_tifs, key=lambda x: int(x.split('_EVI')[0][-4:]))  # 按照年份排序
    # mask_tifs = [os.path.join(country_tif, 'Wu Yong/mask_combined.tif')]
    # temp_dir = os.path.join(country_tif, 'Wu Yong')
    # chunk_dict = defaultdict(dict)
    # for evi_tif in evi_tifs:
    #     # 打开原始EVI文件
    #     with rasterio.open(evi_tif) as src:
    #         # 获取图像尺寸
    #         height, width = src.shape
    #         # 计算分块大小
    #         chunk_height = height // block_size
    #         chunk_width = width // block_size
    #
    #         # 创建分块文件夹
    #         base_name = os.path.basename(evi_tif).split('.')[0]
    #         chunk_dir = os.path.join(temp_dir, base_name)
    #         os.makedirs(chunk_dir, exist_ok=True)
    #
    #         # 分块处理
    #         for i in range(block_size):
    #             for j in range(block_size):
    #
    #                 chunk_path = os.path.join(chunk_dir, f'chunk_{i}_{j}.tif')
    #                 if 'evi' not in chunk_dict[(i, j)]:
    #                     chunk_dict[(i, j)]['evi'] = []
    #                 chunk_dict[(i, j)]['evi'].append(chunk_path)
    #
    # chunk_dirs = sorted(glob.glob(os.path.join(temp_dir, '*')))
    # all_chunk_paths = []
    # for chunk_dir in chunk_dirs:
    #     chunks = sorted(glob.glob(os.path.join(chunk_dir, '*.tif')))
    #     all_chunk_paths.extend(chunks)
    #
    # mask_chunks = []
    # for mask_tif in mask_tifs:
    #     with rasterio.open(mask_tif) as src:
    #         for i in range(block_size):
    #             for j in range(block_size):
    #                 # 计算窗口位置（使用与EVI相同的分块逻辑）
    #
    #                 chunk_path = os.path.join(temp_dir, f'mask_chunk_{i}_{j}.tif')
    #                 mask_chunks.append(chunk_path)
    #                 chunk_dict[(i, j)]['mask'] = [chunk_path]
    #
    # for (i, j), data_paths in chunk_dict.items():
    #     print(f"Processing chunk ({i}, {j})...")
    #
    #     # 确保所有年份的EVI文件按时间排序
    #     evi_chunks = sorted(data_paths['evi'], key=lambda x: int(os.path.split(x)[0].split('_EVI')[0][-4:]))
    #
    #     # 调用处理函数
    #     usless = legacy.summarize_raster_by_mask_aggregation(
    #         evi_chunks,  # 该位置所有年份的EVI分块
    #         data_paths['mask'],  # 该位置的mask分块
    #         SPEI_paths, tem_path, VPD_path, SIF_path, TP_path, TP_path_2224, f'{i}_{j}'
    #     )
    for i,country_tif in tqdm(enumerate(country_tifs)):
        # if 'ukraine' in country_tif or 'belarus' in country_tif or 'andorra' in country_tif or 'albania' in country_tif or 'austria' in country_tif or 'belgium' in country_tif or 'bosniaherzegovina' in country_tif        or 'bulgaria' in country_tif   or 'croatia' in country_tif or 'czechia' in country_tif or 'denmark' in country_tif or 'estonia' in country_tif or 'finland' in country_tif       or 'france' in country_tif or 'germany' in country_tif or 'greece' in country_tif or 'hungary' in country_tif or 'ireland' in country_tif or 'italy' in country_tif or 'latvia' in country_tif :continue
       # if country_tif == country_tifs[0]:

       if ('ukraine' in country_tif or 'belarus' in country_tif 
              or 'andorra' in country_tif or 'liechtenstein' in country_tif or 'luxembourg' in country_tif):continue
       country_name = os.path.split(country_tif)[-1]
       # if i <= norway_index:
       #     print(country_name+'跳过')
       #     continue
       print(country_name)
       evi_tifs = glob.glob(os.path.join(country_tif,'*EVI*.tif'))
       evi_tifs = sorted(evi_tifs, key=lambda x: int(x.split('_EVI')[0][-4:]))          #按照年份排序
       supplement_folder = 'high_elevation_broad'
       mask_tifs = [os.path.join(country_tif,f'Wu Yong/{supplement_folder}/mask_combined.tif')]
       if os.path.exists(os.path.join(os.path.split(mask_tifs[0])[0],f'inform_sum_sa_EVI_SIF_all.csv')):continue
       else:
           if country_name not in ['norway','finland','france','italy','romania','spain','sweden','unitedkingdom']:
               legacy.summarize_raster_by_mask_aggregation(evi_tifs, mask_tifs, SPEI_paths, tem_path, VPD_path,
                                                           SIF_path, TP_path, TP_path_2224, 'all')
           else:
            temp_dir = os.path.join(country_tif, f'Wu Yong/{supplement_folder}')
            chunk_dict = defaultdict(dict)
            position_dict = {}  # 存储每个子块在原始图像中的位置信息

            for evi_tif in evi_tifs:
                # 打开原始EVI文件
                with rasterio.open(evi_tif) as src:
                    # 获取图像尺寸
                    height, width = src.shape
                    # 计算分块大小
                    chunk_height = height // block_size
                    chunk_width = width // block_size

                    # 创建分块文件夹
                    base_name = os.path.basename(evi_tif).split('.')[0]
                    chunk_dir = os.path.join(os.path.split(temp_dir)[0], base_name)
                    os.makedirs(chunk_dir, exist_ok=True)

                    # 分块处理
                    for i in range(block_size):
                        for j in range(block_size):
                            # 计算窗口位置
                            yoff = i * chunk_height
                            xoff = j * chunk_width
                            win_height = chunk_height if i < block_size - 1 else height - yoff
                            win_width = chunk_width if j < block_size - 1 else width - xoff
                            if not os.path.exists(os.path.join(chunk_dir, f'chunk_{i}_{j}.tif')):
                                # 记录位置信息（在原始图像中的行列范围）
                                position_key = (i, j)
                                if position_key not in position_dict:
                                    position_dict[position_key] = {
                                        'start_row': yoff,
                                        'end_row': yoff + win_height - 1,
                                        'start_col': xoff,
                                        'end_col': xoff + win_width - 1,
                                        'height': win_height,
                                        'width': win_width
                                    }

                                # 创建读取窗口
                                window = Window(xoff, yoff, win_width, win_height)

                                # 读取数据
                                data = src.read(window=window)

                                # 更新元数据
                                profile = src.profile
                                profile.update({
                                    'height': win_height,
                                    'width': win_width,
                                    'transform': src.window_transform(window)
                                })

                                # 写入分块文件
                                chunk_path = os.path.join(chunk_dir, f'chunk_{i}_{j}.tif')
                                with rasterio.open(chunk_path, 'w', **profile) as dst:
                                    dst.write(data)
                            chunk_path = os.path.join(chunk_dir, f'chunk_{i}_{j}.tif')
                            if 'evi' not in chunk_dict[(i, j)]:
                                chunk_dict[(i, j)]['evi'] = []
                            chunk_dict[(i, j)]['evi'].append(chunk_path)

            # 获取所有分块文件路径（按原始排序）
            position_file = os.path.join(temp_dir, f"{country_name}_chunk_positions.json")
            if not os.path.exists(position_file):
                with open(position_file, 'w') as f:
                    import json

                    # 将位置信息转换为可序列化格式
                    serializable_positions = {}
                    for key, info in position_dict.items():
                        i, j = key
                        serializable_positions[f"{i}_{j}"] = info
                    json.dump(serializable_positions, f, indent=2)
                print(f"Chunk position information saved to: {position_file}")
            chunk_dirs = sorted(glob.glob(os.path.join(temp_dir, '*')))
            all_chunk_paths = []
            for chunk_dir in chunk_dirs:
                chunks = sorted(glob.glob(os.path.join(chunk_dir, '*.tif')))
                all_chunk_paths.extend(chunks)

            # 对其他数据执行相同的分块操作（这里以mask为例）
            mask_chunks = []
            for mask_tif in mask_tifs:
                with rasterio.open(mask_tif) as src:
                    for i in range(block_size):
                        for j in range(block_size):
                            # 计算窗口位置（使用与EVI相同的分块逻辑）
                            position_key = (i, j)
                            yoff = i * (src.height // block_size)
                            xoff = j * (src.width // block_size)
                            win_height = src.height // block_size if i < block_size - 1 else src.height - yoff
                            win_width = src.width // block_size if j < block_size - 1 else src.width - xoff

                            window = Window(xoff, yoff, win_width, win_height)
                            data = src.read(window=window)

                            profile = src.profile
                            profile.update({
                                'height': win_height,
                                'width': win_width,
                                'transform': src.window_transform(window)
                            })

                            chunk_path = os.path.join(temp_dir, f'mask_chunk_{i}_{j}.tif')
                            with rasterio.open(chunk_path, 'w', **profile) as dst:
                                dst.write(data)
                            mask_chunks.append(chunk_path)
                            chunk_dict[(i, j)]['mask'] = [chunk_path]

            # 调用处理函数（传入分块数据）
            for (i, j), data_paths in chunk_dict.items():
                print(f"Processing chunk ({i}, {j})...")
                if os.path.exists(
                    os.path.join(temp_dir,f'inform_sum_sa_EVI_SIF_{i}_{j}.csv')): continue
                # 确保所有年份的EVI文件按时间排序
                evi_chunks = sorted(data_paths['evi'], key=lambda x: int(os.path.split(x)[0].split('_EVI')[0][-4:]))

                # 调用处理函数
                usless = legacy.summarize_raster_by_mask_aggregation(
                    evi_chunks,  # 该位置所有年份的EVI分块
                    data_paths['mask'],  # 该位置的mask分块
                    SPEI_paths, tem_path, VPD_path, SIF_path, TP_path, TP_path_2224, f'{i}_{j}'
                )
            print(f"All chunks processed for {country_name}")
       # mask_array = gdal.Open(os.path.join(country_tif,'Wu Yong/mask_combined.tif')).ReadAsArray()
       # if mask_array.shape[0] < 1500 and mask_array.shape[1] < 1500:
       #  legacy.summarize_raster_by_mask_aggregation(evi_tifs,mask_tifs,SPEI_paths,tem_path,VPD_path,SIF_path,TP_path,TP_path_2224,'all')
       # else:
       #     print(country_name + 'chunking reqquired')
       #     temp_dir = os.path.join(country_tif, 'Wu Yong')
       #     chunk_dict = defaultdict(dict)
       #     position_dict = {}  # 存储每个子块在原始图像中的位置信息
       #
       #     for evi_tif in evi_tifs:
       #         # 打开原始EVI文件
       #         with rasterio.open(evi_tif) as src:
       #             # 获取图像尺寸
       #             height, width = src.shape
       #             # 计算分块大小
       #             chunk_height = height // block_size
       #             chunk_width = width // block_size
       #
       #             # 创建分块文件夹
       #             base_name = os.path.basename(evi_tif).split('.')[0]
       #             chunk_dir = os.path.join(temp_dir, base_name)
       #             os.makedirs(chunk_dir, exist_ok=True)
       #
       #             # 分块处理
       #             for i in range(block_size):
       #                 for j in range(block_size):
       #                     # 计算窗口位置
       #                     yoff = i * chunk_height
       #                     xoff = j * chunk_width
       #                     win_height = chunk_height if i < block_size - 1 else height - yoff
       #                     win_width = chunk_width if j < block_size - 1 else width - xoff
       #
       #                     # 记录位置信息（在原始图像中的行列范围）
       #                     position_key = (i, j)
       #                     if position_key not in position_dict:
       #                         position_dict[position_key] = {
       #                             'start_row': yoff,
       #                             'end_row': yoff + win_height - 1,
       #                             'start_col': xoff,
       #                             'end_col': xoff + win_width - 1,
       #                             'height': win_height,
       #                             'width': win_width
       #                         }
       #
       #                     # 创建读取窗口
       #                     window = Window(xoff, yoff, win_width, win_height)
       #
       #                     # 读取数据
       #                     data = src.read(window=window)
       #
       #                     # 更新元数据
       #                     profile = src.profile
       #                     profile.update({
       #                         'height': win_height,
       #                         'width': win_width,
       #                         'transform': src.window_transform(window)
       #                     })
       #
       #                     # 写入分块文件
       #                     chunk_path = os.path.join(chunk_dir, f'chunk_{i}_{j}.tif')
       #                     with rasterio.open(chunk_path, 'w', **profile) as dst:
       #                         dst.write(data)
       #                     if 'evi' not in chunk_dict[(i, j)]:
       #                         chunk_dict[(i, j)]['evi'] = []
       #                     chunk_dict[(i, j)]['evi'].append(chunk_path)
       #
       #     # 获取所有分块文件路径（按原始排序）
       #     position_file = os.path.join(temp_dir, f"{country_name}_chunk_positions.json")
       #     with open(position_file, 'w') as f:
       #         import json
       #
       #         # 将位置信息转换为可序列化格式
       #         serializable_positions = {}
       #         for key, info in position_dict.items():
       #             i, j = key
       #             serializable_positions[f"{i}_{j}"] = info
       #         json.dump(serializable_positions, f, indent=2)
       #     print(f"Chunk position information saved to: {position_file}")
       #     chunk_dirs = sorted(glob.glob(os.path.join(temp_dir, '*')))
       #     all_chunk_paths = []
       #     for chunk_dir in chunk_dirs:
       #         chunks = sorted(glob.glob(os.path.join(chunk_dir, '*.tif')))
       #         all_chunk_paths.extend(chunks)
       #
       #     # 对其他数据执行相同的分块操作（这里以mask为例）
       #     mask_chunks = []
       #     for mask_tif in mask_tifs:
       #         with rasterio.open(mask_tif) as src:
       #             for i in range(block_size):
       #                 for j in range(block_size):
       #                     # 计算窗口位置（使用与EVI相同的分块逻辑）
       #                     position_key = (i, j)
       #                     yoff = i * (src.height // block_size)
       #                     xoff = j * (src.width // block_size)
       #                     win_height = src.height // block_size if i < block_size - 1 else src.height - yoff
       #                     win_width = src.width // block_size if j < block_size - 1 else src.width - xoff
       #
       #                     window = Window(xoff, yoff, win_width, win_height)
       #                     data = src.read(window=window)
       #
       #                     profile = src.profile
       #                     profile.update({
       #                         'height': win_height,
       #                         'width': win_width,
       #                         'transform': src.window_transform(window)
       #                     })
       #
       #                     chunk_path = os.path.join(temp_dir, f'mask_chunk_{i}_{j}.tif')
       #                     with rasterio.open(chunk_path, 'w', **profile) as dst:
       #                         dst.write(data)
       #                     mask_chunks.append(chunk_path)
       #                     chunk_dict[(i, j)]['mask'] = [chunk_path]
       #
       #     # 调用处理函数（传入分块数据）
       #     for (i, j), data_paths in chunk_dict.items():
       #         print(f"Processing chunk ({i}, {j})...")
       #
       #         # 确保所有年份的EVI文件按时间排序
       #         evi_chunks = sorted(data_paths['evi'], key=lambda x: int(os.path.split(x)[0].split('_EVI')[0][-4:]))
       #
       #         # 调用处理函数
       #         usless = legacy.summarize_raster_by_mask_aggregation(
       #             evi_chunks,  # 该位置所有年份的EVI分块
       #             data_paths['mask'],  # 该位置的mask分块
       #             SPEI_paths, tem_path, VPD_path, SIF_path, TP_path, TP_path_2224, f'{i}_{j}'
       #         )
       #     print(f"All chunks processed for {country_name}")

