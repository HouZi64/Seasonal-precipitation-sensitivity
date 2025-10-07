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
from matplotlib.path import Path
from matplotlib.patches import PathPatch
os.environ["OMP_NUM_THREADS"] = '7'
from osgeo import gdal,osr, ogr
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
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.model_selection import KFold
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
from collections import Counter
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from scipy.stats import linregress
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import TheilSenRegressor
from sklearn.utils import resample
import os
import math
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap
from matplotlib.path import Path
from matplotlib.gridspec import GridSpec
import rasterio
from rasterio.windows import from_bounds as win_from_bounds
from rasterio.windows import transform as win_transform
from rasterio.plot import plotting_extent
from typing import Dict, Mapping, Optional, List, Tuple, Union
import matplotlib as mpl
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from rasterio.plot import show as rio_show
from shapely.ops import unary_union
from matplotlib import cm, colors
spei_path = r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei03.nc'

EVI_path = r'D:\Data Collection\RS\MODIS\EVI_merge'
tqdm.pandas()
plt.rcParams['font.family'] = 'Arial'


class legacy_effects():
    def __init__(self):
       pass

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
                      landuse_number,all_number,max_category = self.check_landcover(row, col, landcover_array, landcover_geotransform,evi_geotransform,[23,25])
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
       output_nc_path = os.path.join(os.path.split(EVI_files[0])[0],'Wu Yong/EVI.nc')
       if os.path.exists(output_nc_path): return xr.open_dataset(output_nc_path)
       else:
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
           ds.to_netcdf(output_nc_path)
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
        if os.path.exists(os.path.join(os.path.split(evi_path)[0],"Wu Yong/SIF.nc")):
            ds = xr.open_dataset(os.path.join(os.path.split(evi_path)[0],"Wu Yong/SIF.nc"))
        else:
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
            ds.to_netcdf(os.path.join(os.path.split(evi_path)[0],"Wu Yong/SIF.nc"))
        return ds
    def get_SPEI_current_region(self,spei_path,evi_path,spei_scale):
        if os.path.exists(os.path.join(os.path.split(evi_path)[0],"Wu Yong/spei{}_seasonal_sums_2000_2023.nc".format(spei_scale))):
            ds = xr.open_dataset(os.path.join(os.path.split(evi_path)[0],"Wu Yong/spei{}_seasonal_sums_2000_2023.nc".format(spei_scale)))
        else:
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

        if os.path.exists(os.path.join(os.path.split(evi_path)[0],"Wu Yong/tm_seasonal_average_2000_2023.nc")):
            ds = xr.open_dataset(os.path.join(os.path.split(evi_path)[0],"Wu Yong/tm_seasonal_average_2000_2023.nc"))
        else:
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
                    "summerhalf_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                    "winterhalf_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
                    "chilling_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
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
            tm_data = xr.open_dataset(tm_path)
            tm_c = tm_data['t2m'] - 273.15
            for i,year in enumerate(years):

                # 计算全年平均值
                annual_avg = tm_c.sel(time=slice('{}-01-01'.format(year), '{}-12-31'.format(year))).mean(dim='time')

                # 计算春季平均值 (3月1日到5月31日)
                spring_avg = tm_c.sel(time=slice('{}-03-01'.format(year), '{}-05-31'.format(year))).mean(dim='time')

                # 计算夏季平均值 (6月1日到8月31日)
                summer_avg = tm_c.sel(time=slice('{}-06-01'.format(year), '{}-08-31'.format(year))).mean(dim='time')

                # 计算秋季平均值 (9月1日到11月30日)
                autumn_avg = tm_c.sel(time=slice('{}-09-01'.format(year), '{}-11-30'.format(year))).mean(dim='time')
                # 计算夏半年平均值 (4月1日到9月30日)
                summerhalf_avg = tm_c.sel(time=slice('{}-04-01'.format(year), '{}-09-30'.format(year))).mean(dim='time')
                # 计算冬半年平均值 (10月1日到明年的3月30日)
                winterhalf_avg = tm_c.sel(time=slice('{}-10-01'.format(year), '{}-03-30'.format(year+1))).mean(dim='time')
                # 计算寒冷期平均值 (10月1日到明年的3月30日)
                chilling_avg = tm_c.sel(time=slice('{}-11-01'.format(year), '{}-02-28'.format(year+1))).mean(dim='time')

                # 使用坐标从 GDD 数据中提取对应值
                ds['annual_avg'][i] = annual_avg.sel(latitude=lats, longitude=lons, method="nearest")
                ds['spring_avg'][i] = spring_avg.sel(latitude=lats, longitude=lons, method="nearest")
                ds['summer_avg'][i] = summer_avg.sel(latitude=lats, longitude=lons, method="nearest")
                ds['autumn_avg'][i] = autumn_avg.sel(latitude=lats, longitude=lons, method="nearest")

                ds['summerhalf_avg'][i] = summerhalf_avg.sel(latitude=lats, longitude=lons, method="nearest")
                ds['winterhalf_avg'][i] = winterhalf_avg.sel(latitude=lats, longitude=lons, method="nearest")
                ds['chilling_avg'][i] = chilling_avg.sel(latitude=lats, longitude=lons, method="nearest")
            # 保存汇总后的数据为NetCDF文件
            # ds.to_netcdf(os.path.join(os.path.split(evi_path)[0],"Wu Yong/tm_seasonal_average_2000_2023.nc"))
        return ds
    def get_Totalprecipiation_average_current_region(self,tp_path,evi_path,tp_path_22_23):

        if os.path.exists(os.path.join(os.path.split(evi_path)[0],"Wu Yong/tp_seasonal_average_2000_2023.nc")):
            ds = xr.open_dataset(os.path.join(os.path.split(evi_path)[0],"Wu Yong/tp_seasonal_average_2000_2023.nc"))
        else:
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
                winter_avg = tp_c.sel(time=slice('{}-12-01'.format(year), '{}-02-20'.format(year+1))).mean(dim='time')
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
            # ds.to_netcdf(os.path.join(os.path.split(evi_path)[0],"Wu Yong/tp_seasonal_average_2000_2023.nc"))
        return ds
    def get_Totalprecipiation_winter_average_current_region(self,tp_path,evi_path,tp_path_22_23):
        # 由于之前的数据冬季降雨量选择错了，这里重新进行提取，即当年的1，2月，和当年的12月
        if os.path.exists(os.path.join(os.path.split(evi_path)[0],"Wu Yong/tp_winter_average_2000_2023.nc")):
            ds = xr.open_dataset(os.path.join(os.path.split(evi_path)[0],"Wu Yong/tp_winter_average_2000_2023.nc"))
        else:
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
                    "winter_avg": (("year", "lat", "lon"), np.zeros((len(years), len(lats), len(lons)), dtype=np.float32)),
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
                # 计算冬季平均值 (12月1日到2月29日)
                winter_avg = tp_c.sel(
                    time=(tp_c['time.year'] == year) &
                         (tp_c['time.month'].isin([1, 2, 12]))
                ).mean(dim='time')
                # 使用坐标从 GDD 数据中提取对应值
                ds['winter_avg'][i] = winter_avg.sel(latitude=lats, longitude=lons, method="nearest")
            # 保存汇总后的数据为NetCDF文件
            # ds.to_netcdf(os.path.join(os.path.split(evi_path)[0],"Wu Yong/tp_winter_average_2000_2023.nc"))
        return ds

    def get_VPD_average_current_region(self,VPD_path,evi_path):

        if os.path.exists(os.path.join(os.path.split(evi_path)[0],"Wu Yong/VPD_seasonal_average_2000_2023.nc")):
            ds = xr.open_dataset(os.path.join(os.path.split(evi_path)[0],"Wu Yong/VPD_seasonal_average_2000_2023.nc"))
        else:

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
                # 使用坐标从 GDD 数据中提取对应值
                ds['annual_avg'][i] = annual_avg.reindex(lat=lats, lon=lons, method="nearest")
                ds['spring_avg'][i] = spring_avg.reindex(lat=lats, lon=lons, method="nearest")
                ds['summer_avg'][i] = summer_avg.reindex(lat=lats, lon=lons, method="nearest")
                ds['autumn_avg'][i] = autumn_avg.reindex(lat=lats, lon=lons, method="nearest")

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
    def summarize_raster_by_mask_aggregation(self,evi_tifs,mask_tifs,SPEI_paths,temperature_path,VPD_path,SIF_path,TP_path,TP_path_2224):
        if os.path.exists(os.path.join(os.path.split(mask_tifs[0])[0],'inform_sum_sa_EVI_SIF.csv')):return None
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
        # if os.path.exists(os.path.join(os.path.split(mask_tif)[0],'inform_sum_sa.csv')):return None
        try:
            drought_year = re.search(r'_(\d{4})\.tif$', mask_tif).group(1)
        except:drought_year = 2003
        # 看聚合后的栅格中，每个栅格有多少原来的点（经过掩膜后过滤的点）参与了计算,同时绘制直方图
        mask_nc = self.tif_tonc(mask_tif)
        mask_num_nc = self.spatial_aggregation(mask_nc, xr.open_dataset(SPEI_paths[0]), 'sum')
        data_mask_num = mask_num_nc['number']['data'].values.flatten()
        df_mask_num = pd.DataFrame({'grid_id': range(len(data_mask_num)), 'data': data_mask_num})
        # 直接定义阈值为1，后续采用加权回归模型
        mask_num_threshold = 1.0
        mask_array = mask_nc['data'].data
        mask_array_evi = np.repeat(np.expand_dims(mask_array.data,0),23,axis=0)
        mask_array_SIF = np.repeat(np.expand_dims(mask_array.data, 0), 16, axis=0)
        EVI_sa = self.spatial_aggregation(
            EVI_data.where(mask_array_evi == 1),
            xr.open_dataset(SPEI_paths[0]), 'mean')
        del EVI_data
        SIF_sa = self.spatial_aggregation(
            SIF.where(mask_array_SIF == 1),
            xr.open_dataset(SPEI_paths[0]), 'mean')
        del SIF
        mask_array_others = np.repeat(np.expand_dims(mask_array.data, 0), 24, axis=0)     #这里面除了物候数据，其他数据年数都是2000年到2023年，因此统一用一个expandmask的形状就好了
        spei_03_sa = self.spatial_aggregation(spei_03.where(mask_array_others == 1),xr.open_dataset(SPEI_paths[0]), 'mean')
        del spei_03
        spei_06_sa = self.spatial_aggregation(spei_06.where(mask_array_others == 1),xr.open_dataset(SPEI_paths[0]), 'mean')
        del spei_06
        temperature_sa = self.spatial_aggregation(temperature.where(mask_array_others == 1),xr.open_dataset(SPEI_paths[0]), 'mean')
        del temperature
        VPD_sa = self.spatial_aggregation(VPD.where(mask_array_others == 1), xr.open_dataset(SPEI_paths[0]), 'mean')
        del VPD
        TP_sa = self.spatial_aggregation(TP.where(mask_array_others == 1), xr.open_dataset(SPEI_paths[0]), 'mean')
        del TP

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
        df.to_csv(os.path.join(os.path.split(mask_tif)[0],'inform_sum_sa_EVI_SIF.csv'), index=False)
    def summarize_raster_by_mask_aggregation_onlywinterTP(self,evi_tifs,mask_tifs,SPEI_paths,temperature_path,VPD_path,SIF_path,TP_path,TP_path_2224):
        if os.path.exists(os.path.join(os.path.split(mask_tifs[0])[0],'inform_sum_sa_(only_winterTP).csv')):return None
        # 设置波段索引
        band_sos_number = 1  # SOS对应的波段
        band_eos_number = 3  # EOS对应的波段
        # 新建一个吴勇文章的文件夹存放数据，同时新增txt文件，说明该文件夹的作用
        new_path = os.path.join(os.path.split(evi_tifs[5])[0],'Wu Yong')
        if not os.path.exists(new_path):os.makedirs(new_path)
        if not os.path.exists(os.path.join(new_path,'description.txt')):
            with open(os.path.join(new_path,'description.txt'), "w", encoding="utf-8") as file:
                file.write("该文件夹用于存放吴勇论文的各种数据，与遗留效应的数据进行区分")
        TP = self.get_Totalprecipiation_winter_average_current_region(TP_path,evi_tifs[0],TP_path_2224)
        mask_tif = mask_tifs[0]
        # if os.path.exists(os.path.join(os.path.split(mask_tif)[0],'inform_sum_sa.csv')):return None
        try:
            drought_year = re.search(r'_(\d{4})\.tif$', mask_tif).group(1)
        except:drought_year = 2003
        # 看聚合后的栅格中，每个栅格有多少原来的点（经过掩膜后过滤的点）参与了计算,同时绘制直方图
        mask_nc = self.tif_tonc(mask_tif)
        mask_num_nc = self.spatial_aggregation(mask_nc, xr.open_dataset(SPEI_paths[0]), 'sum')
        data_mask_num = mask_num_nc['number']['data'].values.flatten()
        df_mask_num = pd.DataFrame({'grid_id': range(len(data_mask_num)), 'data': data_mask_num})
        # 直接定义阈值为1，后续采用加权回归模型
        mask_num_threshold = 1.0
        mask_array = mask_nc['data'].data

        mask_array_others = np.repeat(np.expand_dims(mask_array.data, 0), 24, axis=0)     #这里面除了物候数据，其他数据年数都是2000年到2023年，因此统一用一个expandmask的形状就好了
        TP_sa = self.spatial_aggregation(TP.where(mask_array_others == 1), xr.open_dataset(SPEI_paths[0]), 'mean')
        del TP
        # 如果该点的有效数据数量小于阈值，就放弃
        rows, cols = np.where(mask_num_nc['number']['data']>mask_num_threshold)
        if len(rows) == 0 and len(cols) == 0: return None

        valid_data_mask = np.zeros(mask_num_nc['number']['data'].shape, dtype=np.uint8)
        # 根据有效数据的位置标记 1
        valid_data_mask[rows, cols] = 1
        '''
        写入csv
        '''
        # 创建一个空的列表来存储所有行的数据
        data= []
        # 定义年份范围
        years = list(range(2000, 2024))  # 2000年到2023年
        for i, (row, col) in tqdm(enumerate(zip(rows, cols))):
            tp_winter = TP_sa['winter_avg'].data[:, row, col]
            # 将数据添加到列表中
            data.append({
                'row': row,
                'col': col,
                'weights':mask_num_nc['ratio']['data'].data[row, col],
                **{f'tp_winter_{year}': tp_winter[i] for i, year in enumerate(years)},
            })
        # 将数据转换为DataFrame
        df = pd.DataFrame(data)
        # 将DataFrame写入CSV文件
        df.to_csv(os.path.join(os.path.split(mask_tif)[0],'inform_sum_sa_(only_winterTP).csv'), index=False)

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

    def random_forest_group(self,target,features,df,year):
        def scatter_density_plot(y_test, y_pred, mse, mae, r2, output_path, year):
            plt.figure(figsize=(8, 6), constrained_layout=True)  # 使用 constrained_layout
            # 创建 2D 直方图，颜色表示像素密度
            hb = plt.hexbin(y_test, y_pred, gridsize=50, cmap='OrRd', norm=LogNorm())
            # 添加颜色条
            cb = plt.colorbar(hb)
            cb.set_label('[pixels]')
            # 添加对角线 y=x 的参考线
            plt.plot([0, np.max([np.max(y_test), np.max(y_pred)])], [0, np.max([np.max(y_test), np.max(y_pred)])],
                     'r--', lw=1.5)
            # 添加统计信息，调整位置
            # 添加统计信息
            plt.text(0.05, 0.95, f'N={len(y_test)}', transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top')
            plt.text(0.05, 0.90, f'R²={r2:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
            plt.text(0.05, 0.85, f'MSE={mse:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
            plt.text(0.05, 0.80, f'MAE={mae:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

            # 设置坐标轴和标题
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.grid(True)
            # 保存图片
            plt.savefig(os.path.join(output_path, f'Density_Scatter_Plot_{year}.png'), dpi=300)
            plt.close()  # 关闭图形以释放内存
        # 定义保存图片路径
        if year == 'Temporal Series':
            output_path = 'images'
        else:
            output_path = 'images window'
        os.makedirs(output_path,exist_ok=True)
        # 载入数据
        df_new = df.copy()
        features_new = features.copy()[0:-4]
        y = df_new[target]
        X = df_new[features]

        # 定义 K 折交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        # 创建随机森林模型
        rf_model = RandomForestRegressor(n_estimators=500, random_state=42,n_jobs=10)
        # 使用交叉验证评估 R²
        r2_scores = []
        mse_scores = []
        mae_scores = []
        # 设立权重
        spatial_data_all = pd.DataFrame()
        data_all = {
            'shap':np.empty((0,len(features_new))),
            'y_val':pd.DataFrame(),
            'x_val':pd.DataFrame()}     #汇总所有的shap值和对应的X_test
        y_pred_all = np.array([])
        y_val_all = np.array([])
        i = 1
        for train_index, val_index in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_index][features_new], X.iloc[val_index][features_new]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
            weights_k = X.iloc[train_index]['weights']
            rf_model.fit(X_train_fold, y_train_fold,sample_weight=weights_k)
            # 在验证集上进行预测
            y_pred = rf_model.predict(X_val_fold)
            y_pred_all = np.concatenate([y_pred_all,y_pred])
            y_val_all = np.concatenate([y_val_all,y_val_fold])
            # 计算验证集的 R² 分数
            r2 = r2_score(y_val_fold, y_pred)
            r2_scores.append(r2)
            mse = mean_squared_error(y_val_fold, y_pred)
            mse_scores.append(mse)
            mae = mean_absolute_error(y_val_fold, y_pred)
            mae_scores.append(mae)
            # 计算验证集的 SHAP 值
            explainer = shap.TreeExplainer(rf_model)  # 创建SHAP解释器
            shap_values = explainer.shap_values(pd.DataFrame(X_val_fold, columns=features_new))  # 计算验证集的SHAP值
            data_shap = X.iloc[val_index][['row','col','country']]
            data_shap[f'tp_spring_relative_shap_ratio'] = shap_values[:,0] / (np.sum(np.abs(shap_values),axis=1))
            # data_shap[f'tp_summer_relative_shap_ratio'] = shap_values[:, 1] / (np.sum(np.abs(shap_values), axis=1))
            # data_shap[f'tp_autumn_relative_shap_ratio'] = shap_values[:, 2] / (np.sum(np.abs(shap_values), axis=1))
            # data_shap[f'tp_winter_relative_shap_ratio'] = shap_values[:, 3] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'vpd_annual_relative_shap_ratio'] = shap_values[:, 1] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'tm_annual_relative_shap_ratio'] = shap_values[:, 2] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'spei_annual_relative_shap_ratio'] = shap_values[:, 3] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'tp_spring_shap_value'] = shap_values[:,0]
            # data_shap[f'tp_summer_shap_value'] = shap_values[:, 1]
            # data_shap[f'tp_autumn_shap_value'] = shap_values[:, 2]
            # data_shap[f'tp_winter_shap_value'] = shap_values[:, 3]
            data_shap[f'vpd_annual_shap_value'] = shap_values[:, 1]
            data_shap[f'tm_annual_shap_value'] = shap_values[:, 2]
            data_shap[f'spei_annual_shap_value'] = shap_values[:, 3]
            # 保存每次验证集的 SHAP 值
            spatial_data_all = pd.concat((spatial_data_all,data_shap))
            data_all['shap'] = np.vstack((data_all['shap'], shap_values))
            data_all['y_val'] = pd.concat((data_all['y_val'],y_val_fold))
            data_all['x_val'] = pd.concat((data_all['x_val'],X_val_fold))
            print(f'计算完{i}折')
            i+=1
        r2 = np.mean(r2_scores)
        mse = np.mean(mse_scores)
        mae = np.mean(mae_scores)
        # 绘制散点图,保存真值和预测值为csv
        scatter_density_plot(y_val_all, y_pred_all, mse,mae, r2, output_path,year)
        pd.DataFrame({'True_Values': y_val_all,'Predicted_Values': y_pred_all}).to_csv(os.path.join(output_path,f'y_val_y_pred_{year}.csv'))
        # 保存shap信息为csv
        data_all['x_val'].add_suffix('_xValue')
        data_all['y_val'].add_suffix('_yValue')
        spatial_data_all = pd.concat([spatial_data_all,data_all['x_val']],axis=1)
        spatial_data_all = pd.concat([spatial_data_all, data_all['y_val']], axis=1)
        spatial_data_all.to_csv(os.path.join(output_path,f'SHAP_summary_{year}.csv'))
        # 可视化shapsummary图
        rename_mapping = {
        f'tp_spring_group{year}': 'TP_Spring',
        f'tp_summer_group{year}': 'TP_Summer',
        f'tp_autumn_group{year}': 'TP_Autumn',
        f'tp_winter_group{year}': 'TP_Winter',
        f'vpd_annual_group{year}': 'VPD',
        f'tm_annual_group{year}': 'TM',
        f'spei_03_annual_spei_group{year}': 'SPEI'
    }
        data_all['x_val'].rename(columns=rename_mapping, inplace=True)
        shap.summary_plot(data_all['shap'], data_all['x_val'][data_all['x_val'].keys().tolist()], show=False)
        plt.savefig(os.path.join(output_path,f'Shap_summary_{year}.jpg'), bbox_inches='tight', dpi=300)
        plt.close()  # 关闭图形以释放内存
    def random_forest_cokfold(self,target,features,df,year):
        def scatter_density_plot(y_test, y_pred, mse, mae, r2, output_path, year):
            plt.figure(figsize=(8, 6), constrained_layout=True)  # 使用 constrained_layout
            # 创建 2D 直方图，颜色表示像素密度
            hb = plt.hexbin(y_test, y_pred, gridsize=50, cmap='OrRd', norm=LogNorm())
            # 添加颜色条
            cb = plt.colorbar(hb)
            cb.set_label('[pixels]')
            # 添加对角线 y=x 的参考线
            plt.plot([0, np.max([np.max(y_test), np.max(y_pred)])], [0, np.max([np.max(y_test), np.max(y_pred)])],
                     'r--', lw=1.5)
            # 添加统计信息，调整位置
            # 添加统计信息
            plt.text(0.05, 0.95, f'N={len(y_test)}', transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top')
            plt.text(0.05, 0.90, f'R²={r2:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
            plt.text(0.05, 0.85, f'MSE={mse:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
            plt.text(0.05, 0.80, f'MAE={mae:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

            # 设置坐标轴和标题
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.grid(True)
            # 保存图片
            plt.savefig(os.path.join(output_path, f'Density_Scatter_Plot_{year}.png'), dpi=300)
            plt.close()  # 关闭图形以释放内存
        # 定义保存图片路径
        if year == 'Temporal Series':
            output_path = 'images'
        else:
            output_path = 'images window'
        os.makedirs(output_path,exist_ok=True)
        # 载入数据
        df_new = df.copy()
        features_new = features.copy()[0:-5]
        y = df_new[target]
        X = df_new[features]

        # 定义 K 折交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        # 创建随机森林模型
        rf_model = RandomForestRegressor(n_estimators=500, random_state=42,n_jobs=10)
        # 使用交叉验证评估 R²
        r2_scores = []
        mse_scores = []
        mae_scores = []
        # 设立权重
        spatial_data_all = pd.DataFrame()
        data_all = {
            'shap':np.empty((0,len(features_new))),
            'y_val':pd.DataFrame(),
            'x_val':pd.DataFrame()}     #汇总所有的shap值和对应的X_test
        y_pred_all = np.array([])
        y_val_all = np.array([])
        i = 1
        for train_index, val_index in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_index][features_new], X.iloc[val_index][features_new]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
            weights_k = X.iloc[train_index]['weights']
            rf_model.fit(X_train_fold, y_train_fold,sample_weight=weights_k)
            # 在验证集上进行预测
            y_pred = rf_model.predict(X_val_fold)
            y_pred_all = np.concatenate([y_pred_all,y_pred])
            y_val_all = np.concatenate([y_val_all,y_val_fold])
            # 计算验证集的 R² 分数
            r2 = r2_score(y_val_fold, y_pred)
            r2_scores.append(r2)
            mse = mean_squared_error(y_val_fold, y_pred)
            mse_scores.append(mse)
            mae = mean_absolute_error(y_val_fold, y_pred)
            mae_scores.append(mae)
            # 计算验证集的 SHAP 值
            explainer = shap.TreeExplainer(rf_model)  # 创建SHAP解释器
            shap_values = explainer.shap_values(pd.DataFrame(X_val_fold, columns=features_new))  # 计算验证集的SHAP值
            data_shap = X.iloc[val_index][['row','col','country','year']]
            data_shap[f'tp_spring_relative_shap_ratio'] = shap_values[:,0] / (np.sum(np.abs(shap_values),axis=1))
            data_shap[f'tp_summer_relative_shap_ratio'] = shap_values[:, 1] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'tp_autumn_relative_shap_ratio'] = shap_values[:, 2] / (np.sum(np.abs(shap_values), axis=1))
            # data_shap[f'tp_winter_relative_shap_ratio'] = shap_values[:, 3] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'vpd_annual_relative_shap_ratio'] = shap_values[:, 3] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'tm_annual_relative_shap_ratio'] = shap_values[:, 4] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'spei_annual_relative_shap_ratio'] = shap_values[:, 5] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'tp_spring_shap_value'] = shap_values[:,0]
            data_shap[f'tp_summer_shap_value'] = shap_values[:, 1]
            data_shap[f'tp_autumn_shap_value'] = shap_values[:, 2]
            # data_shap[f'tp_winter_shap_value'] = shap_values[:, 3]
            data_shap[f'vpd_annual_shap_value'] = shap_values[:, 3]
            data_shap[f'tm_annual_shap_value'] = shap_values[:, 4]
            data_shap[f'spei_annual_shap_value'] = shap_values[:, 5]
            # 保存每次验证集的 SHAP 值
            spatial_data_all = pd.concat((spatial_data_all,data_shap))
            data_all['shap'] = np.vstack((data_all['shap'], shap_values))
            data_all['y_val'] = pd.concat((data_all['y_val'],y_val_fold))
            data_all['x_val'] = pd.concat((data_all['x_val'],X_val_fold))
            print(f'计算完{i}折')
            i+=1
        r2 = np.mean(r2_scores)
        mse = np.mean(mse_scores)
        mae = np.mean(mae_scores)
        # 绘制散点图,保存真值和预测值为csv
        scatter_density_plot(y_val_all, y_pred_all, mse,mae, r2, output_path,year)
        pd.DataFrame({'True_Values': y_val_all,'Predicted_Values': y_pred_all}).to_csv(os.path.join(output_path,f'y_val_y_pred_{year}.csv'))
        # 保存shap信息为csv
        data_all['x_val'].add_suffix('_xValue')
        data_all['y_val'].add_suffix('_yValue')
        spatial_data_all = pd.concat([spatial_data_all,data_all['x_val']],axis=1)
        spatial_data_all = pd.concat([spatial_data_all, data_all['y_val']], axis=1)
        spatial_data_all.to_csv(os.path.join(output_path,f'SHAP_summary_{year}.csv'))
        # 可视化shapsummary图
        rename_mapping = {
        f'tp_spring_group{year}': 'TP_Spring',
        f'tp_summer_group{year}': 'TP_Summer',
        f'tp_autumn_group{year}': 'TP_Autumn',
        f'tp_winter_group{year}': 'TP_Winter',
        f'vpd_annual_group{year}': 'VPD',
        f'tm_annual_group{year}': 'TM',
        f'spei_03_annual_spei_group{year}': 'SPEI'
    }
        data_all['x_val'].rename(columns=rename_mapping, inplace=True)
        shap.summary_plot(data_all['shap'], data_all['x_val'][data_all['x_val'].keys().tolist()], show=False)
        plt.savefig(os.path.join(output_path,f'Shap_summary_{year}.jpg'), bbox_inches='tight', dpi=300)
        plt.close()  # 关闭图形以释放内存

    def random_forest_co(self,target,features,df,year):
        def scatter_density_plot(y_test, y_pred, mse, mae, r2, output_path, year):
            plt.figure(figsize=(8, 6), constrained_layout=True)  # 使用 constrained_layout
            # 创建 2D 直方图，颜色表示像素密度
            hb = plt.hexbin(y_test, y_pred, gridsize=50, cmap='OrRd', norm=LogNorm())
            # 添加颜色条
            cb = plt.colorbar(hb)
            cb.set_label('[pixels]')
            # 添加对角线 y=x 的参考线
            plt.plot([0, np.max([np.max(y_test), np.max(y_pred)])], [0, np.max([np.max(y_test), np.max(y_pred)])],
                     'r--', lw=1.5)
            # 添加统计信息，调整位置
            # 添加统计信息
            plt.text(0.05, 0.95, f'N={len(y_test)}', transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top')
            plt.text(0.05, 0.90, f'R²={r2:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
            plt.text(0.05, 0.85, f'MSE={mse:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
            plt.text(0.05, 0.80, f'MAE={mae:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

            # 设置坐标轴和标题
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.grid(True)
            # 保存图片
            plt.savefig(os.path.join(output_path, f'Density_Scatter_Plot_{year}.png'), dpi=300)
            plt.close()  # 关闭图形以释放内存
        # 定义保存图片路径
        if year == 'Temporal Series':
            output_path = 'images'
        else:
            output_path = 'images window'
        os.makedirs(output_path,exist_ok=True)
        # 载入数据
        df_new = df.copy()
        features_new = features.copy()[0:-5]
        y = df_new[target]
        X = df_new[features]

        # 定义 K 折交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        # 创建随机森林模型
        rf_model = RandomForestRegressor(n_estimators=500, random_state=42,n_jobs=10)
        # 使用交叉验证评估 R²
        r2_scores = []
        mse_scores = []
        mae_scores = []
        # 设立权重
        spatial_data_all = pd.DataFrame()
        data_all = {
            'shap':np.empty((0,len(features_new))),
            'y_val':pd.DataFrame(),
            'x_val':pd.DataFrame()}     #汇总所有的shap值和对应的X_test
        y_pred_all = np.array([])
        y_val_all = np.array([])
        i = 1
        for train_index, val_index in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_index][features_new], X.iloc[val_index][features_new]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
            weights_k = X.iloc[train_index]['weights']
            rf_model.fit(X_train_fold, y_train_fold,sample_weight=weights_k)
            # 在验证集上进行预测
            y_pred = rf_model.predict(X_val_fold)
            y_pred_all = np.concatenate([y_pred_all,y_pred])
            y_val_all = np.concatenate([y_val_all,y_val_fold])
            # 计算验证集的 R² 分数
            r2 = r2_score(y_val_fold, y_pred)
            r2_scores.append(r2)
            mse = mean_squared_error(y_val_fold, y_pred)
            mse_scores.append(mse)
            mae = mean_absolute_error(y_val_fold, y_pred)
            mae_scores.append(mae)
            # 计算验证集的 SHAP 值
            explainer = shap.TreeExplainer(rf_model)  # 创建SHAP解释器
            shap_values = explainer.shap_values(pd.DataFrame(X_val_fold, columns=features_new))  # 计算验证集的SHAP值
            data_shap = X.iloc[val_index][['row','col','country','year']]
            data_shap[f'tp_spring_relative_shap_ratio'] = shap_values[:,0] / (np.sum(np.abs(shap_values),axis=1))
            data_shap[f'tp_summer_relative_shap_ratio'] = shap_values[:, 1] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'tp_autumn_relative_shap_ratio'] = shap_values[:, 2] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'tp_winter_relative_shap_ratio'] = shap_values[:, 3] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'vpd_annual_relative_shap_ratio'] = shap_values[:, 4] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'tm_annual_relative_shap_ratio'] = shap_values[:, 5] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'spei_annual_relative_shap_ratio'] = shap_values[:,6] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'tp_spring_shap_value'] = shap_values[:,0]
            data_shap[f'tp_summer_shap_value'] = shap_values[:, 1]
            data_shap[f'tp_autumn_shap_value'] = shap_values[:, 2]
            data_shap[f'tp_winter_shap_value'] = shap_values[:, 3]
            data_shap[f'vpd_annual_shap_value'] = shap_values[:, 4]
            data_shap[f'tm_annual_shap_value'] = shap_values[:, 5]
            data_shap[f'spei_annual_shap_value'] = shap_values[:, 6]
            # 保存每次验证集的 SHAP 值
            spatial_data_all = pd.concat((spatial_data_all,data_shap))
            data_all['shap'] = np.vstack((data_all['shap'], shap_values))
            data_all['y_val'] = pd.concat((data_all['y_val'],y_val_fold))
            data_all['x_val'] = pd.concat((data_all['x_val'],X_val_fold))
            print(f'计算完{i}折')
            i+=1
        r2 = np.mean(r2_scores)
        mse = np.mean(mse_scores)
        mae = np.mean(mae_scores)
        # 绘制散点图,保存真值和预测值为csv
        scatter_density_plot(y_val_all, y_pred_all, mse,mae, r2, output_path,year)
        pd.DataFrame({'True_Values': y_val_all,'Predicted_Values': y_pred_all}).to_csv(os.path.join(output_path,f'y_val_y_pred_{year}.csv'))
        # 保存shap信息为csv
        data_all['x_val'].add_suffix('_xValue')
        data_all['y_val'].add_suffix('_yValue')
        spatial_data_all = pd.concat([spatial_data_all,data_all['x_val']],axis=1)
        spatial_data_all = pd.concat([spatial_data_all, data_all['y_val']], axis=1)
        spatial_data_all.to_csv(os.path.join(output_path,f'SHAP_summary_{year}.csv'))
        # 可视化shapsummary图
        rename_mapping = {
        f'tp_spring_group{year}': 'TP_Spring',
        f'tp_summer_group{year}': 'TP_Summer',
        f'tp_autumn_group{year}': 'TP_Autumn',
        f'tp_winter_group{year}': 'TP_Winter',
        f'vpd_annual_group{year}': 'VPD',
        f'tm_annual_group{year}': 'TM',
        f'spei_03_annual_spei_group{year}': 'SPEI'
    }
        data_all['x_val'].rename(columns=rename_mapping, inplace=True)
        shap.summary_plot(data_all['shap'], data_all['x_val'][data_all['x_val'].keys().tolist()], show=False)
        plt.savefig(os.path.join(output_path,f'Shap_summary_{year}.jpg'), bbox_inches='tight', dpi=300)
        plt.close()  # 关闭图形以释放内存

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

    def rf_modelling(self,data_path,start_year,end_year,indices_name):
        # 基于滑动窗口的窗口分析建模
        df = pd.read_csv(data_path).dropna()
        filtered_df = df.copy()
        window_length = 3
        # 生成年份列表
        years = np.arange(start_year, end_year + 1)
        # 使用滑动窗口分组
        groups = [years[i:i + window_length] for i in range(len(years) - window_length + 1)]
        variables = [col.replace('_2005', '') for col in filtered_df.columns if '2005' in col]
        filtered_df_new = filtered_df[['row','col','weights','country']]
        if end_year > 2016:variables.remove('sif')  # 如果 end_year > 2017，则移除 'sif' 变量
        new_columns_data = []
        new_columns_name = []
        for group_id,group in enumerate(groups):
            for variable in variables:
                new_column_name = f'{variable}_group{group_id}'
                mean_values = filtered_df[[f'{variable}_{single_year}' for single_year in group]].mean(axis=1)
                new_columns_data.append(mean_values)
                new_column_name = f'{variable}_group{group[0]}'
                new_columns_name.append(new_column_name)
                # filtered_df_new[f'{variable}_group{group_id}'] = filtered_df[[f'{variable}_{single_year}' for single_year in group]].mean(axis=1)
        new_data = pd.DataFrame(new_columns_data).T  # `.T` 转置使列正确
        # 设置新列名
        new_data.columns = new_columns_name
        # 将新列添加到原始 DataFrame
        filtered_df_new = pd.concat([filtered_df_new, new_data], axis=1)
        for group in (groups):
            group_id = group[0]
            target = f'{indices_name}_group{group_id}'
            # 干旱指数建模
            features = [f'tp_spring_group{group_id}',
                        # f'tp_summer_group{group_id}',
                        # f'tp_autumn_group{group_id}',
                        f'vpd_annual_group{group_id}',
                        f'tm_annual_group{group_id}',
                        f'spei_03_annual_spei_group{group_id}',
                        'weights',
                        'row','col','country']
            self.random_forest_group(target,features,filtered_df_new,group_id)

    def rf_modelling_pooling(self, data_path, start_year, end_year, indices_name):
        # 基于池化的窗口分析
        df = pd.read_csv(data_path).dropna()
        filtered_df = df.copy()
        window_length = 3
        years = np.arange(start_year, end_year)
        num_groups = (len(years) // window_length) - 1
        groups = [years[i:i + window_length] for i in range(0, num_groups * window_length, window_length)]
        groups.append(years[num_groups * window_length:])  # 剩余年份直接作为最后一组
        variables = [col.replace('_2005', '') for col in filtered_df.columns if '2005' in col]
        filtered_df_new = filtered_df[['row', 'col', 'weights', 'country']]
        if end_year > 2017: variables.remove('sif')  # 如果 end_year > 2017，则移除 'sif' 变量
        new_columns_data = []
        new_columns_name = []
        for group_id, group in enumerate(groups):
            for variable in variables:
                new_column_name = f'{variable}_group{group_id}'
                mean_values = filtered_df[[f'{variable}_{single_year}' for single_year in group]].mean(axis=1)
                new_columns_data.append(mean_values)
                new_column_name = f'{variable}_group{group_id}'
                new_columns_name.append(new_column_name)
                # filtered_df_new[f'{variable}_group{group_id}'] = filtered_df[[f'{variable}_{single_year}' for single_year in group]].mean(axis=1)
        new_data = pd.DataFrame(new_columns_data).T  # `.T` 转置使列正确
        # 设置新列名
        new_data.columns = new_columns_name
        # 将新列添加到原始 DataFrame
        filtered_df_new = pd.concat([filtered_df_new, new_data], axis=1)
        for group_id, group in enumerate(groups):
            target = f'{indices_name}_group{group_id}'
            # 干旱指数建模
            features = [f'tp_spring_group{group_id}',
                        f'tp_summer_group{group_id}',
                        f'tp_autumn_group{group_id}',
                        # f'tp_winter_group{group_id}',
                        f'vpd_annual_group{group_id}',
                        f'tm_annual_group{group_id}',
                        f'spei_03_annual_spei_group{group_id}',
                        'weights',
                        'row', 'col', 'country']
            self.random_forest(target, features, filtered_df_new, group_id)

    def rf_modelling_co(self,data_path,start_year,end_year,indices_name):
        df = pd.read_csv(data_path).dropna()
        filtered_df = df.copy()
        filtered_df_stack_time_series = self.stack_data(filtered_df, start_year, end_year,indices_name)
        filtered_df_stack_time_series_sample = filtered_df_stack_time_series.sample(n=10000, random_state=42)
        target = f'{indices_name}'
        # 干旱指数建模
        features = [f'tp_spring',
                    f'tp_summer',
                    f'tp_autumn',
                    # f'tp_winter',
                    f'vpd',
                    f'tm',
                    f'spei',
                    'weights',
                    'row','col','country','year']
        self.random_forest_co(target,features,filtered_df_stack_time_series_sample,'Temporal Series')

    def rf_modelling_co_nokfold(self,data_path,start_year,end_year,indices_name):
        df = pd.read_csv(data_path).dropna()
        filtered_df = df.copy()
        filtered_df_stack_time_series = self.stack_data(filtered_df, start_year, end_year,indices_name)
        filtered_df_stack_time_series_sample = filtered_df_stack_time_series.sample(n=10000, random_state=42)
        target = f'{indices_name}'
        # 干旱指数建模
        features = [f'tp_spring',
                    f'tp_summer',
                    f'tp_autumn',
                    f'vpd',
                    f'tm',
                    f'spei',
                    'weights',
                    'row','col','country','year']

        X = filtered_df_stack_time_series_sample[features]
        y = filtered_df_stack_time_series_sample[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        weights = X_train['weights']
        rf_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=10)
        rf_model.fit(X_train[features[:-5]], y_train, sample_weight=weights)
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

    def stack_data(self,input_data,start_year,end_year,indices_name):
        new_data = []
        for index, row in input_data.iterrows():
            for year in range(start_year,end_year):
                new_row = {}
                new_row["gridid"] = f"{row['row']}_{row['col']}_{row['country']}"
                new_row["year"] = year
                new_row['row'] = row['row']
                new_row['col'] = row['col']
                new_row['weights'] = row['weights']
                new_row["country"] = row['country']
                new_row[f"{indices_name}"] = row[f"{indices_name}_{year}"]
                new_row["tp_spring"] = row[f'tp_spring_{year}']
                new_row["tp_summer"] = row[f'tp_summer_{year}']
                new_row["tp_autumn"] = row[f'tp_autumn_{year}']
                new_row[f'tp_winter'] = row[f'tp_winter_{year}']
                new_row[f'vpd'] = row[f'vpd_annual_{year}']
                new_row["tm"] = row[f'tm_annual_{year}']
                new_row["spei"] = row[f'spei_03_annual_spei_{year}']
                new_data.append(new_row)
        new_df = pd.DataFrame(new_data)
        return new_df
    def random_forest_summary(self,data_path):
        data = pd.read_csv(data_path).dropna()
        shap_data = data[['tp_spring_shap_value','tp_summer_shap_value','tp_autumn_shap_value','vpd_annual_shap_value','tm_annual_shap_value','spei_annual_shap_value']].to_numpy()
        shap.summary_plot(shap_data, data[['tp_spring','tp_summer','tp_autumn','vpd','tm','spei']], show=False)
        plt.savefig('test.jpg', bbox_inches='tight', dpi=300)


    def temporal_window_analysis(self,paths,season,start_year,end_year,dataset):
        """
        分析时间窗口 SHAP 比例变化趋势，并拟合散点和绘制置信区间。
        参数：
            paths (list): 包含滑动窗口数据文件路径的列表。
            variable (str): 需要分析的变量名。
            start_year (int): 数据起始年份。
            end_year (int): 数据结束年份。
            dataset (str): 数据集名称 ("EVI" 或 "SIF")。
        """
        total_years = end_year - start_year + 1
        window_size = total_years - len(paths) + 1  # 推导窗口大小
        # 生成年份窗口
        years = [f"{start_year + i}-{start_year + i + window_size - 1}" for i in range(len(paths))]
        # 读取数据并计算变量 SHAP 比例
        variable_shap_ratios = []
        for path in paths:
            data = pd.read_csv(path)
            data[f'tp_{season}_importance'] = data[f'tp_{season}_shap_value'].abs().div(data[[f'tp_{season}_shap_value',f'vpd_{season}_shap_value',f'tm_{season}_shap_value']].abs().sum(axis=1))
            variable_shap_ratios.append(data[f'tp_{season}_importance'].mean())

        # 数值化年份窗口范围
        x_numeric = list(range(len(years)))
        slope, intercept, r_value, p_value, std_err = linregress(x_numeric, variable_shap_ratios)

        # 计算 95% 置信区间
        ci_lower = slope - 1.96 * std_err
        ci_upper = slope + 1.96 * std_err
        # 设置全局样式
        plt.rcParams['font.family'] = 'Arial'
        plt.figure(figsize=(10, 6))
        # 散点颜色映射（基于 SHAP 值）
        cmap = plt.get_cmap('RdBu').reversed()  # 使用 RdBu 渐变色
        norm = plt.Normalize(vmin=min(variable_shap_ratios), vmax=max(variable_shap_ratios))
        scatter_colors = [cmap(norm(val)) for val in variable_shap_ratios]
        # 绘制散点图
        plt.scatter(
            x_numeric,
            variable_shap_ratios,
            color=scatter_colors,
            edgecolor='black',
            s=80
        )
        # 绘制拟合线和置信区间
        sns.regplot(
            x=x_numeric,
            y=variable_shap_ratios,
            scatter=False,
            line_kws={"color": "dimgray", "linewidth": 2.5},  # 深灰色线条
            ci=95,
            label=f"OLS Slope: {slope:.4f} ({ci_lower:.4f}, {ci_upper:.4f})",
            truncate=False
        )
        # variable_y_label_map = {'tp_spring_relative_shap_ratio_new':'TP_Spring','tp_summer_relative_shap_ratio_new':'TP_Summer','tp_autumn_relative_shap_ratio_new':'TP_Autumn'}
        # 绘制背景和轴
        plt.gca().set_facecolor('#f5f5f5')  # 坐标轴背景色设置为白色
        plt.xticks(ticks=x_numeric, labels=years, rotation=45, fontsize=15)
        plt.yticks(fontsize=10)
        plt.ylabel(f'TP {season} effect on forest {dataset}', fontsize=20)
        # 添加网格
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        # 添加图例和颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # 为颜色条创建映射
        cbar = plt.colorbar(sm, aspect=30, pad=0.02)
        # cbar.set_label('SHAP relatively ratio', fontsize=20)
        plt.legend(fontsize=20, loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.split(path)[0],f'TP {season}_temporal_fitting.jpg'),dpi = 600)
        plt.savefig(os.path.join(os.path.split(path)[0], f'TP {season}_temporal_fitting.pdf'), dpi=600)


    def temporal_window_analysis_onlyseason(self,paths,season,start_year,end_year,dataset):
        """
        分析时间窗口 SHAP 比例变化趋势，并拟合散点和绘制置信区间。
        参数：
            paths (list): 包含滑动窗口数据文件路径的列表。
            variable (str): 需要分析的变量名。
            start_year (int): 数据起始年份。
            end_year (int): 数据结束年份。
            dataset (str): 数据集名称 ("EVI" 或 "SIF")。
        """
        total_years = end_year - start_year + 1
        window_size = total_years - len(paths) + 1  # 推导窗口大小
        # 生成年份窗口
        years = [f"{start_year + i}-{start_year + i + window_size - 1}" for i in range(len(paths))]
        # 读取数据并计算变量 SHAP 比例
        variable_shap_ratios = []
        for path in paths:
            data = pd.read_csv(path)
            data[f'tp_{season}_importance'] = data[f'tp_{season}_shap_value'].abs().div(data[[f'tp_spring_shap_value',f'tp_summer_shap_value',f'tp_autumn_shap_value',f'tp_winter_shap_value']].abs().sum(axis=1))
            variable_shap_ratios.append(data[f'tp_{season}_importance'].mean())

        # 数值化年份窗口范围
        x_numeric = list(range(len(years)))
        slope, intercept, r_value, p_value, std_err = linregress(x_numeric, variable_shap_ratios)

        # 计算 95% 置信区间
        ci_lower = slope - 1.96 * std_err
        ci_upper = slope + 1.96 * std_err
        # 设置全局样式
        plt.rcParams['font.family'] = 'Arial'
        plt.figure(figsize=(10, 6))
        # 散点颜色映射（基于 SHAP 值）
        cmap = plt.get_cmap('RdBu').reversed()  # 使用 RdBu 渐变色
        norm = plt.Normalize(vmin=min(variable_shap_ratios), vmax=max(variable_shap_ratios))
        scatter_colors = [cmap(norm(val)) for val in variable_shap_ratios]
        # 绘制散点图
        plt.scatter(
            x_numeric,
            variable_shap_ratios,
            color=scatter_colors,
            edgecolor='black',
            s=80
        )
        # 绘制拟合线和置信区间
        sns.regplot(
            x=x_numeric,
            y=variable_shap_ratios,
            scatter=False,
            line_kws={"color": "dimgray", "linewidth": 2.5},  # 深灰色线条
            ci=95,
            label=f"OLS Slope: {slope:.4f} ({ci_lower:.4f}, {ci_upper:.4f})",
            truncate=False
        )
        # variable_y_label_map = {'tp_spring_relative_shap_ratio_new':'TP_Spring','tp_summer_relative_shap_ratio_new':'TP_Summer','tp_autumn_relative_shap_ratio_new':'TP_Autumn'}
        # 绘制背景和轴
        plt.gca().set_facecolor('#f5f5f5')  # 坐标轴背景色设置为白色
        plt.xticks(ticks=x_numeric, labels=years, rotation=45, fontsize=15)
        plt.yticks(fontsize=10)
        plt.ylabel(f'TP {season} effect on forest {dataset}', fontsize=20)
        # 添加网格
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        # 添加图例和颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # 为颜色条创建映射
        cbar = plt.colorbar(sm, aspect=30, pad=0.02)
        # cbar.set_label('SHAP relatively ratio', fontsize=20)
        plt.legend(fontsize=20, loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.split(path)[0],f'TP {season}_temporal_fitting.jpg'),dpi = 600)
        plt.savefig(os.path.join(os.path.split(path)[0], f'TP {season}_temporal_fitting.pdf'), dpi=600)

    def temporal_window_analysis_with_SPEI_anomaly(self,paths,climate_name,season,start_year,end_year,dataset,summary_path):
        summary =  pd.read_csv(summary_path).dropna()
        if season =='winter':
            chill_cols = [col for col in summary.columns if f'tm_chilling_' in col]
            for chill_col in chill_cols:
                summary = summary.rename(columns={chill_col: chill_col.replace('chilling','winter')})
        climate_baseline = summary[[col for col in summary.columns if f'{climate_name}_{season}_' in col]].mean(axis=1)
        summary[f'{climate_name}_{season}_baseline'] = climate_baseline
        total_years = end_year - start_year + 1
        window_size = total_years - len(paths) + 1  # 推导窗口大小
        # 生成年份窗口
        years = [f"{start_year + i}-{start_year + i + window_size - 1}" for i in range(len(paths))]
        years_single = [int(item.split('-')[0]) for item in years]
        # 读取数据并计算变量 SHAP 比例
        variable_shap_ratios = []
        climate_anomaly = []
        group_years_collection = [np.arange(start_year, end_year + 1)[i:i + window_size] for i in range(len(np.arange(start_year, end_year + 1)) - window_size + 1)]
        for i,path in enumerate(paths):
            group_year = os.path.split(path)[-1].split('_')[-1][:-4]
            group_years = group_years_collection[i]
            climate_group_years = [f'{climate_name}_{season}_{climate_year}' for climate_year in group_years]
            data = pd.read_csv(path)
            data[f'tp_{season}_importance'] = data[f'tp_{season}_shap_value'].abs().div(data[[f'tp_spring_shap_value',
                                                                                              f'tp_summer_shap_value',
                                                                                              f'tp_autumn_shap_value',
                                                                                              f'tp_winter_shap_value']].abs().sum(axis=1))
            data_ord = data.assign(_ord=range(len(data)))
            data_interact = summary.merge(data,on=['row', 'col', 'country', 'biogeo', 'country_chunk'],how='inner')
            variable_shap_ratios.append(data_interact[f'tp_{season}_importance'].mean())
            reisduals = data_interact[climate_group_years].mean(axis=1)-data_interact[f'{climate_name}_{season}_baseline']
            climate_anomaly.append((reisduals.mean())/data_interact[f'{climate_name}_{season}_baseline'].std())   #当前group的vpd减去长期均值，然然后除以长期均值序列的标准差

        # 设置全局样式
        plt.rcParams['font.family'] = 'Arial'
        plt.figure(figsize=(10, 6))
        # 散点颜色映射（基于 SHAP 值）
        cmap = plt.get_cmap('PuBu').reversed()  # 使用 RdBu 渐变色
        norm = plt.Normalize(vmin=min(years_single), vmax=max(years_single))
        scatter_colors = [cmap(norm(val)) for val in years_single]
        # 绘制散点图
        plt.scatter(
            climate_anomaly,
            variable_shap_ratios,
            color=scatter_colors,
            edgecolor='black',
            s=80
        )
        for i, year in enumerate(years_single):
            plt.annotate(str(year), (climate_anomaly[i], variable_shap_ratios[i]),
                         textcoords="offset points", xytext=(5,5), ha='center', fontsize=12)

        # 绘制拟合线和置信区间
        sns.regplot(
            x=climate_anomaly,
            y=variable_shap_ratios,
            scatter=False,
            line_kws={"color": "dimgray", "linewidth": 2.5},  # 深灰色线条
            ci=95,
            label="OLS with 95% CI",
            truncate=False
        )

        plt.gca().set_facecolor('#f5f5f5')  # 坐标轴背景色设置为白色
        x_min, x_max = min(climate_anomaly), max(climate_anomaly)
        # 生成均匀间隔的刻度
        num_ticks = 5  # 你可以调整这个数值
        x_ticks = np.linspace(x_min, x_max, num_ticks)
        # 设置X轴刻度
        plt.xticks(ticks=x_ticks, labels=[f"{tick:.2f}" for tick in x_ticks], fontsize=15)
        plt.xlabel(f'{climate_name.capitalize()} anomaly (Z-Score)',fontsize=20)
        plt.yticks(fontsize=10)
        plt.ylabel(f'TP_{season} effect on forest {dataset}', fontsize=20)
        # 添加网格
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        # 添加图例和颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # 为颜色条创建映射
        cbar = plt.colorbar(sm, aspect=30, pad=0.02)
        # cbar.set_label('SHAP relatively ratio', fontsize=20)
        # plt.legend(loc="upper left",fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.split(path)[0],f'tp_{season}_{climate_name}_analysis.jpg'),dpi = 600)
        plt.savefig(os.path.join(os.path.split(path)[0], f'tp_{season}_{climate_name}_analysis.pdf'), dpi=600)
    def temporal_window_analysis_TheilSen(self,paths,variable,start_year,end_year,dataset):
        """
    使用 Theil-Sen 回归分析时间窗口 SHAP 比例变化趋势，并拟合散点和绘制置信区间。
    参数：
        paths (list): 包含滑动窗口数据文件路径的列表。
        variable (str): 需要分析的变量名。
        start_year (int): 数据起始年份。
        end_year (int): 数据结束年份。
        dataset (str): 数据集名称 ("EVI" 或 "SIF")。
        """
        total_years = end_year - start_year + 1
        window_size = total_years - len(paths) + 1  # 推导窗口大小
        # 生成年份窗口
        years = [f"{start_year + i}-{start_year + i + window_size - 1}" for i in range(len(paths))]
        # 读取数据并计算变量 SHAP 比例
        variable_shap_ratios = []
        for path in paths:
            data = pd.read_csv(path)
            variable_shap_ratios.append(data[variable].mean())

        # 数值化年份窗口范围
        x_numeric = np.arange(len(years)).reshape(-1, 1)  # 必须为二维数组
        y_values = np.array(variable_shap_ratios)

        # 执行 Theil-Sen 回归
        model = TheilSenRegressor(random_state=42)
        model.fit(x_numeric, y_values)
        y_pred = model.predict(x_numeric)

        # Bootstrap 计算置信区间
        n_bootstrap = 1000
        bootstrap_preds = []
        for _ in range(n_bootstrap):
            x_resampled, y_resampled = resample(x_numeric, y_values)
            model_boot = TheilSenRegressor(random_state=42)
            model_boot.fit(x_resampled, y_resampled)
            bootstrap_preds.append(model_boot.predict(x_numeric))
        bootstrap_preds = np.array(bootstrap_preds)

        lower_ci = np.percentile(bootstrap_preds, 2.5, axis=0)
        upper_ci = np.percentile(bootstrap_preds, 97.5, axis=0)

        # 设置全局样式
        plt.rcParams['font.family'] = 'Arial'
        plt.figure(figsize=(10, 6))
        # 散点颜色映射（基于 SHAP 值）
        cmap = plt.get_cmap('RdBu').reversed()  # 使用 RdBu 渐变色
        norm = plt.Normalize(vmin=min(variable_shap_ratios), vmax=max(variable_shap_ratios))
        scatter_colors = [cmap(norm(val)) for val in variable_shap_ratios]

        # 绘制散点图
        plt.scatter(
            x_numeric,
            y_values,
            color=scatter_colors,
            edgecolor='black',
            s=80
        )

        # 绘制 Theil-Sen 拟合线
        plt.plot(x_numeric, y_pred, color="dimgray", linewidth=2.5, label="Theil-Sen Regression")
        # 绘制置信区间
        plt.fill_between(
            x_numeric.ravel(),
            lower_ci,
            upper_ci,
            color="gray",
            alpha=0.3,
            label="95% CI"
        )

        variable_y_label_map = {
            'tp_spring_relative_shap_ratio_new': 'TP_Spring',
            'tp_summer_relative_shap_ratio_new': 'TP_Summer',
            'tp_autumn_relative_shap_ratio_new': 'TP_Autumn'
        }

        # 绘制背景和轴
        plt.gca().set_facecolor('#f5f5f5')  # 坐标轴背景色设置为浅灰色
        plt.xticks(ticks=np.arange(len(years)), labels=years, rotation=45, fontsize=15)
        plt.yticks(fontsize=10)
        plt.ylabel(f'{variable_y_label_map[variable]} effect on forest {dataset}', fontsize=20)
        # 添加网格
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        # 添加图例和颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # 为颜色条创建映射
        cbar = plt.colorbar(sm, aspect=30, pad=0.02)
        plt.legend(fontsize=12, loc="best")
        plt.tight_layout()

        # 保存图片
        plt.savefig(
            os.path.join(os.path.split(path)[0], f'{variable_y_label_map[variable]}_temporal_fitting_theil_sen.jpg'),
            dpi=600
        )
        plt.savefig(
            os.path.join(os.path.split(path)[0], f'{variable_y_label_map[variable]}_temporal_fitting_theil_sen.pdf'),
            dpi=600
        )

    def decependence_plot_analysis(self,path,output_path,condition_variable):

        data = pd.read_csv(path)
        season_name = 'summer'
        tm_variable = f'tm_{season_name}_baseline'
        vpd_variable = f'vpd_{season_name}_baseline'
        tp_variable = f'tp_{season_name}_baseline'
        tp_shap_variable = f'tp_{season_name}_shap_value'

        if condition_variable == 'tm':condition_variable = tm_variable
        if condition_variable == 'vpd':condition_variable = vpd_variable
        lower_bound = data[condition_variable].quantile(0.25)
        upper_bound = data[condition_variable].quantile(0.75)
        data = data[(data[condition_variable] <= lower_bound) | (data[condition_variable] >= upper_bound)]


        df = pd.DataFrame({
            'TP': data[tp_variable],
            'TP_SHAP': data[f'tp_{season_name}_shap_value'],
            f"{condition_variable}": data[condition_variable]
        })
        # 分组：低 VPD 和高 VPD
        low_vpd_threshold = np.percentile(df[f"{condition_variable}"], 50)
        high_vpd_threshold = np.percentile(df[f"{condition_variable}"], 50)

        low_vpd = df[df[f"{condition_variable}"] <= low_vpd_threshold]
        high_vpd = df[df[f"{condition_variable}"] >= high_vpd_threshold]

        # # 绘图：添加拟合线和置信区间    最原始散点图
        # plt.figure(figsize=(8, 6))
        #
        # # 低 VPD
        # sns.regplot(
        #     x=low_vpd["TP_Spring"],
        #     y=low_vpd["SHAP for TP_Spring"],
        #     scatter_kws={"color": "blue", "label": "Low VPD"},
        #     line_kws={"color": "blue"},
        #     ci=95  # 显示置信区间
        # )
        #
        # # 高 VPD
        # sns.regplot(
        #     x=high_vpd["TP_Spring"],
        #     y=high_vpd["SHAP for TP_Spring"],
        #     scatter_kws={"color": "red", "label": "High VPD"},
        #     line_kws={"color": "red"},
        #     ci=95  # 显示置信区间
        # )
        #
        # # 添加图例和标签
        # plt.xlabel("tp_spring (Feature Value)", fontsize=12)
        # plt.ylabel("SHAP value for tp_spring", fontsize=12)
        # plt.legend(loc="best", fontsize=12)
        # plt.title("Dependence Plot: tp_spring vs SHAP Value", fontsize=14)
        #
        # # 显示图像
        # plt.show()
        # 计算低 VPD 的拟合公式
        plt.rcParams['font.family'] = 'Arial'
        low_vpd_slope, low_vpd_intercept, _, _, _ = linregress(
            low_vpd["TP"], low_vpd["TP_SHAP"]
        )
        low_vpd_formula = f"y = {low_vpd_slope:.3f}x + {low_vpd_intercept:.3f}"
        if low_vpd_intercept < 0:
            low_vpd_formula = f"y = {low_vpd_slope:.3f}x - {-low_vpd_intercept:.3f}"
        else:
            low_vpd_formula = f"y = {low_vpd_slope:.3f}x + {low_vpd_intercept:.3f}"

        # 计算高 VPD 的拟合公式
        high_vpd_slope, high_vpd_intercept, _, _, _ = linregress(
            high_vpd["TP"], high_vpd["TP_SHAP"]
        )
        if high_vpd_intercept < 0:
            high_vpd_formula = f"y = {high_vpd_slope:.3f}x - {-high_vpd_intercept:.3f}"
        else:
            high_vpd_formula = f"y = {high_vpd_slope:.3f}x + {high_vpd_intercept:.3f}"

        # 创建绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor('#f5f5f5')  # 使用淡灰色

        # 3. 绘制密度图 (轮廓线)
        sns.kdeplot(
            x=low_vpd["TP"],
            y=low_vpd["TP_SHAP"],
            cmap="Blues",
            fill=True,
            alpha=1,
            thresh=0.1,
            levels=10,
            zorder=1,
            linewidths=1
        )

        sns.kdeplot(
            x=high_vpd["TP"],
            y=high_vpd["TP_SHAP"],
            cmap="Reds",
            fill=True,
            alpha=0.8,
            thresh=0.1,
            levels=10,
            zorder=1,
            linewidths=1
        )

        # 4. 绘制回归线和置信区间
        def plot_regression_with_ci(ax, data, color, zorder, quantile):
            x = data["TP"]
            y = data["TP_SHAP"]
            slope, intercept, _, _, _ = linregress(x, y)

            # 获取数据范围，用于截断回归线
            x_min = np.min(x)
            x_max = np.quantile(x, quantile)
            # 计算回归线上的点
            x_range = np.linspace(x_min, x_max, 100)
            y_pred = intercept + slope * x_range
            # 计算置信区间
            n = len(x)
            x_mean = np.mean(x)
            residuals = y - (intercept + slope * x)
            residual_std = np.std(residuals, ddof=2)
            confidence_interval = 1.96 * residual_std * np.sqrt(
                1 / n + (x_range - x_mean) ** 2 / np.sum((x - x_mean) ** 2))
            # 绘制回归线
            ax.plot(x_range, y_pred, color=color, linewidth=2.5, zorder=zorder)

            # 绘制置信区间，设置 alpha 和 linewidths
            ax.fill_between(x_range, y_pred - confidence_interval, y_pred + confidence_interval,
                            color=color, alpha=0.3, zorder=zorder, linewidth=0)

        # 绘制低 VPD 的拟合曲线
        plot_regression_with_ci(ax, low_vpd, "blue", zorder=2, quantile=0.95)

        # 绘制高 VPD 的拟合曲线
        plot_regression_with_ci(ax, high_vpd, "red", zorder=2, quantile=1)

        # 添加图例，包含公式
        plt.plot([], [], color="blue", linewidth=2.5, label=f"Low {condition_variable} Fit: {low_vpd_formula}")
        plt.plot([], [], color="red", linewidth=2.5, label=f"High {condition_variable} Fit: {high_vpd_formula}")
        plt.legend(loc="lower left", fontsize=20)
        # 添加标签和标题
        plt.xlabel("TP", fontsize=20)
        plt.ylabel("TP_SHAP", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(0, np.quantile(low_vpd["TP"], 0.95))
        plt.savefig(output_path, dpi=600)
        plt.close()

    def seasonal_precipitataion_shaprelatively_boxplot(self,paths,analysis_items,output_path):
        seasons = ['Spring', 'Summer', 'Autumn','Winter']
        season_dic = {
            'Spring':'tp_spring_relative_shap_ratio_new', 'Summer':'tp_summer_relative_shap_ratio_new', 'Autumn':'tp_autumn_relative_shap_ratio_new','Winter':'tp_winter_relative_shap_ratio_new',
        }
        indices = ['EVI','SIF']
        # 存储所有指数数据的空列表
        contribution_data = []

        # 遍历所有文件和对应的指数
        for index, path in zip(indices, paths):
            data = pd.read_csv(path)
            data_analysis = data[analysis_items]

            # 重塑数据用于绘图
            melted_data = data_analysis.melt(var_name="Season", value_name="Relative Contribution")
            melted_data["Index"] = index  # 添加指数名称

            # 将数据转换为百分比
            melted_data["Relative Contribution"] = melted_data["Relative Contribution"] * 100

            contribution_data.append(melted_data)

        # 合并所有指数的数据
        contribution_df = pd.concat(contribution_data, ignore_index=True)

        # 设置配色为绿红黄
        custom_palette = ["green", "red", "yellow"]

        # 调整数据顺序以便展示：每个指数对应3个季节
        plt.rcParams['font.family'] = 'Arial'
        plt.figure(figsize=(6,4))
        # 绘制箱型图
        sns.boxplot(
            data=contribution_df,
            x="Index",  # 将x轴设置为指数
            y="Relative Contribution",  # y轴为相对贡献
            hue="Season",  # hue为季节
            palette=custom_palette,  # 设置自定义颜色
            showfliers=False,  # 隐藏离群值
            width=0.5  # 调整箱型图宽度
        )

        # 设置自定义的 x 轴刻度标签
        plt.xticks([0, 1], ['EVI', 'SIF'], fontsize=12)

        # 添加标题和轴标签
        plt.ylabel("Relative Importance (%)", fontsize=14)
        plt.yticks(fontsize=12)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, ['Spring Precipitation', 'Summer Precipitation', 'Autumn Precipitation'], fontsize=12, loc="best")
        # 显示图像
        plt.tight_layout()
        plt.savefig(output_path,dpi=600)
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


    def indices_spatial_visualization(self,data_path,start_year,end_year,indices_file_name):
        data = pd.read_csv(data_path).dropna()
        data['country'] = data['country'].str.replace(r'_plot_\d+', '', regex=True)
        countries = data['country'].unique().tolist()
        paths = [os.path.split(item)[-1] for item in
                 os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
        countries = paths
        for country_name in countries:
            if os.path.exists(os.path.join(f'D:\Data Collection\RS\Disturbance/7080016/{country_name}',
                                           f'Wu Yong/{country_name}_2023_{indices_file_name}_offset_sa.tif')):
                print(country_name)
                continue
            if country_name == 'norway':continue
            indices_path = glob.glob(os.path.join(f'D:\Data Collection\RS\Disturbance/7080016/{country_name}',f'{country_name}_*_{indices_file_name}.tif'))

            mask_tif = os.path.join(f'D:\Data Collection\RS\Disturbance/7080016/{country_name}','Wu Yong/mask_combined.tif')
            mask_nc = self.tif_tonc(mask_tif)
            mask_num_nc = self.spatial_aggregation(mask_nc, xr.open_dataset(r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei03.nc'), 'sum')
            mask_array = mask_nc['data'].data
            mask_array_evi = np.repeat(np.expand_dims(mask_array.data, 0), 23, axis=0)
            mask_array_SIF = np.repeat(np.expand_dims(mask_array.data, 0), 16, axis=0)

            if indices_file_name=='EVI':
                nc_data = self.stack_bands_tonc_EVI(indices_path, 1)
                baseline_data = nc_data.mean(dim='time')
                offset_data = nc_data - baseline_data
                offset_data_sa = self.spatial_aggregation(
                    offset_data.where(mask_array_evi == 1),
                    xr.open_dataset(r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei03.nc'), 'mean')
            else:
                SIF_path = r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\2 forest sensitivity to water\data collection/SIF_tempory.nc'
                nc_data = self.get_SIF_current_region(SIF_path,os.path.join(f'D:\Data Collection\RS\Disturbance/7080016/{country_name}',f'{country_name}_2001_EVI.tif'))
                baseline_data = nc_data.mean(dim='year')
                offset_data = nc_data - baseline_data
                offset_data_sa = self.spatial_aggregation(
                    offset_data.where(mask_array_SIF == 1),
                    xr.open_dataset(r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei03.nc'), 'mean')
            for year in range(start_year, end_year):
                if indices_file_name == 'EVI':
                    year_data = offset_data_sa.sel(time=f'{year}-01-01')
                else:
                    year_data = offset_data_sa.sel(year=year)
                year_data = year_data.rename({'lon': 'x', 'lat': 'y'})
                year_data = year_data.rio.set_spatial_dims(x_dim='x', y_dim='y')
                year_data = year_data.rio.write_crs("EPSG:4326")  # 替换为实际的投影 EPSG 编码
                year_data.rio.to_raster(os.path.join(f'D:\Data Collection\RS\Disturbance/7080016/{country_name}',f'Wu Yong/{country_name}_{year}_{indices_file_name}_offset_sa.tif'))

    def plot_tifs_with_colorbar(self,tif_paths, output_path, cmap='viridis', ncols=5):
        """
        按时间顺序绘制 TIFF 文件并统一配色和图例。
        参数：
        tif_paths: list of str
            按时间顺序排列的 TIFF 文件路径列表。
        output_path: str
            输出图片路径（如 PNG 文件）。
        cmap: str
            Matplotlib 的配色方案名称（如 'viridis', 'plasma', 'cividis' 等）。
        ncols: int
            每行的子图数量。
        """
        plt.rcParams['font.family'] = 'Arial'
        # 设置统一的配色范围
        vmin, vmax = None, None  # 自动计算数据范围
        for tif_path in tif_paths:
            with rasterio.open(tif_path) as src:
                data = src.read(1)
                if vmin is None or np.nanmin(data) < vmin:
                    vmin = np.nanmin(data)
                if vmax is None or np.nanmax(data) > vmax:
                    vmax = np.nanmax(data)
        # 布局设置
        nrows = int(np.ceil(len(tif_paths) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), constrained_layout=True)
        axes = axes.flatten()

        # 绘制每个 TIFF 文件
        for idx, tif_path in enumerate(tif_paths):
            with rasterio.open(tif_path) as src:
                data = src.read(1)
                transform = src.transform  # 获取地理变换信息
                crs = src.crs  # 获取 TIFF 的投影信息
            # 绘制 TIFF 数据
            im = axes[idx].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[idx].set_title(os.path.basename(tif_path).split('.')[0],fontsize=36)  # 使用文件名作为标题
            axes[idx].axis('off')  # 隐藏坐标轴

        # 隐藏多余的子图
        for ax in axes[len(tif_paths):]:
            ax.axis('off')

        # 添加统一的颜色条
        cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.05, pad=0.05)
        cbar.set_label('Value', fontsize=24)
        cbar.ax.tick_params(labelsize=20)  # 设置颜色条刻度字体大小

        # 保存图像
        plt.savefig(output_path, dpi=600)
        plt.savefig(output_path.replace('.jpg','.pdf'), dpi=600)
        plt.close()
        print(f"图像已保存到 {output_path}")

    def plot_indices_timeseries_offset_with_colorbar_left_right_gradient(self,tif_paths, output_path, cmap='viridis', ncols=5):

        years = np.arange(1986, 2021)
        values = np.random.uniform(-0.3, 0.3, len(years))  # 模拟数据

        # 定义颜色映射
        cmap = get_cmap("RdYlGn")  # 使用 'RdYlGn' 渐变色
        norm = Normalize(vmin=-0.3, vmax=0.3)  # 数据归一化

        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(years, values, color='black', linewidth=1.5)

        # 使用梯度填充
        for i in range(len(years) - 1):
            # 为每段创建平滑的渐变填充
            x = np.linspace(years[i], years[i + 1], 500)  # 创建插值点
            y = np.linspace(values[i], values[i + 1], 500)
            z = np.linspace(values[i], values[i + 1], 500)  # 用于计算颜色
            colors = cmap(norm(z))  # 获取渐变颜色

            # 绘制每个渐变片段
            for j in range(len(x) - 1):
                ax.fill_between(
                    [x[j], x[j + 1]], [0, 0], [y[j], y[j + 1]], color=colors[j], edgecolor='none'
                )

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", label="Grassland vitality")

        # 样式调整
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')  # 中心线
        ax.set_xticks(np.arange(1986, 2022, 2))  # 设置 X 轴刻度
        ax.set_xticklabels(np.arange(1986, 2022, 2), rotation=45)  # X 轴标签旋转
        ax.set_ylabel("Grassland vitality")
        ax.set_xlabel("Year")
        ax.set_facecolor("#f0f0f0")  # 设置背景颜色

        plt.tight_layout()
        plt.savefig('test.jpg')
    def plot_indices_timeseries_offset_with_colorbar(self, tif_paths, output_path, cmap='viridis', ncols=5):
        """
        改进点：
        1) 按年份排序，稳住时序
        2) 在线性插值处显式插入 y=0 交点，保证路径连续
        3) 用 imshow 画整幅垂直渐变，然后用 clip path 一次性裁剪成“折线与 y=0 之间的区域”
           ——彻底消除多段 fill 带来的拼接缝
        4) 不再逐段 ax.fill 覆盖；可选加 colorbar
        """

        def load_year_means(paths):
            pairs = []
            for p in paths:
                year = int(os.path.splitext(os.path.basename(p))[0])
                with rasterio.open(p) as src:
                    data = src.read(1)
                    mean_value = np.nanmean(data)
                pairs.append((year, float(mean_value)))
            # 关键：按年份排序，避免乱序导致的交点计算/extent 异常
            pairs.sort(key=lambda x: x[0])
            years = np.array([k for k,_ in pairs], dtype=float)
            vals  = np.array([v for _,v in pairs], dtype=float)
            return years, vals

        def insert_zero_crossings(x, y, y0=0.0):
            """在每个跨越 y0 的线段上插入精确交点，保证后续分段/裁剪连续"""
            xs = [x[0]]; ys = [y[0]]
            for i in range(1, len(x)):
                x1, y1 = x[i-1], y[i-1]
                x2, y2 = x[i],   y[i]
                # 是否跨过 y0（含一上一下）
                if (y1 - y0) * (y2 - y0) < 0:
                    # 线性插值找交点
                    t = (y0 - y1) / (y2 - y1)
                    xi = x1 + t * (x2 - x1)
                    xs.append(xi); ys.append(y0)
                xs.append(x2); ys.append(y2)
            return np.array(xs), np.array(ys)

        def runs_where(mask):
            """把 True/False 序列拆成若干连续 True 的区间 [start, end]（闭区间索引）"""
            runs = []
            start = None
            for i, m in enumerate(mask):
                if m and start is None:
                    start = i
                if (not m) and (start is not None):
                    runs.append((start, i-1))
                    start = None
            if start is not None:
                runs.append((start, len(mask)-1))
            return runs

        def make_band_paths(x, y, y0=0.0):
            """
            生成“折线与 y0 之间”的多段多边形路径集合：
            - 对 y>=y0 的连续区间建一个多边形（底边在 y0，上边沿着折线）
            - 对 y<=y0 的连续区间同理（底边同样在 y0）
            最终返回一个 compound Path，可直接作为 clip path 使用
            """
            paths = []

            # 为了稳妥地把 y==y0 也算在内，分别做 >= 和 <= 两类
            for sign_mask in [(y >= y0), (y <= y0)]:
                for s, e in runs_where(sign_mask):
                    # 该区间的折线顶边
                    x_seg = x[s:e+1]
                    y_seg = y[s:e+1]
                    if len(x_seg) < 2:
                        continue
                    # 多边形顶点顺序：从左下(y0) → 沿折线 → 右下(y0) → 回到起点
                    verts = [(x_seg[0], y0)]
                    verts += list(zip(x_seg, y_seg))
                    verts.append((x_seg[-1], y0))
                    verts.append((x_seg[0], y0))  # CLOSEPOLY 需要重复起点

                    codes = [Path.MOVETO] + [Path.LINETO]*(len(verts)-2) + [Path.CLOSEPOLY]
                    paths.append(Path(verts, codes))

            if not paths:
                return None
            # 合并成一个复合路径
            return Path.make_compound_path(*paths)

        # ========= 数据 =========
        plt.rcParams['font.family'] = 'Arial'
        years, temporal_series = load_year_means(tif_paths)

        # 若全为 NaN 或长度太小，提前返回
        if len(years) < 2 or not np.isfinite(temporal_series).any():
            raise ValueError("无有效时序数据可绘制。")

        # 颜色归一化范围
        vmin, vmax = np.nanmin(temporal_series), np.nanmax(temporal_series)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            raise ValueError("时序中全为 NaN。")
        if np.isclose(vmin, vmax):
            # 防止全常数导致色标崩溃
            eps = 1e-9
            vmin, vmax = vmin - eps, vmax + eps

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = get_cmap(cmap)

        # 插入与 y=0 的交点，保证后续分段连续
        full_x, full_y = insert_zero_crossings(years, temporal_series, y0=0.0)

        # ========= 画图 =========
        fig, ax = plt.subplots(figsize=(15, 5))

        # 先画整体垂直渐变（y 方向）
        res = 800
        gradient = np.linspace(vmin, vmax, res).reshape(-1, 1)
        extent = [float(years.min()), float(years.max()), float(vmin), float(vmax)]
        im = ax.imshow(
            gradient,
            aspect="auto",
            extent=extent,
            origin="lower",
            cmap=cmap,
            norm=norm,
            alpha=1.0,      # 裁剪后就不需要半透明了
            zorder=1
        )

        # 用“折线与 y=0 之间”的复合路径裁剪整幅渐变 —— 关键去缝一步
        band_path = make_band_paths(full_x, full_y, y0=0.0)
        if band_path is not None:
            im.set_clip_path(band_path, transform=ax.transData)

        # 折线置顶
        ax.plot(years, temporal_series, color='black', linewidth=1.5, zorder=3)

        # 中心线
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', zorder=2)

        # 轴与刻度
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(vmin, vmax)
        # 年份用整数刻度（包含末端）
        xticks = np.arange(int(years.min()), int(years.max()) + 1, 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, rotation=45)
        ax.set_ylabel("Index Deviation from Baseline", fontsize=20)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_facecolor("#f0f0f0")

        # 可选：颜色条
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
        # cbar = plt.colorbar(sm, ax=ax, orientation="vertical", label="Grassland vitality")

        plt.tight_layout()
        plt.savefig(output_path, dpi=600)
        plt.savefig(output_path.replace('.jpg', '.pdf'), dpi=600)
        print(f"图像已保存到 {output_path}")

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
    def generate_spatial_basedata(self,phenology_path,SPEI_path):
        phenology_nc = xr.open_dataset(phenology_path)
        base_data = self.spatial_aggregation(phenology_nc, xr.open_dataset(SPEI_path), 'mean')
        base_data['Legacy'] = (('time', 'lat', 'lon'), np.full_like(base_data['SOS'].values, -9999))
        legacy_data = base_data['Legacy'].sel(time='2001-01-01')
        legacy_data.attrs = base_data.attrs
        legacy_data = legacy_data.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
        legacy_data.rio.to_raster(os.path.join(os.path.split(phenology_path)[0],'legacy.tif'))
    def rf_results_legacy_spatial_visualization(self,data_path,season):
        data = pd.read_csv(data_path)
        # data[f'tp_{season}_importance'] = data[f'tp_{season}_shap_value'].abs().div(data[[f'tp_spring_shap_value',
        #                                                                                   f'tp_summer_shap_value',
        #                                                                                   f'tp_autumn_shap_value',
        #                                                                                   f'tp_winter_shap_value']].abs().sum(axis=1))
        # data[f'tp_{season}_importance'] = (data[f'tp_{season}_shap_value'].abs()-(data[f'tp_{season}_shap_value'].abs().mean())).div(data[f'tp_{season}_shap_value'].std())
        data[f'tp_{season}_importance'] = data[f'tp_{season}_shap_value'].abs().div(data[[f'tp_{season}_shap_value',
                                                                                          f'vpd_{season}_shap_value',
                                                                                          f'tm_{season}_shap_value']].abs().sum(axis=1))
        seasonal_items = [f'tp_{season}_importance']
        countries = data['country'].unique().tolist()
        for country_name in countries:
            country_data = data[data['country'] == country_name]
            if country_data['country_chunk'].unique()[0] == 'all':
                base_path = os.path.join(f'D:\Data Collection\RS\Disturbance/7080016/{country_name}/Wu Yong',f'empty_base_all.tif')
                country_basetif = gdal.Open(base_path)
                country_basetif_data = country_basetif.GetRasterBand(1).ReadAsArray()
                modified_data = np.full_like(country_basetif_data, np.nan, dtype=np.float32)
                for _, row in country_data.iterrows():
                    modified_data[int(row['row']), int(row['col'])] = row[f'tp_{season}_importance']

                driver = gdal.GetDriverByName('GTiff')
                # Create a new GeoTIFF file with the modified data
                os.makedirs(r'after_first_revision\5. Spatial pattern/test',exist_ok=True)
                output_path = os.path.join(r'after_first_revision\5. Spatial pattern/test', f'{country_name}.tif')
                out_tif = driver.Create(output_path, country_basetif.RasterXSize, country_basetif.RasterYSize, 1,
                                        gdal.GDT_Float32)
                # Set the GeoTIFF's spatial reference system
                out_tif.SetGeoTransform(country_basetif.GetGeoTransform())
                out_tif.SetProjection(country_basetif.GetProjection())
                # Write the modified data to the new file
                out_band = out_tif.GetRasterBand(1)
                out_band.WriteArray(modified_data)
                out_band.SetNoDataValue(np.nan)
                # Flush data to disk and close the file
                out_band.FlushCache()
                out_tif = None
                print(f'Modified GeoTIFF saved as {output_path}')
            else:
                for chunk_id in country_data['country_chunk'].unique():
                    base_path = os.path.join(f'D:\Data Collection\RS\Disturbance/7080016/{country_name}/Wu Yong',
                                             f'empty_base_{chunk_id}.tif')
                    country_basetif = gdal.Open(base_path)
                    country_basetif_data = country_basetif.GetRasterBand(1).ReadAsArray()
                    modified_data = np.full_like(country_basetif_data, np.nan, dtype=np.float32)
                    for _, row in country_data[country_data['country_chunk']==chunk_id].iterrows():
                        modified_data[int(row['row']), int(row['col'])] = row[f'tp_{season}_importance']

                    driver = gdal.GetDriverByName('GTiff')
                    # Create a new GeoTIFF file with the modified data
                    os.makedirs(r'after_first_revision\5. Spatial pattern/test', exist_ok=True)
                    output_path = os.path.join(r'after_first_revision\5. Spatial pattern/test', f'{country_name}_{chunk_id}.tif')
                    out_tif = driver.Create(output_path, country_basetif.RasterXSize, country_basetif.RasterYSize, 1,
                                            gdal.GDT_Float32)
                    # Set the GeoTIFF's spatial reference system
                    out_tif.SetGeoTransform(country_basetif.GetGeoTransform())
                    out_tif.SetProjection(country_basetif.GetProjection())
                    # Write the modified data to the new file
                    out_band = out_tif.GetRasterBand(1)
                    out_band.WriteArray(modified_data)
                    out_band.SetNoDataValue(np.nan)
                    # Flush data to disk and close the file
                    out_band.FlushCache()
                    out_tif = None
                    print(f'Modified GeoTIFF saved as {output_path}')
    def offset_calculation_spatial_visualization(self,data_path):
        data = pd.read_csv(data_path)
        for indices_name in ['evi','sif']:
            indices_cols = [item for item in data.columns if indices_name in item]
            indices_df =  data[indices_cols]
            indices_baseline = indices_df.mean(axis=1)
            for indices_col in indices_cols:
                data[indices_col.replace(indices_name,f'{indices_name}_anomaly')] = data[indices_col] - indices_baseline

        countries = data['country'].unique().tolist()
        for indices_name in ['evi','sif']:
            indices_cols = [item for item in data.columns if f'{indices_name}_anomaly' in item]
            for indices_col in indices_cols:
                for country_name in countries:
                    country_data = data[data['country'] == country_name]
                    if country_data['country_chunk'].unique()[0] == 'all':
                        base_path = os.path.join(f'D:\Data Collection\RS\Disturbance/7080016/{country_name}/Wu Yong',f'empty_base_all.tif')
                        country_basetif = gdal.Open(base_path)
                        country_basetif_data = country_basetif.GetRasterBand(1).ReadAsArray()
                        modified_data = np.full_like(country_basetif_data, np.nan, dtype=np.float32)
                        for _, row in country_data.iterrows():
                            modified_data[int(row['row']), int(row['col'])] = row[indices_col]

                        driver = gdal.GetDriverByName('GTiff')
                        # Create a new GeoTIFF file with the modified data
                        os.makedirs(f'after_first_revision/6. offset/temp/{indices_name}',exist_ok=True)
                        output_path = os.path.join(f'after_first_revision/6. offset/temp/{indices_name}', f'{country_name}_{indices_col}.tif')
                        out_tif = driver.Create(output_path, country_basetif.RasterXSize, country_basetif.RasterYSize, 1,
                                                gdal.GDT_Float32)
                        # Set the GeoTIFF's spatial reference system
                        out_tif.SetGeoTransform(country_basetif.GetGeoTransform())
                        out_tif.SetProjection(country_basetif.GetProjection())
                        # Write the modified data to the new file
                        out_band = out_tif.GetRasterBand(1)
                        out_band.WriteArray(modified_data)
                        out_band.SetNoDataValue(np.nan)
                        # Flush data to disk and close the file
                        out_band.FlushCache()
                        out_tif = None
                        print(f'Modified GeoTIFF saved as {output_path}')
                    else:
                        for chunk_id in country_data['country_chunk'].unique():
                            base_path = os.path.join(f'D:\Data Collection\RS\Disturbance/7080016/{country_name}/Wu Yong',
                                                     f'empty_base_{chunk_id}.tif')
                            country_basetif = gdal.Open(base_path)
                            country_basetif_data = country_basetif.GetRasterBand(1).ReadAsArray()
                            modified_data = np.full_like(country_basetif_data, np.nan, dtype=np.float32)
                            for _, row in country_data[country_data['country_chunk']==chunk_id].iterrows():
                                modified_data[int(row['row']), int(row['col'])] = row[indices_col]

                            driver = gdal.GetDriverByName('GTiff')
                            # Create a new GeoTIFF file with the modified data
                            os.makedirs(f'after_first_revision/6. offset/temp/{indices_name}', exist_ok=True)
                            output_path = os.path.join(f'after_first_revision/6. offset/temp/{indices_name}', f'{country_name}_{chunk_id}_{indices_col}.tif')
                            out_tif = driver.Create(output_path, country_basetif.RasterXSize, country_basetif.RasterYSize, 1,
                                                    gdal.GDT_Float32)
                            # Set the GeoTIFF's spatial reference system
                            out_tif.SetGeoTransform(country_basetif.GetGeoTransform())
                            out_tif.SetProjection(country_basetif.GetProjection())
                            # Write the modified data to the new file
                            out_band = out_tif.GetRasterBand(1)
                            out_band.WriteArray(modified_data)
                            out_band.SetNoDataValue(np.nan)
                            # Flush data to disk and close the file
                            out_band.FlushCache()
                            out_tif = None
                            print(f'Modified GeoTIFF saved as {output_path}')

    def rf_results_latitude_sensitivity_visualization(self,data_path,year_range):
        tif_files = sorted(glob.glob(os.path.join(data_path, '*.tif')))
        years = list(range(year_range[0], year_range[1]))
        # 2. 存储每个纬度的敏感性
        lat_values = []
        sensitivities = []
        for tif_file in tif_files:
            with rasterio.open(tif_file) as src:
                data = src.read(1)  # 读取第一个波段
                transform = src.transform  # 获取地理信息
                height, width = data.shape
                # 计算像素对应的纬度
                lats = np.array([transform[5] + i * transform[4] for i in range(height)])
                # 对每个纬度计算均值（忽略 NaN 值）
                for lat_idx, lat in enumerate(lats):
                    if lat not in lat_values:
                        lat_values.append(lat)
                        sensitivities.append([])
                    sensitivities[lat_values.index(lat)].append(np.nanmean(data[lat_idx, :]))
        # 3. 对每个纬度拟合趋势（线性回归）
        lat_values = np.array(lat_values)
        sensitivities = np.array(sensitivities)
        trend_slopes = []
        confidence_intervals = []
        for sensitivity in sensitivities:
            X = sm.add_constant(years)  # 添加截距项
            y = np.array(sensitivity)
            model = sm.OLS(y, X).fit()  # 线性回归拟合
            trend_slopes.append(model.params[1])  # 斜率表示变化率
            conf_int = model.conf_int(0.05)  # 95% 置信区间
            confidence_intervals.append(conf_int[1])  # 斜率的置信区间
        trend_slopes = np.array(trend_slopes)
        confidence_intervals = np.array(confidence_intervals)
        # 设置全局字体
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 20  # 字体大小

        plt.figure(figsize=(8, 16))
        plt.plot(trend_slopes, lat_values, label="Sensitivity Trend", color="black", linewidth=2)  # 黑色曲线
        plt.fill_betweenx(lat_values, confidence_intervals[:, 0], confidence_intervals[:, 1],
                          color="gray", alpha=0.3, label="95% CI")  # 灰色阴影
        plt.xlabel("Sensitivity Change Rate")
        plt.ylabel("Latitude")
        plt.title("Latitude vs. Sensitivity Change Rate")
        # 设置图例
        plt.legend(loc="upper left",fontsize=20)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(os.path.join(r'paper plots/sensitivity latitude analysis/',os.path.split(data_path)[-1]+'.jpg'),dpi=600,bbox_inches='tight')



def generate_empty_base(data_path,empty_base_path):
    with rasterio.open(data_path) as src:
        profile = src.profile
        # 创建全是 -9999 的数组
        empty_array = np.full((src.height, src.width), -9999, dtype=profile['dtype'])

    # 写出新的 empty_base.tif
    with rasterio.open(empty_base_path, 'w', **profile) as dst:
        dst.write(empty_array, 1)


def plot_indices_timeseries_with_spatial_panels(
    tif_paths,
    output_path,
    shp_path,
    cmap='viridis',
    ncols=5,
    robust=True,
    percentiles=(2, 98),
    sample_per_image=200000,
    figsize=(16, 12),

    # 上：地图每行高度；下：时序高度（你说“时序比例现在合理”——默认就不改）
    panel_height_ratio=(0.8, 2.2),

    # 颜色条外观与位置
    cb_label="Index value",
    cbar_thickness_frac=0.35,   # 色条厚度（相对所在单元的高度）
    cbar_xshift_frac=0.00,      # 色条整体向右平移（相对所在单元的宽度）
    cbar_length_frac = 0.75,
    # 让时序轴整体右移一点，把 y 轴刻度+轴名“收进去”
    ts_left_shift_frac=0.035,   # 相对整张图宽度的右移比例（可微调 0.03~0.05）
):
    """
    上：空间时序图（每幅显示自身 TIF 边界，叠加 shapefile 外/内边界）
    下：时序折线（无缝渐变裁剪）

    - 横向 colorbar：最后一行若有空格子则用空格子，否则在地图与时序之间加一薄行；
      支持右移/变薄；两端标注 Low/High。
    - 时序图 y 轴不再“凸出”：创建完轴后用 set_position 向右平移 `ts_left_shift_frac`，
      并等量缩窄，保持右边界不变，保证与上方空间图“整体等宽”（含坐标轴与标签）。
    """
    FONT_FAMILY = 'Arial'  # 中文可改 'SimHei'、'Microsoft YaHei'
    MAP_YEAR_FS = 15  # 空间图里角标年份
    CB_LABEL_FS = 15  # 颜色条标签
    CB_TICK_FS = 13  # 颜色条刻度
    TS_YLABEL_FS = 15  # 时序图 y 轴标题
    TS_TICK_FS = 13  # 时序图刻度
    TS_XTICK_FS = 13  # 时序图 x 轴刻度
    LOWHIGH_FS = 13  # 色条两端 "Low/High"
    LABEL_FS = 16  # (a)/(b) 字号
    LABEL_XOFF = 0.015  # 向左的偏移（图坐标比例）
    LABEL_YOFF = 0.002  # 向下的偏移（图坐标比例）
    # ---------- 小工具 ----------
    def load_year_means(paths):
        pairs = []
        for p in paths:
            year = int(os.path.splitext(os.path.basename(p))[0])
            with rasterio.open(p) as src:
                data = src.read(1, masked=True).filled(np.nan)
                mean_value = np.nanmean(data)
            pairs.append((year, float(mean_value)))
        pairs.sort(key=lambda x: x[0])
        years = np.array([k for k, _ in pairs], dtype=float)
        vals  = np.array([v for _, v in pairs], dtype=float)
        return years, vals

    def insert_zero_crossings(x, y, y0=0.0):
        xs = [x[0]]; ys = [y[0]]
        for i in range(1, len(x)):
            x1, y1 = x[i-1], y[i-1]
            x2, y2 = x[i],   y[i]
            if (y1 - y0) * (y2 - y0) < 0:
                t = (y0 - y1) / (y2 - y1)
                xi = x1 + t * (x2 - x1)
                xs.append(xi); ys.append(y0)
            xs.append(x2); ys.append(y2)
        return np.array(xs), np.array(ys)

    def runs_where(mask):
        runs = []; start = None
        for i, m in enumerate(mask):
            if m and start is None: start = i
            if (not m) and (start is not None):
                runs.append((start, i-1)); start = None
        if start is not None: runs.append((start, len(mask)-1))
        return runs

    def make_band_paths(x, y, y0=0.0):
        paths = []
        for sign_mask in [(y >= y0), (y <= y0)]:
            for s, e in runs_where(sign_mask):
                x_seg = x[s:e+1]; y_seg = y[s:e+1]
                if len(x_seg) < 2: continue
                verts = [(x_seg[0], y0)]
                verts += list(zip(x_seg, y_seg))
                verts.append((x_seg[-1], y0))
                verts.append((x_seg[0], y0))
                codes = [Path.MOVETO] + [Path.LINETO]*(len(verts)-2) + [Path.CLOSEPOLY]
                paths.append(Path(verts, codes))
        if not paths: return None
        return Path.make_compound_path(*paths)

    # ---------- 预处理 ----------
    if not tif_paths:
        raise ValueError("tif_paths 为空。")
    tif_paths = sorted(tif_paths, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    # shapefile（保留内部边界）
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("Shapefile 为空。")

    # 时序均值
    years, temporal_series = load_year_means(tif_paths)
    if len(years) < 2 or not np.isfinite(temporal_series).any():
        raise ValueError("无有效时序数据可绘制。")
    ts_vmin = float(np.nanmin(temporal_series))
    ts_vmax = float(np.nanmax(temporal_series))
    if np.isclose(ts_vmin, ts_vmax):
        eps = 1e-9
        ts_vmin, ts_vmax = ts_vmin - eps, ts_vmax + eps
    ts_norm = Normalize(vmin=ts_vmin, vmax=ts_vmax)
    cmap_obj = get_cmap(cmap)
    full_x, full_y = insert_zero_crossings(years, temporal_series, y0=0.0)

    # 统一色阶：对每张完整 TIF 抽样
    samples = []
    items = []
    for p in tif_paths:
        year = int(os.path.splitext(os.path.basename(p))[0])
        with rasterio.open(p) as src:
            arr = src.read(1, masked=True).filled(np.nan)
            ext = plotting_extent(arr, src.transform)
            items.append({
                'year': year,
                'array': arr,
                'extent': ext,
                'crs': src.crs,
                'bounds': (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
            })
            finite = np.isfinite(arr)
            if finite.any():
                vals = arr[finite]
                if vals.size > sample_per_image:
                    idx = np.random.choice(vals.size, sample_per_image, replace=False)
                    samples.append(vals[idx])
                else:
                    samples.append(vals)

    if len(samples) == 0:
        raise ValueError("所有 TIF 都没有有效像元。")
    sample_vals = np.concatenate(samples)
    if robust:
        lo, hi = np.nanpercentile(sample_vals, percentiles)
        map_vmin, map_vmax = float(lo), float(hi)
    else:
        map_vmin, map_vmax = float(np.nanmin(sample_vals)), float(np.nanmax(sample_vals))
        if np.isclose(map_vmin, map_vmax):
            eps = 1e-9
            map_vmin, map_vmax = map_vmin - eps, map_vmax + eps
    map_norm = Normalize(vmin=map_vmin, vmax=map_vmax)

    # ---------- 布局（上地图，下时序；横向 colorbar） ----------
    n_maps = len(items)
    ncols = max(1, int(ncols))
    nrows = math.ceil(n_maps / ncols)
    spare = nrows * ncols - n_maps            # 最后一行空格子数
    has_cbar_row = (spare == 0)               # 铺满则新加一行放色条

    height_ratios = [panel_height_ratio[0]] * nrows
    if has_cbar_row:
        height_ratios += [0.22]               # 色条行薄一些
    height_ratios += [panel_height_ratio[1]]  # 时序

    fig = plt.figure(figsize=figsize)         # 不用 constrained_layout，便于手动 set_position
    gs = GridSpec(nrows=nrows + (1 if has_cbar_row else 0) + 1,
                  ncols=ncols, figure=fig, height_ratios=height_ratios)

    # 统一边距 + 更紧凑的网格
    plt.subplots_adjust(left=0.06, right=0.94, top=0.97, bottom=0.11,
                        wspace=0.03, hspace=0.05)
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['axes.unicode_minus'] = False
    # ---- (A) 空间图（上） ----
    map_axes = []
    im_last = None
    for i, d in enumerate(items):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(gs[r, c])

        im_last = ax.imshow(d['array'], extent=d['extent'], origin='upper',
                            cmap=cmap_obj, norm=map_norm, interpolation='nearest', zorder=1)

        # 视窗范围 = 该 TIF 的边界
        xmin, ymin, xmax, ymax = d['bounds']
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # 叠加 shapefile 边界（外部+内部）
        gdf.to_crs(d['crs']).boundary.plot(ax=ax, color='k', linewidth=0.6, zorder=3)

        # 年份角标（不与图重叠）
        ax.text(0.02, 0.98, f"{int(d['year'])}", transform=ax.transAxes,
                ha='left', va='top', fontsize=MAP_YEAR_FS, weight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1.5))

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
        map_axes.append(ax)
    if len(map_axes) > 0:
        pos_a = map_axes[0].get_position()  # 第一张地图子图的位置（图坐标）
        fig.text(pos_a.x0 - 3.5*LABEL_XOFF,  # 往左挪一点
                 pos_a.y1 - LABEL_YOFF,  # 顶边略往下
                 "(a)", fontsize=LABEL_FS, fontweight="bold",
                 ha="right", va="top")
    grid_left = min(ax.get_position().x0 for ax in map_axes)  # 空间面板最左
    grid_right = max(ax.get_position().x1 for ax in map_axes)  # 空间面板最右
    # ---- 横向 colorbar ----
    if im_last is not None:
        if not has_cbar_row:
            # 放在最后一行的空格子（从第一个空列到末列）
            last_row = nrows - 1
            used_in_last = n_maps - (nrows - 1) * ncols
            c_start = used_in_last
            c_end = ncols
            if c_start < c_end:
                cax = fig.add_subplot(gs[last_row, c_start:c_end])
                bbox = cax.get_position()
                h = bbox.height * cbar_thickness_frac
                y = bbox.y0 + (bbox.height - h) / 2.0

                # 关键：让色条“变短”，且右边界与空间图对齐
                # 目标右边界 = grid_right；目标长度 = 可用宽度 * cbar_length_frac
                available_left = bbox.x0  # 该单元当前的左边界
                desired_len = (grid_right - available_left) * cbar_length_frac
                new_left = grid_right - desired_len  # 右对齐→向左收长度
                cax.set_position([new_left, y, desired_len, h])
                cbar = fig.colorbar(im_last, cax=cax, orientation='horizontal')
                cbar.set_label(cb_label, fontsize=CB_LABEL_FS, labelpad=4)
                cbar.ax.tick_params(labelsize=CB_TICK_FS)
                cbar.outline.set_visible(False)
                cbar.ax.text(0, -0.5, "Low",  transform=cbar.ax.transAxes,
                             ha='left', va='top', fontsize=LOWHIGH_FS)
                cbar.ax.text(1, -0.5, "High", transform=cbar.ax.transAxes,
                             ha='right', va='top', fontsize=LOWHIGH_FS)
        else:
            # 在地图与时序之间新增一整行放色条
            cbar_row = nrows
            cax = fig.add_subplot(gs[cbar_row, :])
            bbox = cax.get_position()
            h = bbox.height * cbar_thickness_frac
            y = bbox.y0 + (bbox.height - h) / 2.0

            # 关键：让色条“变短”，且右边界与空间图对齐
            # 目标右边界 = grid_right；目标长度 = 可用宽度 * cbar_length_frac
            available_left = bbox.x0  # 该单元当前的左边界
            desired_len = (grid_right - available_left) * cbar_length_frac
            new_left = grid_right - desired_len  # 右对齐→向左收长度
            cax.set_position([new_left, y, desired_len, h])
            cbar = fig.colorbar(im_last, cax=cax, orientation='horizontal')
            cbar.set_label(cb_label, fontsize=CB_LABEL_FS, labelpad=4)
            cbar.ax.tick_params(labelsize=CB_TICK_FS)
            cbar.outline.set_visible(False)
            cbar.ax.text(0, -0.5, "Low",  transform=cbar.ax.transAxes,
                         ha='left', va='top', fontsize=LOWHIGH_FS)
            cbar.ax.text(1, -0.5, "High", transform=cbar.ax.transAxes,
                         ha='right', va='top', fontsize=LOWHIGH_FS)

    # ---- (B) 时序（下） ----
    ts_row_index = nrows + (1 if has_cbar_row else 0)
    ax_ts = fig.add_subplot(gs[ts_row_index, :])

    # 无缝渐变 + 裁剪
    res = 800
    gradient = np.linspace(ts_vmin, ts_vmax, res).reshape(-1, 1)
    extent_ts = [float(years.min()), float(years.max()), float(ts_vmin), float(ts_vmax)]
    im_grad = ax_ts.imshow(gradient, aspect="auto", extent=extent_ts, origin="lower",
                           cmap=cmap_obj, norm=ts_norm, alpha=1.0, zorder=1)
    band_path = make_band_paths(full_x, full_y, y0=0.0)
    if band_path is not None:
        im_grad.set_clip_path(band_path, transform=ax_ts.transData)

    ax_ts.plot(years, temporal_series, color='black', linewidth=2.0, zorder=3)
    ax_ts.axhline(0, color='gray', linewidth=1.0, linestyle='--', zorder=2)

    # 轴样式
    ax_ts.set_xlim(extent_ts[0], extent_ts[1])
    ax_ts.set_ylim(ts_vmin, ts_vmax)
    xticks = np.arange(int(years.min()), int(years.max()) + 1, 1)
    ax_ts.set_xticks(xticks)
    ax_ts.set_xticklabels(xticks, rotation=45, fontsize=TS_XTICK_FS)
    ax_ts.set_ylabel("Index deviation from baseline", fontsize=TS_YLABEL_FS, labelpad=6)
    ax_ts.tick_params(axis='y', labelsize=TS_TICK_FS, pad=3)
    ax_ts.set_facecolor("#f0f0f0")
    pos = ax_ts.get_position()
    ax_ts.set_position([grid_left, pos.y0, grid_right - grid_left, pos.height])
    pos_b = ax_ts.get_position()  # 注意：要在 set_position 之后再取
    fig.text(pos_a.x0 - 3.5*LABEL_XOFF,
             pos_b.y1 - LABEL_YOFF,
             "(b)", fontsize=LABEL_FS, fontweight="bold",
             ha="right", va="top")
    # —— 关键：把时序轴整体右移一点，并等量缩窄，右边界不变 —— #
    # pos = ax_ts.get_position()
    # shift = ts_left_shift_frac
    # ax_ts.set_position([pos.x0 + shift, pos.y0, pos.width - shift - 0.02 , pos.height])

    # 导出
    plt.savefig(output_path, dpi=600)
    root, ext = os.path.splitext(output_path)
    plt.savefig(root + '.pdf', dpi=600)
    plt.close(fig)
    print(f"图像已保存到 {output_path}")
def plot_indices_timeseries_with_spatial_panels_with_drought_vertical_line(
    tif_paths,
    output_path,
    shp_path,
    cmap='viridis',
    ncols=5,
    robust=True,
    percentiles=(2, 98),
    sample_per_image=200000,
    figsize=(16, 12),

    # 上：地图每行高度；下：时序高度
    panel_height_ratio=(0.8, 2.2),

    # 颜色条外观与位置
    cb_label="Index value",        # 若保持默认，将自动识别为 EVI/SIF 并替换；手动传入则不会自动改
    cbar_thickness_frac=0.35,
    cbar_xshift_frac=0.00,
    cbar_length_frac=0.75,

    # 时序整体右移（收紧 y 轴刻度与轴名）
    ts_left_shift_frac=0.035,

    # —— 干旱年份竖线 —— #
    drought_years=(2003, 2015, 2018, 2019, 2022),
    drought_line_kwargs=None,

    # —— 新增：时序轴标题与“增/减”标注的细调参数 —— #
    ts_ylabel_pad=24,             # y 轴标题向左移动的距离（越大越靠左）
    ts_between_label_x=-0.06,    # 介于 y 轴与 y 轴标题之间的“Increase/Decrease”x 位置（轴坐标）
    index_name = 'EVI',
):
    """
    上：空间时序图（支持 shapefile 叠加），自动统一色阶与年份角标；
    下：时序折线 + 无缝渐变裁剪；支持干旱年份竖线。
    新增：
      1) 色条自动加学术图注：EVI → 'Forest greenness (EVI)'；SIF → 'Photosynthetic activity (SIF)'；
      2) 色条两端文字改为 'Decrease' / 'Increase'；
      3) 时序 y 轴标题左移，并在 y 轴与标题之间加竖排 'Increase' / 'Decrease'。
    """
    import os, math
    import numpy as np
    import geopandas as gpd
    import rasterio
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec
    from matplotlib.cm import get_cmap
    from matplotlib.path import Path
    from rasterio.plot import plotting_extent
    from matplotlib.lines import Line2D

    # ---------- 字体与样式 ----------
    FONT_FAMILY = 'Arial'
    MAP_YEAR_FS = 15
    CB_LABEL_FS = 15
    CB_TICK_FS = 13
    TS_YLABEL_FS = 15
    TS_TICK_FS = 13
    TS_XTICK_FS = 13
    LOWHIGH_FS = 13
    LABEL_FS = 16
    LABEL_XOFF = 0.015
    LABEL_YOFF = 0.002

    # 竖线样式
    if drought_line_kwargs is None:
        drought_line_kwargs = dict(color='crimson', linestyle='--', linewidth=1.6, alpha=0.9, zorder=2.5)

    # ---------- 根据路径自动识别变量类型用于色条标题 ----------
    paths_lower = " ".join(map(str, tif_paths)).lower()
    auto_cb = None
    if index_name == 'EVI':
        auto_cb = "Forest greenness (EVI)"
    else:
        auto_cb = "Forest photosynthetic activity (SIF)"
    cb_effective_label = auto_cb if auto_cb is not None else cb_label

    # ---------- 小工具 ----------
    def load_year_means(paths):
        pairs = []
        for p in paths:
            year = int(os.path.splitext(os.path.basename(p))[0])
            with rasterio.open(p) as src:
                data = src.read(1, masked=True).filled(np.nan)
                mean_value = np.nanmean(data)
            pairs.append((year, float(mean_value)))
        pairs.sort(key=lambda x: x[0])
        years = np.array([k for k, _ in pairs], dtype=float)
        vals  = np.array([v for _, v in pairs], dtype=float)
        return years, vals

    def insert_zero_crossings(x, y, y0=0.0):
        xs = [x[0]]; ys = [y[0]]
        for i in range(1, len(x)):
            x1, y1 = x[i-1], y[i-1]
            x2, y2 = x[i],   y[i]
            if (y1 - y0) * (y2 - y0) < 0:
                t = (y0 - y1) / (y2 - y1)
                xi = x1 + t * (x2 - x1)
                xs.append(xi); ys.append(y0)
            xs.append(x2); ys.append(y2)
        return np.array(xs), np.array(ys)

    def runs_where(mask):
        runs = []; start = None
        for i, m in enumerate(mask):
            if m and start is None: start = i
            if (not m) and (start is not None):
                runs.append((start, i-1)); start = None
        if start is not None: runs.append((start, len(mask)-1))
        return runs

    def make_band_paths(x, y, y0=0.0):
        paths = []
        for sign_mask in [(y >= y0), (y <= y0)]:
            for s, e in runs_where(sign_mask):
                x_seg = x[s:e+1]; y_seg = y[s:e+1]
                if len(x_seg) < 2: continue
                verts = [(x_seg[0], y0)]
                verts += list(zip(x_seg, y_seg))
                verts.append((x_seg[-1], y0))
                verts.append((x_seg[0], y0))
                codes = [Path.MOVETO] + [Path.LINETO]*(len(verts)-2) + [Path.CLOSEPOLY]
                paths.append(Path(verts, codes))
        if not paths: return None
        return Path.make_compound_path(*paths)

    # ---------- 预处理 ----------
    if not tif_paths:
        raise ValueError("tif_paths 为空。")
    tif_paths = sorted(tif_paths, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("Shapefile 为空。")

    years, temporal_series = load_year_means(tif_paths)
    if len(years) < 2 or not np.isfinite(temporal_series).any():
        raise ValueError("无有效时序数据可绘制。")
    ts_vmin = float(np.nanmin(temporal_series))
    ts_vmax = float(np.nanmax(temporal_series))
    if np.isclose(ts_vmin, ts_vmax):
        eps = 1e-9
        ts_vmin, ts_vmax = ts_vmin - eps, ts_vmax + eps
    ts_norm = Normalize(vmin=ts_vmin, vmax=ts_vmax)
    cmap_obj = get_cmap(cmap)
    full_x, full_y = insert_zero_crossings(years, temporal_series, y0=0.0)

    # 统一色阶抽样
    samples = []
    items = []
    for p in tif_paths:
        year = int(os.path.splitext(os.path.basename(p))[0])
        with rasterio.open(p) as src:
            arr = src.read(1, masked=True).filled(np.nan)
            ext = plotting_extent(arr, src.transform)
            items.append({
                'year': year,
                'array': arr,
                'extent': ext,
                'crs': src.crs,
                'bounds': (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
            })
            finite = np.isfinite(arr)
            if finite.any():
                vals = arr[finite]
                if vals.size > sample_per_image:
                    idx = np.random.choice(vals.size, sample_per_image, replace=False)
                    samples.append(vals[idx])
                else:
                    samples.append(vals)

    if len(samples) == 0:
        raise ValueError("所有 TIF 都没有有效像元。")
    sample_vals = np.concatenate(samples)
    if robust:
        lo, hi = np.nanpercentile(sample_vals, percentiles)
        map_vmin, map_vmax = float(lo), float(hi)
    else:
        map_vmin, map_vmax = float(np.nanmin(sample_vals)), float(np.nanmax(sample_vals))
        if np.isclose(map_vmin, map_vmax):
            eps = 1e-9
            map_vmin, map_vmax = map_vmin - eps, map_vmax + eps
    map_norm = Normalize(vmin=map_vmin, vmax=map_vmax)

    # ---------- 布局 ----------
    n_maps = len(items)
    ncols = max(1, int(ncols))
    nrows = math.ceil(n_maps / ncols)
    spare = nrows * ncols - n_maps
    has_cbar_row = (spare == 0)

    height_ratios = [panel_height_ratio[0]] * nrows
    if has_cbar_row:
        height_ratios += [0.22]
    height_ratios += [panel_height_ratio[1]]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows=nrows + (1 if has_cbar_row else 0) + 1,
                  ncols=ncols, figure=fig, height_ratios=height_ratios)

    plt.subplots_adjust(left=0.06, right=0.94, top=0.97, bottom=0.11,
                        wspace=0.03, hspace=0.05)
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['axes.unicode_minus'] = False

    # ---- (A) 空间图 ----
    map_axes = []
    im_last = None
    for i, d in enumerate(items):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(gs[r, c])

        im_last = ax.imshow(d['array'], extent=d['extent'], origin='upper',
                            cmap=cmap_obj, norm=map_norm, interpolation='nearest', zorder=1)

        xmin, ymin, xmax, ymax = d['bounds']
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        gdf.to_crs(d['crs']).boundary.plot(ax=ax, color='k', linewidth=0.6, zorder=3)

        ax.text(0.02, 0.98, f"{int(d['year'])}", transform=ax.transAxes,
                ha='left', va='top', fontsize=MAP_YEAR_FS, weight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1.5))

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
        map_axes.append(ax)

    if len(map_axes) > 0:
        pos_a = map_axes[0].get_position()
        fig.text(pos_a.x0 - 3.5*LABEL_XOFF,
                 pos_a.y1 - LABEL_YOFF,
                 "(a)", fontsize=LABEL_FS, fontweight="bold",
                 ha="right", va="top")
    grid_left = min(ax.get_position().x0 for ax in map_axes)
    grid_right = max(ax.get_position().x1 for ax in map_axes)

    # ---- 横向 colorbar ----
    def _draw_cbar(cax_like):
        bbox = cax_like.get_position()
        h = bbox.height * cbar_thickness_frac
        y = bbox.y0 + (bbox.height - h) / 2.0
        available_left = bbox.x0
        desired_len = (grid_right - available_left) * cbar_length_frac
        new_left = grid_right - desired_len
        cax_like.set_position([new_left + cbar_xshift_frac*(grid_right-new_left), y, desired_len, h])
        cbar = fig.colorbar(im_last, cax=cax_like, orientation='horizontal')
        # 标题：自动识别的图注（EVI/SIF），或保留传入 cb_label
        cbar.set_label(cb_effective_label, fontsize=CB_LABEL_FS, labelpad=4)
        cbar.ax.xaxis.set_label_position('top')  # ⬅️ 关键：标题在上
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.ax.tick_params(labelsize=CB_TICK_FS)
        cbar.outline.set_visible(False)

        # 两端文字：Decrease / Increase（学术表达）
        cbar.ax.text(0, -0.5, "Decrease",  transform=cbar.ax.transAxes,
                     ha='left', va='top', fontsize=LOWHIGH_FS)
        cbar.ax.text(1, -0.5, "Increase", transform=cbar.ax.transAxes,
                     ha='right', va='top', fontsize=LOWHIGH_FS)
        return cbar

    if im_last is not None:
        if not has_cbar_row:
            last_row = nrows - 1
            used_in_last = n_maps - (nrows - 1) * ncols
            c_start = used_in_last
            c_end = ncols
            if c_start < c_end:
                cax = fig.add_subplot(gs[last_row, c_start:c_end])
                _draw_cbar(cax)
        else:
            cbar_row = nrows
            cax = fig.add_subplot(gs[cbar_row, :])
            _draw_cbar(cax)

    # ---- (B) 时序（下） ----
    ts_row_index = nrows + (1 if has_cbar_row else 0)
    ax_ts = fig.add_subplot(gs[ts_row_index, :])

    # 渐变裁剪
    res = 800
    gradient = np.linspace(ts_vmin, ts_vmax, res).reshape(-1, 1)
    extent_ts = [float(years.min()), float(years.max()), float(ts_vmin), float(ts_vmax)]
    im_grad = ax_ts.imshow(gradient, aspect="auto", extent=extent_ts, origin="lower",
                           cmap=cmap_obj, norm=ts_norm, alpha=1.0, zorder=1)
    band_path = make_band_paths(full_x, full_y, y0=0.0)
    if band_path is not None:
        im_grad.set_clip_path(band_path, transform=ax_ts.transData)

    ax_ts.plot(years, temporal_series, color='black', linewidth=2.0, zorder=3)
    ax_ts.axhline(0, color='gray', linewidth=1.0, linestyle='--', zorder=2)

    # 干旱年份竖线（只画存在于 years 的年份）
    available_years = set(int(y) for y in years)
    plotted_years = [y for y in drought_years if y in available_years]
    for y in plotted_years:
        ax_ts.axvline(float(y), **drought_line_kwargs)

    # 轴样式
    ax_ts.set_xlim(extent_ts[0], extent_ts[1])
    ax_ts.set_ylim(ts_vmin, ts_vmax)
    xticks = np.arange(int(years.min()), int(years.max()) + 1, 1)
    ax_ts.set_xticks(xticks)
    ax_ts.set_xticklabels(xticks, rotation=45, fontsize=TS_XTICK_FS)
    ax_ts.tick_params(axis='y', labelsize=TS_TICK_FS, pad=3)
    ax_ts.set_facecolor("#f0f0f0")

    # 对齐上方空间图宽度
    pos = ax_ts.get_position()
    ax_ts.set_position([grid_left, pos.y0, grid_right - grid_left, pos.height])
    pos_b = ax_ts.get_position()
    fig.text(pos_a.x0 - 3.5*LABEL_XOFF,
             pos_b.y1 - LABEL_YOFF,
             "(b)", fontsize=LABEL_FS, fontweight="bold",
             ha="right", va="top")

    # y 轴标题左移，并在 y 轴与标题之间加“Increase/Decrease”
    ax_ts.set_ylabel("Index deviation from baseline", fontsize=TS_YLABEL_FS, labelpad=ts_ylabel_pad)
    # 在 y 轴与标题之间加竖排提示（位置可用 ts_between_label_x 微调）
    ax_ts.text(ts_between_label_x, 0.85, "Increase", rotation=90, va="center", ha="center",
               transform=ax_ts.transAxes, fontsize=TS_TICK_FS, clip_on=False)
    ax_ts.text(ts_between_label_x, 0.15, "Decrease", rotation=90, va="center", ha="center",
               transform=ax_ts.transAxes, fontsize=TS_TICK_FS, clip_on=False)

    # 图例（干旱年份）
    if plotted_years:
        legend_handle = Line2D([0], [0], **{k: v for k, v in drought_line_kwargs.items()
                                           if k in ['color','linestyle','linewidth','alpha']})
        legend_label = "Drought years: " + ", ".join(str(y) for y in plotted_years)
        ax_ts.legend([legend_handle], [legend_label],
                     loc='upper left', fontsize=TS_TICK_FS, frameon=False)

    # 导出
    plt.savefig(output_path, dpi=600)
    root, ext = os.path.splitext(output_path)
    plt.savefig(root + '.pdf', dpi=600)
    plt.close(fig)
    print(f"图像已保存到 {output_path}")

from typing import Dict, Mapping, Optional, List, Tuple, Union


def plot_sensitivity_panel(
        weights_paths: Mapping[str, str],
        indices_data_paths: Mapping[str, Mapping[str, str]],
        shp_path: str,
        *,
        out_path: str = "sensitivity_panel.jpg",
        show: bool = False,
        figsize: Tuple[float, float] = (20, 11.5),
        dpi: int = 600,

        # 显示顺序（从上到下）
        season_order: Tuple[str, str, str, str] = ("spring", "summer", "autumn", "winter"),

        # 固定 biogeo 列：数据仍用这些真实编号，但“仅重命名”为 R1..Rn 显示
        fixed_bio_order: Tuple[int, ...] = (2, 3, 4, 5, 7, 8),

        # 从分析与展示中排除
        exclude_biogeo: Tuple[int, ...] = (1, 9),

        # 左列每行下方标签（与场景顺序一一对应）
        weight_row_labels: Tuple[str, ...] = (
                "Broadleaved and mixed (low elevation)",
                "Broadleaved and mixed (high elevation)",
                "Coniferous (low elevation)",
                "Coniferous (high elevation)",
        ),
):
    # ----------------- 全局字体：Arial & 略放大 -----------------
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.size"] = 12
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["xtick.labelsize"] = 11
    mpl.rcParams["ytick.labelsize"] = 11
    mpl.rcParams["legend.fontsize"] = 11

    # ----------------- helpers -----------------
    SEASONS_CANON = ["spring", "summer", "autumn", "winter"]

    def _read_and_standardize_csv(path: str) -> pd.DataFrame:
        """读取 CSV → 智能映射季节列；返回四季 share（|x|/Σ|x|）+ biogeo(int)。"""
        df0 = pd.read_csv(path)

        def pick_season_col(df, season):
            patt = season if season != "autumn" else r"(autumn|fall)"
            cands = [c for c in df.columns if re.search(rf"\b{patt}\b", str(c), flags=re.I)]
            if not cands:
                cands = [c for c in df.columns if re.search(patt, str(c), flags=re.I)]
            if not cands:
                return None

            def score(c):
                cl = str(c).lower()
                return (0 if "shap" in cl else 1,
                        0 if "value" in cl else 1,
                        0 if re.search(r"\btp\b|tp_", cl) else 1,
                        len(cl))

            cands.sort(key=score)
            return cands[0]

        season_map = {}
        for s in SEASONS_CANON:
            col = pick_season_col(df0, s)
            if col is None:
                raise ValueError(f"{os.path.basename(path)} 找不到季节列: {s}\n现有列: {list(df0.columns)}")
            season_map[col] = s
        df = df0.rename(columns=season_map).copy()

        # biogeo：提取数字编号（支持 'R3'/'3-Alpine'）
        bio_col = None
        for c in df.columns:
            if "biogeo" in str(c).lower():
                bio_col = c;
                break
        if bio_col is None:
            df["biogeo"] = 0
        else:
            nums = df[bio_col].astype(str).str.extract(r"(\d+)")[0]
            df["biogeo"] = pd.to_numeric(nums, errors="coerce").fillna(0).astype(int)

        # 四季 share
        abs_vals = df[SEASONS_CANON].abs()
        denom = abs_vals.sum(axis=1).replace(0, np.nan)
        shares = abs_vals.div(denom, axis=0)
        shares["biogeo"] = df["biogeo"]
        return shares

    def truncate_cmap(cmap, minval=0.0, maxval=1.0, n=256):
        return colors.LinearSegmentedColormap.from_list(
            f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
            cmap(np.linspace(minval, maxval, n))
        )

    # 读 shp
    gdf = gpd.read_file(shp_path) if (shp_path and os.path.exists(shp_path)) else None

    # 统一 weights 色标范围（稳健对比：用 2%~98% 分位）
    all_vals = []
    for tif in weights_paths.values():
        if not os.path.exists(tif): continue
        with rasterio.open(tif) as src:
            arr = src.read(1).astype(float)
            if src.nodata is not None: arr[arr == src.nodata] = np.nan
            all_vals.append(arr[np.isfinite(arr)])
    if len(all_vals):
        concat = np.concatenate(all_vals) if len(all_vals) > 1 else all_vals[0]
        w_vmin = float(np.nanpercentile(concat, 2))
        w_vmax = float(np.nanpercentile(concat, 98))
        if not np.isfinite(w_vmin) or not np.isfinite(w_vmax) or w_vmax <= w_vmin:
            w_vmin, w_vmax = np.nanmin(concat), np.nanmax(concat)
    else:
        w_vmin, w_vmax = 0.0, 1.0

    # 预读 CSV（缓存 & 用于第三列色标范围）
    scenarios = list(weights_paths.keys())
    shares_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    rel_p95_vals: List[float] = []
    for sc in scenarios:
        evi_sh = _read_and_standardize_csv(indices_data_paths[sc]['evi'])
        shares_cache[(sc, 'evi')] = evi_sh
        sif_sh = _read_and_standardize_csv(indices_data_paths[sc]['sif'])
        shares_cache[(sc, 'sif')] = sif_sh
        gmean_evi = evi_sh[SEASONS_CANON].mean()
        gmean_sif = sif_sh[SEASONS_CANON].mean()
        for df_sh, gmean in [(evi_sh, gmean_evi), (sif_sh, gmean_sif)]:
            rel = df_sh[SEASONS_CANON] - gmean
            rel_p95_vals.append(np.nanpercentile(np.abs(rel.values), 95))
            #rel = df_sh[SEASONS_CANON] - gmean
            # 把每个样本的季节值减去对应季节的全局均值，得到相对偏离（biogeo 后面就是拿这个“与全局差值”来画）。
            #
            # rel_p95_vals.append(np.nanpercentile(np.abs(rel.values), 95))
            # 对所有偏离值取绝对值（只看偏离幅度，不看正负），再算95 分位数，加入列表。
            # 用 P95 是为了稳健：不受少数极端值影响，但又能反映“多数情况下的最大偏离量”。
    vmax_rel = 0.5 if len(rel_p95_vals) == 0 else min(0.5, max(0.25, float(np.nanmedian(rel_p95_vals))))
    rel_norm = colors.TwoSlopeNorm(vmin=-vmax_rel, vcenter=0.0, vmax=vmax_rel)

    # 宽度比例：左=1.25；中=0.70（窄）；右= 3.30（宽）
    width_ratios = [1.25, 0.70, 3.30]

    # colormaps
    cmap_w = cm.get_cmap("Greens").copy();
    cmap_w.set_bad(color="lightgray")  # NoData → 浅灰
    cmap_glb = truncate_cmap(cm.get_cmap("Blues"), 0.30, 1.00)  # 第二列 Blues 从 30% 起
    cmap_rel = cm.get_cmap("RdBu_r")  # 第三列对称

    # figure 与 grids（更大底部留白）
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=False)
    gs = GridSpec(nrows=len(scenarios), ncols=3, width_ratios=width_ratios,
                  hspace=0.35, wspace=0.15, figure=fig)

    col1_axes, col2_axes, col3_axes = [], [], []
    top_third_row_axes = []  # 第一行第三列各小图轴，用于唯一的 biogeo 顶部标注（仅重命名为 R1..）

    # —— whisker 兜底 —— #
    def whisker_from_series(series: pd.Series, xlim: Tuple[float, float]) -> Tuple[float, float, float]:
        vals = series.dropna().astype(float).values
        if vals.size == 0:
            return np.nan, np.nan, np.nan,np.nan
        if vals.size >= 3:
            q05, q50, q95 = np.nanpercentile(vals, [5, 50, 95])
            qmean = np.nanmean(vals)
        else:
            q50 = np.nanmedian(vals)
            rng = (xlim[1] - xlim[0]) * 0.01
            q05, q95 = q50 - rng, q50 + rng
            qmean = np.nanmean(vals)
        if not np.isfinite(q05) or not np.isfinite(q95) or q95 <= q05:
            q05, q95 = np.nanmin(vals), np.nanmax(vals)
            if q95 <= q05:  # 单点
                rng = (xlim[1] - xlim[0]) * 0.01
                q05, q95 = q50 - rng, q50 + rng
        return q05, q50, q95, qmean

    # ----------------- per row -----------------
    for r, sc in enumerate(scenarios):
        # ===== 左：weights（增强对比 + 有效区边缘等值线）=====
        ax_map = fig.add_subplot(gs[r, 0]);
        col1_axes.append(ax_map)
        ax_map.set_facecolor("white")
        if os.path.exists(weights_paths[sc]):
            with rasterio.open(weights_paths[sc]) as src:
                arr = src.read(1).astype(float)
                if src.nodata is not None: arr[arr == src.nodata] = np.nan
                bounds = src.bounds
                # 增强：PowerNorm 提升中间调对比
                norm_w = colors.PowerNorm(gamma=0.7, vmin=w_vmin, vmax=w_vmax)
                rio_show(arr, ax=ax_map, transform=src.transform,
                         cmap=cmap_w, norm=norm_w, origin='upper', zorder=2)
                # shp 边界（黑色）
                if gdf is not None:
                    try:
                        gdf_crs = gdf.to_crs(src.crs)
                    except Exception:
                        gdf_crs = gdf
                    gdf_crs.boundary.plot(ax=ax_map, color='k', linewidth=1.6, zorder=3)
                # 有效数据区边缘等值线（让 TIF 区域更“显眼”）
                mask = np.isfinite(arr).astype(float)
                if np.any(mask > 0):
                    ny, nx = arr.shape
                    xs = np.linspace(bounds.left, bounds.right, nx)
                    ys = np.linspace(bounds.bottom, bounds.top, ny)
                    X, Y = np.meshgrid(xs, ys)
                    ax_map.contour(X, Y, mask, levels=[0.5], colors='k', linewidths=0.6, alpha=0.8, zorder=3.5)

                ax_map.set_xlim(bounds.left, bounds.right)
                ax_map.set_ylim(bounds.bottom, bounds.top)
                ax_map.set_aspect('equal', adjustable='box')
        ax_map.set_xticks([]);
        ax_map.set_yticks([])
        for sp in ax_map.spines.values(): sp.set_visible(False)
        label = weight_row_labels[r] if r < len(weight_row_labels) else f"Scenario {r + 1}"
        ax_map.text(0.5, -0.070, label, transform=ax_map.transAxes, ha="center", va="top", fontsize=12)

        # ===== 中：global（EVI 上、SIF 下；Blues；点稍小 + 黑描边；仅最底行保留一次 0/1 数字）=====
        mid = gs[r, 1].subgridspec(2, 1, hspace=0.15)
        ax_evi = fig.add_subplot(mid[0, 0]);
        col2_axes.append(ax_evi)
        ax_sif = fig.add_subplot(mid[1, 0]);
        col2_axes.append(ax_sif)
        for ax in (ax_evi, ax_sif):
            ax.set_facecolor("0.95")
            for sp in ax.spines.values(): sp.set_visible(False)

        evi_sh = shares_cache[(sc, 'evi')]
        sif_sh = shares_cache[(sc, 'sif')]

        def draw_global(ax, df_sh: pd.DataFrame, label: str):
            y = np.arange(len(season_order))
            for i, s in enumerate(season_order):
                q05, q50, q95,qmean = whisker_from_series(df_sh[s], (0, 1))
                c = cmap_glb(q50 if np.isfinite(q50) else 0.5)
                ax.hlines(y[i], q05, q95, color=c, lw=0.9, zorder=1)
                ax.scatter([qmean], [y[i]], s=20, color=c, edgecolor="k", linewidths=0.6, zorder=2)
            ax.set_yticks(y, labels=list(season_order))
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.grid(False);
            ax.tick_params(axis='both', length=0)
            ax.set_title(label, fontsize=12, loc='left', pad=2, weight='bold')

        draw_global(ax_evi, evi_sh, "EVI")
        draw_global(ax_sif, sif_sh, "SIF")

        # 只在最后一行 SIF 面板保留一次 0/1（确保仅一个）
        if r == len(scenarios) - 1:
            ax_evi.set_xticks([])
            ax_sif.set_xticks([0, 1], labels=["0", "1"])
        else:
            ax_evi.set_xticks([]);
            ax_sif.set_xticks([])

        # ===== 右：Δ biogeo（固定列顺序；顶部仅一次“重命名 R1..”）=====
        evi_sh = evi_sh.loc[~evi_sh["biogeo"].isin(exclude_biogeo)].copy()
        sif_sh = sif_sh.loc[~sif_sh["biogeo"].isin(exclude_biogeo)].copy()

        bio_cols: List[int] = list(fixed_bio_order)  # 数据用真实编号顺序
        n_b = len(bio_cols)
        right = gs[r, 2].subgridspec(2, n_b, wspace=0.08, hspace=0.15)

        gmean_evi = evi_sh[SEASONS_CANON].mean()
        gmean_sif = sif_sh[SEASONS_CANON].mean()

        # 更紧凑的刻度位置：±0.25（避免最右列被挤没）
        def compact_ticks(vmax):
            t = 0.25
            return [-min(t, vmax), 0.0, min(t, vmax)]

        def draw_rel(ax, df_sh: pd.DataFrame, gmean: pd.Series, bid: int,
                     row_is_bottom: bool, col_idx: int):
            ax.set_facecolor("0.95")
            for sp in ax.spines.values(): sp.set_visible(False)

            sub = df_sh.loc[df_sh["biogeo"] == bid, SEASONS_CANON]
            rel = sub - gmean
            y = np.arange(len(season_order))
            for i, s in enumerate(season_order):
                q05, q50, q95,qmean = whisker_from_series(rel[s], (-vmax_rel, vmax_rel))
                c = cmap_rel(rel_norm(q50 if np.isfinite(q50) else 0.0))
                ax.hlines(y[i], q05, q95, color=c, lw=1.0, zorder=1)
                ax.scatter([qmean], [y[i]], s=20, color=c, edgecolor="k", linewidths=0.6, zorder=2)

            ax.axvline(0, color='k', ls='--', lw=0.6, alpha=0.6)
            ax.set_yticks([]);
            ax.invert_yaxis()
            ax.set_xlim(-vmax_rel, vmax_rel)
            ax.grid(False);
            ax.tick_params(axis='both', length=0)

            # 仅在整图最后一行显示数字，且防重叠：偶数列显示，奇数列留空
            if row_is_bottom and r == len(scenarios) - 1:
                ticks = compact_ticks(vmax_rel)
                if (col_idx % 2) == 0:
                    ax.set_xticks(ticks, labels=[f"{ticks[0]:.2f}", "0", f"{ticks[2]:.2f}"])
                else:
                    ax.set_xticks(ticks, labels=["", "", ""])
            else:
                ax.set_xticks([])

        row_top_axes = []
        for j, bid in enumerate(bio_cols):
            ax_top = fig.add_subplot(right[0, j])
            col3_axes.append(ax_top)
            draw_rel(ax_top, evi_sh, gmean_evi, bid, row_is_bottom=False, col_idx=j)
            # row_top_axes.append((ax_top, j))  # 注意：这里存列索引 j，用于“仅重命名”为 R1..Rn
            row_top_axes.append((ax_top, bid))
            ax_bot = fig.add_subplot(right[1, j])
            col3_axes.append(ax_bot)
            draw_rel(ax_bot, sif_sh, gmean_sif, bid, row_is_bottom=True, col_idx=j)
        if r == 0:
            top_third_row_axes = row_top_axes

    # —— 顶部唯一 biogeo 标注：“仅重命名”为 R1..Rn（不改数据编号）——
    for ax, col_idx in top_third_row_axes:
        ax.text(0.5, 1.06, f"R{col_idx + 1}", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=12, fontweight='bold')

    # ----------------- 底部三条颜色条（更靠下；第二列=Blues；无轴标题） -----------------
    def _bbox_union(ax_list):
        xs0 = [ax.get_position().x0 for ax in ax_list]
        xs1 = [ax.get_position().x1 for ax in ax_list]
        ys0 = [ax.get_position().y0 for ax in ax_list]
        return min(xs0), max(xs1), min(ys0)

    # 更大底部留白
    plt.tight_layout(rect=[0.04, 0.26, 0.99, 0.99])  # bottom=0.26

    # 左：weights（Greens，PowerNorm）
    if col1_axes:
        x0, x1, y0 = _bbox_union(col1_axes)
        cax1 = fig.add_axes([x0, y0 - 0.10, x1 - x0, 0.022])
        m1 = cm.ScalarMappable(norm=colors.PowerNorm(gamma=0.7, vmin=w_vmin, vmax=w_vmax), cmap=cmap_w)
        cb1 = plt.colorbar(m1, cax=cax1, orientation="horizontal")
        cb1.set_label("weights");
        cb1.ax.tick_params(length=0)

    # 中：global（Blues 截断）
    if col2_axes:
        x0, x1, y0 = _bbox_union(col2_axes)
        cax2 = fig.add_axes([x0, y0 - 0.10, x1 - x0, 0.022])
        m2 = cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=1), cmap=cmap_glb)
        cb2 = plt.colorbar(m2, cax=cax2, orientation="horizontal")
        cb2.set_label("sensitivity (global)");
        cb2.ax.tick_params(length=0)

    # 右：Δ（RdBu_r 对称、稳健）
    if col3_axes:
        x0, x1, y0 = _bbox_union(col3_axes)
        cax3 = fig.add_axes([x0, y0 - 0.10, x1 - x0, 0.022])
        m3 = cm.ScalarMappable(norm=rel_norm, cmap=cmap_rel)
        cb3 = plt.colorbar(m3, cax=cax3, orientation="horizontal")
        cb3.set_label("relative (biogeo − global)");
        cb3.ax.tick_params(length=0)

    # 保存
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.replace('.jpg','.pdf'), dpi=dpi, bbox_inches="tight")
    if show: plt.show()
    plt.close(fig)
    print(f"[OK] Figure saved to: {out_path}")

def plot_sensitivity_panel_without_biogeo(
        weights_paths: Mapping[str, str],
        indices_data_paths: Mapping[str, Mapping[str, str]],
        shp_path: str,
        *,
        out_path: str = "sensitivity_panel.jpg",
        show: bool = False,
        figsize: Tuple[float, float] = (16, 12),  # 放大一点
        dpi: int = 600,

        # 显示顺序（从上到下）
        season_order: Tuple[str, str, str, str] = ("spring", "summer", "autumn", "winter"),

        # 以下参数为兼容旧签名，已不再使用
        fixed_bio_order: Tuple[int, ...] = (2, 3, 4, 5, 7, 8),
        exclude_biogeo: Tuple[int, ...] = (1, 9),

        # 左列每行下方标签（与场景顺序一一对应）
        weight_row_labels: Tuple[str, ...] = (
            "Broadleaved and mixed (low elevation)",
            "Broadleaved and mixed (high elevation)",
            "Coniferous (low elevation)",
            "Coniferous (high elevation)",
        ),
):
    # ----------------- 全局字体：调大 -----------------
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.size"] = 13
    mpl.rcParams["axes.titlesize"] = 13
    mpl.rcParams["axes.labelsize"] = 13
    mpl.rcParams["xtick.labelsize"] = 12.5
    mpl.rcParams["ytick.labelsize"] = 12.5
    mpl.rcParams["legend.fontsize"] = 12.5

    # ----------------- helpers -----------------
    SEASONS_CANON = ["spring", "summer", "autumn", "winter"]

    def _read_and_standardize_csv(path: str) -> pd.DataFrame:
        """读取 CSV → 智能映射季节列；返回四季 share（|x|/Σ|x|）+ biogeo(int, 兼容保留)."""
        df0 = pd.read_csv(path)

        def pick_season_col(df, season):
            patt = season if season != "autumn" else r"(autumn|fall)"
            cands = [c for c in df.columns if re.search(rf"\b{patt}\b", str(c), flags=re.I)]
            if not cands:
                cands = [c for c in df.columns if re.search(patt, str(c), flags=re.I)]
            if not cands:
                return None

            def score(c):
                cl = str(c).lower()
                return (0 if "shap" in cl else 1,
                        0 if "value" in cl else 1,
                        0 if re.search(r"\btp\b|tp_", cl) else 1,
                        len(cl))

            cands.sort(key=score)
            return cands[0]

        season_map = {}
        for s in SEASONS_CANON:
            col = pick_season_col(df0, s)
            if col is None:
                raise ValueError(f"{os.path.basename(path)} 找不到季节列: {s}\n现有列: {list(df0.columns)}")
            season_map[col] = s
        df = df0.rename(columns=season_map).copy()

        # biogeo：兼容提取
        bio_col = None
        for c in df.columns:
            if "biogeo" in str(c).lower():
                bio_col = c
                break
        if bio_col is None:
            df["biogeo"] = 0
        else:
            nums = df[bio_col].astype(str).str.extract(r"(\d+)")[0]
            df["biogeo"] = pd.to_numeric(nums, errors="coerce").fillna(0).astype(int)

        # 四季 share
        abs_vals = df[SEASONS_CANON].abs()
        denom = abs_vals.sum(axis=1).replace(0, np.nan)
        shares = abs_vals.div(denom, axis=0)
        shares["biogeo"] = df["biogeo"]
        return shares

    def truncate_cmap(cmap, minval=0.0, maxval=1.0, n=256):
        return colors.LinearSegmentedColormap.from_list(
            f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
            cmap(np.linspace(minval, maxval, n))
        )

    # 读 shp（若提供）
    gdf = gpd.read_file(shp_path) if (shp_path and os.path.exists(shp_path)) else None

    # 统一 weights 色标范围（稳健对比：用 2%~98% 分位）
    all_vals = []
    for tif in weights_paths.values():
        if not os.path.exists(tif):
            continue
        with rasterio.open(tif) as src:
            arr = src.read(1).astype(float)
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            all_vals.append(arr[np.isfinite(arr)])
    if len(all_vals):
        concat = np.concatenate(all_vals) if len(all_vals) > 1 else all_vals[0]
        w_vmin = float(np.nanpercentile(concat, 2))
        w_vmax = float(np.nanpercentile(concat, 98))
        if (not np.isfinite(w_vmin)) or (not np.isfinite(w_vmax)) or (w_vmax <= w_vmin):
            w_vmin, w_vmax = np.nanmin(concat), np.nanmax(concat)
    else:
        w_vmin, w_vmax = 0.0, 1.0

    # 预读 CSV（缓存，供第二列绘制）
    scenarios = list(weights_paths.keys())
    shares_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    for sc in scenarios:
        shares_cache[(sc, 'evi')] = _read_and_standardize_csv(indices_data_paths[sc]['evi'])
        shares_cache[(sc, 'sif')] = _read_and_standardize_csv(indices_data_paths[sc]['sif'])

    # 宽度比例：左=略宽；右=略窄；整体更紧凑（减小间距）
    width_ratios = [1.65, 1.05]
    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=False)
    gs = GridSpec(
        nrows=len(scenarios), ncols=2, width_ratios=width_ratios,
        hspace=0.28,   # 更紧凑
        wspace=0.12,   # 更紧凑
        figure=fig
    )

    col1_axes, col2_axes = [], []

    # —— whisker 兜底 —— #
    def whisker_from_series(series: pd.Series, xlim: Tuple[float, float]) -> Tuple[float, float, float, float]:
        vals = series.dropna().astype(float).values
        if vals.size == 0:
            return np.nan, np.nan, np.nan, np.nan
        if vals.size >= 3:
            q05, q50, q95 = np.nanpercentile(vals, [5, 50, 95])
            qmean = np.nanmean(vals)
        else:
            q50 = np.nanmedian(vals)
            rng = (xlim[1] - xlim[0]) * 0.01
            q05, q95 = q50 - rng, q50 + rng
            qmean = np.nanmean(vals)
        if not np.isfinite(q05) or not np.isfinite(q95) or q95 <= q05:
            q05, q95 = np.nanmin(vals), np.nanmax(vals)
            if q95 <= q05:  # 单点
                rng = (xlim[1] - xlim[0]) * 0.01
                q05, q95 = q50 - rng, q50 + rng
        return q05, q50, q95, qmean

    # ===== 第二列配色：改成第三列的 RdBu_r（以 0.25 为中点）=====
    cmap_rel = cm.get_cmap("RdBu_r")
    norm_glb = colors.TwoSlopeNorm(vmin=0.0, vcenter=0.28, vmax=0.5)

    # ----------------- per row -----------------
    for r, sc in enumerate(scenarios):
        # ===== 左：weights（Greens + PowerNorm + 等值线）=====
        ax_map = fig.add_subplot(gs[r, 0])
        col1_axes.append(ax_map)
        ax_map.set_facecolor("white")
        if os.path.exists(weights_paths[sc]):
            with rasterio.open(weights_paths[sc]) as src:
                arr = src.read(1).astype(float)
                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan
                bounds = src.bounds
                norm_w = colors.PowerNorm(gamma=0.7, vmin=w_vmin, vmax=w_vmax)
                rio_show(arr, ax=ax_map, transform=src.transform,
                         cmap=cm.get_cmap("Greens"), norm=norm_w, origin='upper', zorder=2)
                if gdf is not None:
                    try:
                        gdf_crs = gdf.to_crs(src.crs)
                    except Exception:
                        gdf_crs = gdf
                    gdf_crs.boundary.plot(ax=ax_map, color='k', linewidth=1.6, zorder=3)
                mask = np.isfinite(arr).astype(float)
                if np.any(mask > 0):
                    ny, nx = arr.shape
                    xs = np.linspace(bounds.left, bounds.right, nx)
                    ys = np.linspace(bounds.bottom, bounds.top, ny)
                    X, Y = np.meshgrid(xs, ys)
                    ax_map.contour(X, Y, mask, levels=[0.5], colors='k', linewidths=0.6, alpha=0.8, zorder=3.5)

                ax_map.set_xlim(bounds.left, bounds.right)
                ax_map.set_ylim(bounds.bottom, bounds.top)
                ax_map.set_aspect('equal', adjustable='box')
        ax_map.set_xticks([]); ax_map.set_yticks([])
        for sp in ax_map.spines.values(): sp.set_visible(False)
        label = weight_row_labels[r] if r < len(weight_row_labels) else f"Scenario {r + 1}"
        ax_map.text(0.5, -0.070, label, transform=ax_map.transAxes, ha="center", va="top", fontsize=13)

        # ===== 右：global（EVI 上、SIF 下；改用 RdBu_r + TwoSlopeNorm(0→1, 中心0.25)）=====
        mid = gs[r, 1].subgridspec(2, 1, hspace=0.12)  # 再紧凑一点
        ax_evi = fig.add_subplot(mid[0, 0]); col2_axes.append(ax_evi)
        ax_sif = fig.add_subplot(mid[1, 0]); col2_axes.append(ax_sif)
        for ax in (ax_evi, ax_sif):
            ax.set_facecolor("0.95")
            for sp in ax.spines.values(): sp.set_visible(False)

        evi_sh = shares_cache[(sc, 'evi')]
        sif_sh = shares_cache[(sc, 'sif')]

        def draw_global(ax, df_sh: pd.DataFrame, label: str):
            y = np.arange(len(season_order))
            for i, s in enumerate(season_order):
                q05, q50, q95, qmean = whisker_from_series(df_sh[s], (0, 1))
                # 注意：这里用 RdBu_r + TwoSlopeNorm，把 0.25 作为“等份季节占比”的中性点
                c = cmap_rel(norm_glb(q50 if np.isfinite(q50) else 0.25))
                ax.hlines(y[i], q05, q95, color=c, lw=1.05, zorder=1)
                ax.scatter([qmean], [y[i]], s=24, color=c, edgecolor="k", linewidths=0.65, zorder=2)
            ax.set_yticks(y, labels=list(season_order))
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.grid(False); ax.tick_params(axis='both', length=0)
            ax.set_title(label, fontsize=13, loc='left', pad=2, weight='bold')

        draw_global(ax_evi, evi_sh, "EVI")
        draw_global(ax_sif, sif_sh, "SIF")

        # 只在最后一行 SIF 面板保留一次 0/1（确保仅一个）
        if r == len(scenarios) - 1:
            ax_evi.set_xticks([])
            ax_sif.set_xticks([0, 1], labels=["0", "1"])
        else:
            ax_evi.set_xticks([]); ax_sif.set_xticks([])

    # ----------------- 底部两个颜色条：更靠下、略短 -----------------
    def _bbox_union(ax_list):
        xs0 = [ax.get_position().x0 for ax in ax_list]
        xs1 = [ax.get_position().x1 for ax in ax_list]
        ys0 = [ax.get_position().y0 for ax in ax_list]
        return min(xs0), max(xs1), min(ys0)

    # 留白：进一步紧凑（两条色标）
    plt.tight_layout(rect=[0.04, 0.20, 0.99, 0.99])

    # 左：weights（Greens，PowerNorm）
    if col1_axes:
        x0, x1, y0 = _bbox_union(col1_axes)
        cax1 = fig.add_axes([x0, y0 - 0.088, x1 - x0, 0.020])  # 稍微下移&变薄
        m1 = cm.ScalarMappable(norm=colors.PowerNorm(gamma=0.7, vmin=w_vmin, vmax=w_vmax),
                               cmap=cm.get_cmap("Greens"))
        cb1 = plt.colorbar(m1, cax=cax1, orientation="horizontal")
        cb1.set_label("weights")
        cb1.ax.tick_params(length=0)

    # 右：global（改用 RdBu_r，中心 0.25）
    if col2_axes:
        x0, x1, y0 = _bbox_union(col2_axes)
        cax2 = fig.add_axes([x0, y0 - 0.088, x1 - x0, 0.020])
        m2 = cm.ScalarMappable(norm=norm_glb, cmap=cmap_rel)
        cb2 = plt.colorbar(m2, cax=cax2, orientation="horizontal")
        cb2.set_label("sensitivity (continental)")
        cb2.ax.tick_params(length=0)
        # 可选：显示中心 0.25 刻度，直观表达“等份季节占比”
        cb2.set_ticks([0.0, 0.28, 0.5])

    # 保存
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_path.replace('.jpg', '.pdf'), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[OK] Figure saved to: {out_path}")



from scipy.stats import linregress
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def decependence_plot_analysis_for_single(
    #     针对一个表格一个交互进行可视化
    path,
    output_path,
    season_name='summer',
    # —— 变量命名：按 {var}_{season}_{suffix} 组织列名 ——
    x_feature='tp',                # 自变量（默认 TP）
    shap_suffix='shap_value',      # SHAP 列后缀
    baseline_suffix='baseline',    # 基线列后缀
    condition_var='tm',            # 条件变量（可改成 'vpd'/'tm'/任意）
    # —— 数据筛选与分组 ——
    keep_outside_quantiles=(0.25, 0.75),  # 在条件变量上保留低于 q1 或高于 q3 的样本；传 None 关闭
    split_point='median',          # 低/高组的分割方式：'median' 或 'quantile'
    split_quantile=0.5,            # 当 split_point='quantile' 时使用，如 0.5=中位
    # —— 绘图风格 ——
    line_span='axis',              # 'axis' 覆盖整轴；'group' 仅在该组分位范围内
    show_kde=True,
    kde_levels=8,
    show_sample_scatter=True,
    scatter_frac=0.07,
    # —— 轴与标签 ——
    axis_quantiles=(0.02, 0.98),   # 统一 x/y 轴范围所用分位
    x_label=None,
    y_label=None,
    condition_label=None
):
    """
    绘制：x_feature vs 其 SHAP，在 condition_var 低/高条件下的依赖关系图（KDE + 抽样散点 + 回归线+95%CI）。
    列名假定为：{var}_{season}_{baseline_suffix or shap_suffix}
    例如：tp_summer_baseline, tp_summer_shap_value, tm_summer_baseline, vpd_summer_baseline 等。
    """

    # ====== 列名拼接 ======
    x_col        = f'{x_feature}_{season_name}_{baseline_suffix}'
    shap_col     = f'{x_feature}_{season_name}_{shap_suffix}'
    cond_col     = f'{condition_var}_{season_name}_{baseline_suffix}'

    # ====== 读数 & 列检查 ======
    data = pd.read_csv(path)
    for col in [x_col, shap_col, cond_col]:
        if col not in data.columns:
            raise ValueError(f'找不到列: {col}（请确认命名模式 {{var}}_{season_name}_{{suffix}} 是否正确）')

    df = data[[x_col, shap_col, cond_col]].rename(columns={
        x_col: 'X',
        shap_col: 'Y',
        cond_col: 'COND'
    }).dropna()

    # ====== 条件变量的外侧样本保留（可选；默认保留低25%和高75%两端） ======
    if keep_outside_quantiles is not None:
        q1, q3 = keep_outside_quantiles
        low_b  = df['COND'].quantile(q1)
        high_b = df['COND'].quantile(q3)
        df = df[(df['COND'] <= low_b) | (df['COND'] >= high_b)].copy()

    # ====== 低/高组划分（median 或 指定分位） ======
    if split_point == 'median':
        split_val = np.median(df['COND'])
    elif split_point == 'quantile':
        split_val = np.quantile(df['COND'], split_quantile)
    else:
        raise ValueError("split_point 仅支持 'median' 或 'quantile'")

    low_grp  = df[df['COND'] <  split_val]
    high_grp = df[df['COND'] >= split_val]

    # ====== 统一风格 ======
    sns.set_theme(context='talk', style='whitegrid')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['legend.fontsize'] = 13

    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=600)
    ax.set_facecolor('#fafafa')

    # ====== 轴范围（稳健分位） ======
    x_lo, x_hi = np.nanquantile(df['X'], axis_quantiles)
    y_lo, y_hi = np.nanquantile(df['Y'], axis_quantiles)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # ====== KDE ======
    if show_kde:
        blues = sns.color_palette('Blues', as_cmap=True)
        reds  = sns.color_palette('Reds',  as_cmap=True)
        kde_kws = dict(fill=True, thresh=0.08, levels=kde_levels, linewidths=0.8,
                       bw_adjust=1.1, clip=((x_lo, x_hi), (y_lo, y_hi)))
        if len(low_grp)  > 5: sns.kdeplot(x=low_grp['X'],  y=low_grp['Y'], cmap=blues, alpha=0.25, ax=ax, **kde_kws)
        if len(high_grp) > 5: sns.kdeplot(x=high_grp['X'], y=high_grp['Y'], cmap=reds,  alpha=0.25, ax=ax, **kde_kws)

    # ====== 抽样散点（可选） ======
    if show_sample_scatter:
        rng = np.random.default_rng(2024)
        def sample_df(d):
            if len(d) == 0: return d
            n = max(50, int(len(d) * scatter_frac))
            n = min(n, len(d))
            idx = rng.choice(d.index.to_numpy(), size=n, replace=False)
            return d.loc[idx]
        s_low, s_high = sample_df(low_grp), sample_df(high_grp)
        if len(s_low):  ax.scatter(s_low['X'],  s_low['Y'],  s=9, lw=0, alpha=0.5, color='#2b6cb0', zorder=3)
        if len(s_high): ax.scatter(s_high['X'], s_high['Y'], s=9, lw=0, alpha=0.5, color='#c53030', zorder=3)

    # ====== 回归线 + 95%CI ======
    def fit_and_plot(d, color, span='axis', hi_q=0.98, z=4):
        if len(d) < 3 or np.nanstd(d['X']) < 1e-12:
            return None
        x = d['X'].to_numpy()
        y = d['Y'].to_numpy()
        slope, intercept, r, _, _ = linregress(x, y)

        ax_min, ax_max = ax.get_xlim()
        if span == 'axis':
            left, right = ax_min, ax_max
        else:
            g_min, g_max = np.nanquantile(x, [0.02, hi_q])
            left  = max(ax_min, g_min)
            right = min(ax_max, g_max)
            if not np.isfinite(left) or not np.isfinite(right) or right - left < (ax_max - ax_min) * 1e-3:
                left, right = ax_min, ax_max

        xs = np.linspace(left, right, 400)
        yhat = intercept + slope * xs

        # 95% CI
        n = len(x)
        x_bar = x.mean()
        resid = y - (intercept + slope * x)
        s_err = np.std(resid, ddof=2)
        denom = np.sum((x - x_bar) ** 2)
        ci = np.zeros_like(xs) if denom <= 0 else 1.96 * s_err * np.sqrt(1/n + (xs - x_bar)**2 / denom)

        ax.plot(xs, yhat, color=color, lw=2.6, zorder=z)
        ax.fill_between(xs, yhat - ci, yhat + ci, color=color, alpha=0.20, zorder=z)

        a, b = slope, intercept
        sign = '-' if b < 0 else '+'
        return f"y = {a:.3f}x {sign} {abs(b):.3f}"
        # return f"y = {a:.3f}x {sign} {abs(b):.3f}  (R²={r**2:.3f})"

    low_formula  = fit_and_plot(low_grp,  color='#1e40af', span=line_span)
    high_formula = fit_and_plot(high_grp, color='#b91c1c', span=line_span)

    # ====== 轴标签 & 网格 ======
    if x_label is None:
        x_label = f"{x_feature.capitalize()} ({season_name})"
    if y_label is None:
        y_label = f"SHAP of {x_feature}_{season_name}"
    if condition_label is None:
        condition_label = condition_var.upper()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which='major', alpha=0.18)
    ax.grid(True, which='minor', alpha=0.10)
    ax.minorticks_on()
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    # ====== 图例 ======
    legend_elements = []
    if low_formula:                  legend_elements.append(Line2D([0], [0], color='#1e40af', lw=2.6, label=f'Low {condition_label} fit: {low_formula}'))
    if high_formula:                 legend_elements.append(Line2D([0], [0], color='#b91c1c', lw=2.6, label=f'High {condition_label} fit: {high_formula}'))
    if legend_elements:
        leg = ax.legend(handles=legend_elements, loc='upper left', frameon=True, fancybox=True, framealpha=0.9)
        leg.get_frame().set_edgecolor('#e5e7eb')

    fig.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
def _trunc_cmap(base_cmap, vmin, vmax, n=256, reverse=False):
    xs = np.linspace(vmin, vmax, n)
    cmap = LinearSegmentedColormap.from_list(
        f"{base_cmap.name}_{vmin:.2f}_{vmax:.2f}", base_cmap(xs)
    )
    return cmap.reversed() if reverse else cmap

def _palette_by_condition(condition_var):
    """
    仅颜色随条件变化；其余参数（alpha/范围/绘制逻辑）与 TM 保持一致。
    - TM：蓝(低)/红(高)（原样）
    - VPD：BrBG 绿(低)/棕(高)，使用色带上下两半
    """
    cond = str(condition_var).lower()
    if cond == 'vpd':
        brbg = cm.get_cmap('BrBG')
        # 仅换颜色：低=绿端，高=棕端（取色位置可按需微调 0.65/0.35）
        line_low  = to_hex(brbg(0.9))   # 绿（低 VPD）
        line_high = to_hex(brbg(0.1))   # 棕（高 VPD）
        scat_low  = to_hex(brbg(0.9))
        scat_high = to_hex(brbg(0.1))
        # KDE：使用 BrBG 的上下两半，等价于 TM 用 Blues/Reds 的“整段”
        kde_low   = _trunc_cmap(brbg, 0.50, 1.00)   # 绿半段
        kde_high  = _trunc_cmap(brbg, 0.00, 0.50, reverse=True)   # 棕（反转！）
        return dict(
            line_low=line_low, line_high=line_high,
            scat_low=scat_low, scat_high=scat_high,
            kde_low=kde_low,   kde_high=kde_high,
            kde_alpha=0.25,    # 与 TM 一致
            scat_alpha=0.50    # 与 TM 一致
        )
    else:
        # TM 或默认：蓝/红（保持原样）
        return dict(
            line_low='#1e40af', line_high='#b91c1c',
            scat_low='#2b6cb0', scat_high='#c53030',
            kde_low=sns.color_palette('Blues', as_cmap=True),
            kde_high=sns.color_palette('Reds',  as_cmap=True),
            kde_alpha=0.25,
            scat_alpha=0.50
        )

def _draw_single_on_ax_strict(
    ax,
    df,                      # 已读取好的 DataFrame（包含所需列）
    season_name,
    x_feature='tp',
    shap_suffix='shap_value',
    baseline_suffix='baseline',
    condition_var='tm',
    keep_outside_quantiles=(0.25, 0.75),
    split_point='median',
    split_quantile=0.5,
    line_span='axis',
    show_kde=True,
    kde_levels=8,
    show_sample_scatter=True,
    scatter_frac=0.07,
    axis_quantiles=(0.02, 0.98),
):
    # 背景淡灰
    ax.set_facecolor('#f5f5f5')

    # ====== 列名 ======
    x_col    = f'{x_feature}_{season_name}_{baseline_suffix}'
    shap_col = f'{x_feature}_{season_name}_{shap_suffix}'
    cond_col = f'{condition_var}_{season_name}_{baseline_suffix}'

    for col in (x_col, shap_col, cond_col):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"缺少列：\n{col}", ha='center', va='center',
                    fontsize=10, color='crimson', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            return

    dat = df[[x_col, shap_col, cond_col]].rename(columns={x_col:'X', shap_col:'Y', cond_col:'COND'}).dropna()

    # ====== 条件变量外侧样本 ======
    if keep_outside_quantiles is not None:
        q1, q3 = keep_outside_quantiles
        lo = dat['COND'].quantile(q1); hi = dat['COND'].quantile(q3)
        dat = dat[(dat['COND'] <= lo) | (dat['COND'] >= hi)].copy()

    # ====== 低/高组划分 ======
    if split_point == 'median':
        split_val = np.median(dat['COND'])
    elif split_point == 'quantile':
        split_val = np.quantile(dat['COND'], split_quantile)
    else:
        raise ValueError("split_point 仅支持 'median' 或 'quantile'")
    low_grp  = dat[dat['COND'] <  split_val]
    high_grp = dat[dat['COND'] >= split_val]

    # ====== 轴范围（稳健分位） ======
    if len(dat) >= 3:
        x_lo, x_hi = np.nanquantile(dat['X'], axis_quantiles)
        y_lo, y_hi = np.nanquantile(dat['Y'], axis_quantiles)
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

    # 取配色（根据 condition_var 自动切换）
    pal = _palette_by_condition(condition_var)

    # ====== KDE ======
    if show_kde:
        kde_kws = dict(fill=True, thresh=0.08, levels=kde_levels, linewidths=0.8, bw_adjust=1.1,
                       clip=((x_lo, x_hi), (y_lo, y_hi)))
        if len(low_grp)  > 5: sns.kdeplot(x=low_grp['X'],  y=low_grp['Y'], cmap=pal['kde_low'],  alpha=0.5, ax=ax, **kde_kws)
        if len(high_grp) > 5: sns.kdeplot(x=high_grp['X'], y=high_grp['Y'], cmap=pal['kde_high'], alpha=0.5, ax=ax, **kde_kws)

    # ====== 抽样散点 ======
    if show_sample_scatter:
        rng = np.random.default_rng(2024)
        def sample_df(d):
            if len(d) == 0: return d
            n = max(50, int(len(d) * scatter_frac))
            n = min(n, len(d))
            idx = rng.choice(d.index.to_numpy(), size=n, replace=False)
            return d.loc[idx]
        s_low, s_high = sample_df(low_grp), sample_df(high_grp)
        if len(s_low):  ax.scatter(s_low['X'],  s_low['Y'],  s=9, lw=0, alpha=0.5, color=pal['scat_low'],  zorder=3)
        if len(s_high): ax.scatter(s_high['X'], s_high['Y'], s=9, lw=0, alpha=0.5, color=pal['scat_high'], zorder=3)

    # ====== 回归线 + 95%CI ======
    def fit_and_plot(d, color, span='axis', hi_q=0.98, z=4):
        if len(d) < 3 or np.nanstd(d['X']) < 1e-12:
            return None
        x = d['X'].to_numpy(); y = d['Y'].to_numpy()
        slope, intercept, r, _, _ = linregress(x, y)

        ax_min, ax_max = ax.get_xlim()
        if span == 'axis':
            left, right = ax_min, ax_max
        else:
            g_min, g_max = np.nanquantile(x, [0.02, hi_q])
            left, right = max(ax_min, g_min), min(ax_max, g_max)
            if (not np.isfinite(left)) or (not np.isfinite(right)) or (right-left < (ax_max-ax_min)*1e-3):
                left, right = ax_min, ax_max

        xs = np.linspace(left, right, 400)
        yhat = intercept + slope * xs

        n = len(x); x_bar = x.mean()
        resid = y - (intercept + slope * x)
        s_err = np.std(resid, ddof=2)
        denom = np.sum((x - x_bar)**2)
        ci = np.zeros_like(xs) if denom <= 0 else 1.96 * s_err * np.sqrt(1/n + (xs - x_bar)**2 / denom)

        ax.plot(xs, yhat, color=color, lw=2.6, zorder=4)
        ax.fill_between(xs, yhat - ci, yhat + ci, color=color, alpha=0.20, zorder=4)

        a, b = slope, intercept
        sign = '-' if b < 0 else '+'
        return f"y = {a:.3f}x {sign} {abs(b):.3f}"

    low_formula  = fit_and_plot(low_grp,  color=pal['line_low'],  span=line_span)
    high_formula = fit_and_plot(high_grp, color=pal['line_high'], span=line_span)

    # ====== 样式 & 图例（保持你原风格） ======
    ax.grid(True, which='major', alpha=0.18)
    ax.grid(True, which='minor', alpha=0.10)
    ax.minorticks_on()
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    legend_elements = []
    if low_formula:
        legend_elements.append(Line2D([0], [0], color=pal['line_low'],  lw=2.6,  label=f"{season_name.capitalize()} {condition_var.upper()} 25th percentile\n{low_formula}"))
    if high_formula:
        legend_elements.append(Line2D([0], [0], color=pal['line_high'], lw=2.6, label=f"{season_name.capitalize()} {condition_var.upper()} 75th percentile\n{high_formula}"))
    if legend_elements:
        leg = ax.legend(handles=legend_elements, loc='upper left', frameon=True, fancybox=True, framealpha=0.9, fontsize=9)
        leg.get_frame().set_edgecolor('#e5e7eb')

def grid_evi_sif_tp_by_season_files(
    paths_evi,                      # dict 或 list/tuple（与 seasons 顺序一致）
    paths_sif,                      # dict 或 list/tuple（与 seasons 顺序一致）
    out_path,
    seasons=('spring', 'summer', 'autumn', 'winter'),
    condition_vars=('tm', 'vpd'),
    x_feature='tp',
    shap_suffix_evi='shap_value',
    shap_suffix_sif='shap_value',
    baseline_suffix='baseline',
    # 传给每个子图的参数（保持与你单图一致）
    keep_outside_quantiles=(0.25, 0.75),
    split_point='median',
    split_quantile=0.5,
    line_span='axis',
    show_kde=True,
    kde_levels=8,
    show_sample_scatter=True,
    scatter_frac=0.07,
    axis_quantiles=(0.02, 0.98),
    # 版式
    dpi=600,
    font_family='Arial'
):
    """
    读取 EVI/SIF 各四个季节文件，拼 2(指数:EVI/SIF) × 2(条件:TM/VPD) × 4(季节) 的 16 子图。
    - 每个子图背景淡灰；
    - 底部仅最下行标注 TP(Season)；
    - 左列 y 轴标注 'shap'，其它列不标；
    - 左侧只标一次 EVI（覆盖前两行）与 SIF（覆盖后两行）；
    - TM 颜色蓝/红不变；VPD 自动用 BrBG 绿/棕。
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --------- 正常化路径输入：支持 dict 或 list/tuple 或 单一字符串 ---------
    def _normalize_paths(paths, seasons):
        # 单一字符串：所有季节共用一份（兼容旧用法）
        if isinstance(paths, str):
            return {s: paths for s in seasons}
        # list/tuple：按 seasons 顺序对应
        if isinstance(paths, (list, tuple)):
            if len(paths) != len(seasons):
                raise ValueError("paths 列表长度必须与 seasons 数量一致。")
            return {s: p for s, p in zip(seasons, paths)}
        # dict：必须包含所有季节键
        if isinstance(paths, dict):
            missing = [s for s in seasons if s not in paths]
            if missing:
                raise ValueError(f"paths 缺少这些季节键：{missing}")
            return paths
        raise TypeError("paths_evi/paths_sif 必须是 str、list/tuple 或 dict。")

    paths_evi = _normalize_paths(paths_evi, seasons)
    paths_sif = _normalize_paths(paths_sif, seasons)

    # --------- 读取为 {season: DataFrame}，避免重复读 ---------
    cache_evi = {s: pd.read_csv(paths_evi[s]) for s in seasons}
    cache_sif = {s: pd.read_csv(paths_sif[s]) for s in seasons}

    # --------- 全局风格 ---------
    sns.set_theme(context='talk', style='whitegrid')
    plt.rcParams['font.family'] = font_family
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['legend.fontsize'] = 13

    n_rows = 2 * len(condition_vars)   # 4
    n_cols = len(seasons)              # 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6*n_cols, 2.9*n_rows), dpi=dpi)
    axes = np.atleast_2d(axes)

    for r in range(n_rows):
        is_evi = (r // len(condition_vars) == 0)
        cond   = condition_vars[r % len(condition_vars)]
        shap_suffix_cur = shap_suffix_evi if is_evi else shap_suffix_sif

        for c, season in enumerate(seasons):
            ax = axes[r, c]
            # 背景淡灰
            ax.set_facecolor('#f5f5f5')

            df_cur = cache_evi[season] if is_evi else cache_sif[season]

            _draw_single_on_ax_strict(
                ax=ax, df=df_cur, season_name=season,
                x_feature=x_feature, shap_suffix=shap_suffix_cur,
                baseline_suffix=baseline_suffix, condition_var=cond,
                keep_outside_quantiles=keep_outside_quantiles,
                split_point=split_point, split_quantile=split_quantile,
                line_span=line_span, show_kde=show_kde, kde_levels=kde_levels,
                show_sample_scatter=show_sample_scatter, scatter_frac=scatter_frac,
                axis_quantiles=axis_quantiles
            )

            # 左列 y 轴标 'shap'；其它列隐藏 ytick label
            if c == 0:
                ax.set_ylabel('SHAP')
            else:
                ax.set_ylabel('')
                for lbl in ax.get_yticklabels():
                    lbl.set_visible(False)

            # 仅最底一行：x 轴标 TP(Season)
            if r == n_rows - 1:
                ax.set_xlabel(f"TP({season.capitalize()})")
            else:
                ax.set_xlabel('')
                for lbl in ax.get_xticklabels():
                    lbl.set_visible(False)

            # 不加任何标题/面板字母

    # 紧凑布局后加左侧的大标签（EVI/SIF 各一次）
    fig.tight_layout()

    # —— 用虚线框把前两行与后两行圈起来，并在框内角落标注 EVI / SIF ——

    def _add_dashed_group_including_axes(
            fig,
            axes,
            row_indices,
            label,
            corner='left',  # 'left' or 'right'
            pad_x_px=24,  # 左右向外扩的留白（像素）
            pad_y_px=14,  # 上下向外扩的基础留白（像素）
            inner_gap_px=20,  # 两个框之间的“额外间距”（像素）
            shrink_edge=None,  # 'bottom'：上框底边上移；'top'：下框顶边下移
            outer_top_px=18,  # 顶边再向外扩（像素）
            outer_bottom_px=18,  # 底边再向外扩（像素）
            label_pad_px=10,  # 标签离框边的内边距（像素）
            label_offset_px=8,  # 让标签再往下“吃”一点（像素）——解决文字贴顶边
            lw=1.2, dash=(6, 4),
            color='#9ca3af',
            fontsize=13, fontweight='bold',
    ):
        # 必须先 draw 一次以获取 renderer
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # 聚合目标行的 tightbbox（像素；含坐标轴刻度与轴标签）
        bbs = []
        for r in row_indices:
            for ax in axes[r, :]:
                bbs.append(ax.get_tightbbox(renderer))
        bb_union = Bbox.union(bbs)  # 像素

        # 像素 -> figure 坐标
        fig_w_px = fig.get_size_inches()[0] * fig.dpi
        fig_h_px = fig.get_size_inches()[1] * fig.dpi
        px2fx = 1.0 / fig_w_px
        px2fy = 1.0 / fig_h_px

        pad_x = pad_x_px * px2fx
        pad_y = pad_y_px * px2fy
        gap_y = inner_gap_px * px2fy
        add_top = outer_top_px * px2fy
        add_bot = outer_bottom_px * px2fy
        lbl_pad_x = label_pad_px * px2fx
        lbl_pad_y = label_pad_px * px2fy
        lbl_off_y = label_offset_px * px2fy

        bb_fig = bb_union.transformed(fig.transFigure.inverted())
        x0 = bb_fig.x0 - pad_x
        x1 = bb_fig.x1 + pad_x
        y0 = bb_fig.y0 - pad_y
        y1 = bb_fig.y1 + pad_y

        # 两框之间的“内间距”
        if shrink_edge == 'bottom':
            y0 = y0 + gap_y  # 上框：底边上移
        elif shrink_edge == 'top':
            y1 = y1 - gap_y  # 下框：顶边下压

        # 上下再向外扩一点
        y0 = y0 - add_bot
        y1 = y1 + add_top

        # 画虚线框（包含坐标轴与标签）
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                         transform=fig.transFigure, fill=False,
                         linewidth=lw, linestyle=(0, dash), edgecolor=color,
                         clip_on=False)
        fig.add_artist(rect)

        # 框内角落标签（往下再挪 label_offset_px）
        if corner == 'left':
            tx, ha = x0 + lbl_pad_x, 'left'
        else:
            tx, ha = x1 - lbl_pad_x, 'right'
        ty = y1 - (lbl_pad_y + lbl_off_y)
        fig.text(tx, ty, label, transform=fig.transFigure,
                 ha=ha, va='top', fontsize=fontsize, fontweight=fontweight, color='#374151')

    rows_per_index = len(condition_vars)  # = 2

    # 第一、二行 = EVI（左上角），底边上提；上下/左右都更松
    _add_dashed_group_including_axes(
        fig, axes,
        row_indices=range(0, rows_per_index),
        label='EVI',
        corner='left',
        pad_x_px=24, pad_y_px=14,
        inner_gap_px=24,
        shrink_edge='bottom',
        outer_top_px=18, outer_bottom_px=18,
        label_pad_px=10, label_offset_px=10
    )

    # 第三、四行 = SIF（左上角），顶边下压；上下/左右都更松
    _add_dashed_group_including_axes(
        fig, axes,
        row_indices=range(rows_per_index, 2 * rows_per_index),
        label='SIF',
        corner='left',
        pad_x_px=24, pad_y_px=14,
        inner_gap_px=24,
        shrink_edge='top',
        outer_top_px=18, outer_bottom_px=18,
        label_pad_px=10, label_offset_px=10
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    fig.savefig(out_path.replace('.jpg', '.pdf'), dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def grid_evi_sif_tp_by_season_files_biogeo(
        biogeo_id,
        paths_evi,  # dict 或 list/tuple（与 seasons 顺序一致）
        paths_sif,  # dict 或 list/tuple（与 seasons 顺序一致）
        out_path,
        seasons=('spring', 'summer', 'autumn', 'winter'),
        condition_vars=('tm', 'vpd'),
        x_feature='tp',
        shap_suffix_evi='shap_value',
        shap_suffix_sif='shap_value',
        baseline_suffix='baseline',
        # 传给每个子图的参数（保持与你单图一致）
        keep_outside_quantiles=(0.25, 0.75),
        split_point='median',
        split_quantile=0.5,
        line_span='axis',
        show_kde=True,
        kde_levels=8,
        show_sample_scatter=True,
        scatter_frac=0.07,
        axis_quantiles=(0.02, 0.98),
        # 版式
        dpi=600,
        font_family='Arial'
):
    """
    读取 EVI/SIF 各四个季节文件，拼 2(指数:EVI/SIF) × 2(条件:TM/VPD) × 4(季节) 的 16 子图。
    - 每个子图背景淡灰；
    - 底部仅最下行标注 TP(Season)；
    - 左列 y 轴标注 'shap'，其它列不标；
    - 左侧只标一次 EVI（覆盖前两行）与 SIF（覆盖后两行）；
    - TM 颜色蓝/红不变；VPD 自动用 BrBG 绿/棕。
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --------- 正常化路径输入：支持 dict 或 list/tuple 或 单一字符串 ---------
    def _normalize_paths(paths, seasons):
        # 单一字符串：所有季节共用一份（兼容旧用法）
        if isinstance(paths, str):
            return {s: paths for s in seasons}
        # list/tuple：按 seasons 顺序对应
        if isinstance(paths, (list, tuple)):
            if len(paths) != len(seasons):
                raise ValueError("paths 列表长度必须与 seasons 数量一致。")
            return {s: p for s, p in zip(seasons, paths)}
        # dict：必须包含所有季节键
        if isinstance(paths, dict):
            missing = [s for s in seasons if s not in paths]
            if missing:
                raise ValueError(f"paths 缺少这些季节键：{missing}")
            return paths
        raise TypeError("paths_evi/paths_sif 必须是 str、list/tuple 或 dict。")

    paths_evi = _normalize_paths(paths_evi, seasons)
    paths_sif = _normalize_paths(paths_sif, seasons)

    # --------- 读取为 {season: DataFrame}，避免重复读 ---------
    cache_evi = {s: pd.read_csv(paths_evi[s])[pd.read_csv(paths_evi[s])['biogeo']==biogeo_id] for s in seasons}
    cache_sif = {s: pd.read_csv(paths_sif[s])[pd.read_csv(paths_sif[s])['biogeo']==biogeo_id] for s in seasons}

    # --------- 全局风格 ---------
    sns.set_theme(context='talk', style='whitegrid')
    plt.rcParams['font.family'] = font_family
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['legend.fontsize'] = 13

    n_rows = 2 * len(condition_vars)  # 4
    n_cols = len(seasons)  # 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.9 * n_rows), dpi=dpi)
    axes = np.atleast_2d(axes)

    for r in range(n_rows):
        is_evi = (r // len(condition_vars) == 0)
        cond = condition_vars[r % len(condition_vars)]
        shap_suffix_cur = shap_suffix_evi if is_evi else shap_suffix_sif

        for c, season in enumerate(seasons):
            ax = axes[r, c]
            # 背景淡灰
            ax.set_facecolor('#f5f5f5')

            df_cur = cache_evi[season] if is_evi else cache_sif[season]

            _draw_single_on_ax_strict(
                ax=ax, df=df_cur, season_name=season,
                x_feature=x_feature, shap_suffix=shap_suffix_cur,
                baseline_suffix=baseline_suffix, condition_var=cond,
                keep_outside_quantiles=keep_outside_quantiles,
                split_point=split_point, split_quantile=split_quantile,
                line_span=line_span, show_kde=show_kde, kde_levels=kde_levels,
                show_sample_scatter=show_sample_scatter, scatter_frac=scatter_frac,
                axis_quantiles=axis_quantiles
            )

            # 左列 y 轴标 'shap'；其它列隐藏 ytick label
            if c == 0:
                ax.set_ylabel('SHAP')
            else:
                ax.set_ylabel('')
                for lbl in ax.get_yticklabels():
                    lbl.set_visible(False)

            # 仅最底一行：x 轴标 TP(Season)
            if r == n_rows - 1:
                ax.set_xlabel(f"TP({season.capitalize()})")
            else:
                ax.set_xlabel('')
                for lbl in ax.get_xticklabels():
                    lbl.set_visible(False)

            # 不加任何标题/面板字母

    # 紧凑布局后加左侧的大标签（EVI/SIF 各一次）
    fig.tight_layout()

    # —— 用虚线框把前两行与后两行圈起来，并在框内角落标注 EVI / SIF ——

    def _add_dashed_group_including_axes(
            fig,
            axes,
            row_indices,
            label,
            corner='left',  # 'left' or 'right'
            pad_x_px=24,  # 左右向外扩的留白（像素）
            pad_y_px=14,  # 上下向外扩的基础留白（像素）
            inner_gap_px=20,  # 两个框之间的“额外间距”（像素）
            shrink_edge=None,  # 'bottom'：上框底边上移；'top'：下框顶边下移
            outer_top_px=18,  # 顶边再向外扩（像素）
            outer_bottom_px=18,  # 底边再向外扩（像素）
            label_pad_px=10,  # 标签离框边的内边距（像素）
            label_offset_px=8,  # 让标签再往下“吃”一点（像素）——解决文字贴顶边
            lw=1.2, dash=(6, 4),
            color='#9ca3af',
            fontsize=13, fontweight='bold',
    ):
        # 必须先 draw 一次以获取 renderer
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # 聚合目标行的 tightbbox（像素；含坐标轴刻度与轴标签）
        bbs = []
        for r in row_indices:
            for ax in axes[r, :]:
                bbs.append(ax.get_tightbbox(renderer))
        bb_union = Bbox.union(bbs)  # 像素

        # 像素 -> figure 坐标
        fig_w_px = fig.get_size_inches()[0] * fig.dpi
        fig_h_px = fig.get_size_inches()[1] * fig.dpi
        px2fx = 1.0 / fig_w_px
        px2fy = 1.0 / fig_h_px

        pad_x = pad_x_px * px2fx
        pad_y = pad_y_px * px2fy
        gap_y = inner_gap_px * px2fy
        add_top = outer_top_px * px2fy
        add_bot = outer_bottom_px * px2fy
        lbl_pad_x = label_pad_px * px2fx
        lbl_pad_y = label_pad_px * px2fy
        lbl_off_y = label_offset_px * px2fy

        bb_fig = bb_union.transformed(fig.transFigure.inverted())
        x0 = bb_fig.x0 - pad_x
        x1 = bb_fig.x1 + pad_x
        y0 = bb_fig.y0 - pad_y
        y1 = bb_fig.y1 + pad_y

        # 两框之间的“内间距”
        if shrink_edge == 'bottom':
            y0 = y0 + gap_y  # 上框：底边上移
        elif shrink_edge == 'top':
            y1 = y1 - gap_y  # 下框：顶边下压

        # 上下再向外扩一点
        y0 = y0 - add_bot
        y1 = y1 + add_top

        # 画虚线框（包含坐标轴与标签）
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                         transform=fig.transFigure, fill=False,
                         linewidth=lw, linestyle=(0, dash), edgecolor=color,
                         clip_on=False)
        fig.add_artist(rect)

        # 框内角落标签（往下再挪 label_offset_px）
        if corner == 'left':
            tx, ha = x0 + lbl_pad_x, 'left'
        else:
            tx, ha = x1 - lbl_pad_x, 'right'
        ty = y1 - (lbl_pad_y + lbl_off_y)
        fig.text(tx, ty, label, transform=fig.transFigure,
                 ha=ha, va='top', fontsize=fontsize, fontweight=fontweight, color='#374151')

    rows_per_index = len(condition_vars)  # = 2

    # 第一、二行 = EVI（左上角），底边上提；上下/左右都更松
    _add_dashed_group_including_axes(
        fig, axes,
        row_indices=range(0, rows_per_index),
        label='EVI',
        corner='left',
        pad_x_px=24, pad_y_px=14,
        inner_gap_px=24,
        shrink_edge='bottom',
        outer_top_px=18, outer_bottom_px=18,
        label_pad_px=10, label_offset_px=10
    )

    # 第三、四行 = SIF（左上角），顶边下压；上下/左右都更松
    _add_dashed_group_including_axes(
        fig, axes,
        row_indices=range(rows_per_index, 2 * rows_per_index),
        label='SIF',
        corner='left',
        pad_x_px=24, pad_y_px=14,
        inner_gap_px=24,
        shrink_edge='top',
        outer_top_px=18, outer_bottom_px=18,
        label_pad_px=10, label_offset_px=10
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    fig.savefig(out_path.replace('.jpg','.pdf'), dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_trend_anomaly():
    import os, glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.lines import Line2D
    from scipy.stats import linregress

    # ---------- 基本工具 ----------
    def _sliding_groups(start_year, end_year, n_files):
        years_all = np.arange(start_year, end_year + 1)
        total_years = end_year - start_year + 1
        win = total_years - n_files + 1
        groups = [years_all[i:i + win] for i in range(len(years_all) - win + 1)]
        labels = [f"{g[0]}-{g[-1]}" for g in groups]
        years_single = [int(g[0]) for g in groups]
        return groups, labels, years_single, win

    def _prepare_summary_exact(summary_path, season):
        """
        严格按你单图：仅在 winter 时把 tm_chilling_* → tm_winter_*。
        """
        summ = pd.read_csv(summary_path).dropna()
        if season == 'winter':
            cols = [c for c in summ.columns if 'tm_chilling_' in c]
            for c in cols:
                summ = summ.rename(columns={c: c.replace('chilling', 'winter')})
        return summ

    # ---------- 与单图一致的计算 ----------
    def _compute_shap_ratio_nomerged(paths, season):
        """
        第一列：完全等同 temporal_window_analysis_onlyseason：
        - 不与 summary 合并
        - 每个窗口 CSV 内按像元计算 tp_{season}_importance 后取均值
        """
        denom_cols = [f'tp_{s}_shap_value' for s in ('spring', 'summer', 'autumn', 'winter')]
        ratios = []
        for p in paths:  # 不排序，保持传入顺序
            df = pd.read_csv(p)
            denom = df[denom_cols].abs().sum(axis=1)
            df[f'tp_{season}_importance'] = df[f'tp_{season}_shap_value'].abs().div(denom)
            ratios.append(float(df[f'tp_{season}_importance'].mean()))
        return ratios

    def _compute_shap_ratio_merged(paths, season, summary_df):
        """
        第二/三列用：等同 temporal_window_analysis_with_SPEI_anomaly：
        - merge 交集像元后再对 tp_{season}_importance 取均值
        """
        denom_cols = [f'tp_{s}_shap_value' for s in ('spring', 'summer', 'autumn', 'winter')]
        ratios = []
        for p in paths:
            df = pd.read_csv(p)
            denom = df[denom_cols].abs().sum(axis=1)
            df[f'tp_{season}_importance'] = df[f'tp_{season}_shap_value'].abs().div(denom)
            data_interact = summary_df.merge(
                df, on=['row', 'col', 'country', 'biogeo', 'country_chunk'], how='inner'
            )
            ratios.append(float(data_interact[f'tp_{season}_importance'].mean()))
        return ratios

    def _climate_anomaly_exact(paths, summary_df, climate_name, season, start_year, end_year):
        """
        完全复制你单图的 anomaly 计算（逐 path 合并后再算）：
          baseline_per_pixel = summary[[f'{climate_name}_{season}_*']].mean(axis=1)
          对每个窗口 i：
            - 合并交集 data_interact
            - residuals = data_interact[window_year_cols].mean(axis=1) - data_interact[baseline]
            - anomaly_i = residuals.mean() / data_interact[baseline].std()
        返回与 paths 等长的 list。
        """
        # 先在 summary 上准备 baseline（像元级均值）
        base_cols = [c for c in summary_df.columns if c.startswith(f'{climate_name}_{season}_')]
        baseline_series_full = summary_df[base_cols].mean(axis=1)
        # 放进 summary（与单图一致）
        summary_df = summary_df.copy()
        summary_df[f'{climate_name}_{season}_baseline'] = baseline_series_full

        years_all = np.arange(start_year, end_year + 1)
        total_years = end_year - start_year + 1
        window_size = total_years - len(paths) + 1
        group_years_collection = [years_all[i:i + window_size] for i in range(len(years_all) - window_size + 1)]

        anoms = []
        for i, p in enumerate(paths):
            yrs = group_years_collection[i]
            climate_group_years = [f'{climate_name}_{season}_{int(y)}' for y in yrs]

            df = pd.read_csv(p)  # 合并用
            data_interact = summary_df.merge(
                df, on=['row', 'col', 'country', 'biogeo', 'country_chunk'], how='inner'
            )
            # residuals & anomaly（都在交集像元上）
            residuals = data_interact[climate_group_years].mean(axis=1) - data_interact[
                f'{climate_name}_{season}_baseline']
            denom_std = float(data_interact[f'{climate_name}_{season}_baseline'].std())
            if not np.isfinite(denom_std) or denom_std == 0:
                anoms.append(np.nan)
            else:
                anoms.append(float(residuals.mean()) / denom_std)
        return anoms

    # ---------- 色条（右下，顶部仅数值） ----------
    def _inset_colorbar_toplabel(ax, sm, *, where='lower right', width="18%", height="4%", pad=0.30,
                                 horizontal=True, top_label="", titlesize=10):
        cax = inset_axes(ax, width=width, height=height, loc=where, borderpad=pad)
        cb = plt.colorbar(sm, cax=cax, orientation='horizontal' if horizontal else 'vertical')
        cb.set_ticks([])  # 不显示刻度
        if top_label:
            cb.ax.set_title(top_label, fontsize=titlesize, pad=1.5)
        return cb

    def _letter_tag(ax, text):
        ax.text(0.02, 0.95, text, transform=ax.transAxes,
                ha='left', va='top', fontsize=12, weight='bold')

    # ---------- 第1列：onlyseason 可视化 ----------
    def _trend_ax_with_own_cbar(ax, ratios, labels, season, dataset_label, show_xticks=False):
        ax.set_facecolor('#f5f5f5');
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        x = np.arange(len(ratios), dtype=float)
        if len(ratios) == 0 or np.all(~np.isfinite(ratios)):
            ax.text(0.5, 0.5, 'NO VALID DATA', transform=ax.transAxes,
                    ha='center', va='center', color='gray', fontsize=11)
            ax.set_ylabel(f'TP_{season} effect on forest {dataset_label}', fontsize=12)
            ax.set_xticks([]);
            return

        vals = np.asarray(ratios, dtype=float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if vmin == vmax: vmin -= 1e-9
        cmap = plt.get_cmap('RdBu').reversed();
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        colors = [cmap(norm(v)) if np.isfinite(v) else (0.8, 0.8, 0.8, 1) for v in vals]

        ax.scatter(x, vals, color=colors, edgecolor='black', s=55, zorder=3)

        # OLS 回归（与单图一致） + 纯线段 legend
        m = np.isfinite(x) & np.isfinite(vals)
        if m.sum() >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(x[m], vals[m])
            ci_lo, ci_hi = slope - 1.96 * std_err, slope + 1.96 * std_err
            sns.regplot(x=x[m], y=vals[m], scatter=False, ax=ax,
                        line_kws={"color": "dimgray", "linewidth": 2.5}, ci=95, truncate=False)
            handle = Line2D([0], [0], color='dimgray', linewidth=2.5)
            ax.legend(handles=[handle],
                      labels=[f'OLS slope: {slope:.4f} ({ci_lo:.4f}, {ci_hi:.4f})'],
                      fontsize=9, loc='upper right', frameon=False)
        else:
            ax.text(0.5, 0.08, 'NEED ≥2 WINDOWS', transform=ax.transAxes,
                    ha='center', va='bottom', color='gray', fontsize=9)

        if show_xticks:
            ax.set_xticks(x);
            ax.set_xticklabels(labels, rotation=45, fontsize=9)
        else:
            ax.set_xticks([])

        ax.tick_params(axis='y', labelsize=10)
        ax.set_ylabel(f'TP_{season} effect on forest {dataset_label}', fontsize=12)

        # 色条：上方仅数值
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm);
        sm.set_array([])
        _inset_colorbar_toplabel(ax, sm, top_label=f"{vmin:.2f} – {vmax:.2f}",
                                 where='lower right', width="18%", height="4%", pad=0.28,
                                 horizontal=True, titlesize=10)

    # ---------- 第2/3列：with_SPEI_anomaly 可视化（无 OLS legend） ----------
    def _scatter_reg_with_yearcolor(ax, x_anom, y_ratio, years,
                                    cmap_years, norm_years, xlab,
                                    show_xlabel=False, show_xticks=True,
                                    annotate_points=False):
        ax.set_facecolor('#f5f5f5');
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        x = np.asarray(x_anom, dtype=float)
        y = np.asarray(y_ratio, dtype=float)
        t = np.asarray(years, dtype=float)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)

        if m.sum() == 0:
            if show_xlabel: ax.set_xlabel(xlab, fontsize=12)
            ax.set_ylabel('');
            ax.set_xticks([] if not show_xticks else ax.get_xticks())
            ax.text(0.5, 0.5, 'NO VALID DATA', transform=ax.transAxes,
                    ha='center', va='center', color='gray', fontsize=11)
            return

        ax.scatter(x[m], y[m], c=t[m], cmap=cmap_years, norm=norm_years,
                   edgecolor='black', s=55, zorder=3)

        # 拟合：完全按单图
        sns.regplot(x=x[m], y=y[m], scatter=False, ax=ax,
                    line_kws={"color": "dimgray", "linewidth": 2.5}, ci=95, truncate=False)
        # 不添加 legend（第二/三列不要 OLS SLOPE）

        if annotate_points:
            yrs = t[m].astype(int)
            for xv, yv, lab in zip(x[m], y[m], yrs):
                ax.annotate(str(lab), (xv, yv), textcoords="offset points",
                            xytext=(3, 3), ha='left', va='bottom', fontsize=8)

        # 每图自定 x 刻度（min~max 均分 5 档）；xlabel 仅底行
        if show_xticks:
            x_min, x_max = float(np.nanmin(x[m])), float(np.nanmax(x[m]))
            if x_min == x_max:
                ticks = [x_min];
                labels = [f"{x_min:.2f}"]
            else:
                ticks = np.linspace(x_min, x_max, 5);
                labels = [f"{v:.2f}" for v in ticks]
            ax.set_xticks(ticks);
            ax.set_xticklabels(labels, fontsize=10)
        else:
            ax.set_xticks([])

        ax.set_xlabel(xlab if show_xlabel else '', fontsize=12 if show_xlabel else 10)
        ax.set_ylabel('')  # 不显示 y 轴标题
        ax.tick_params(axis='y', labelsize=10)

        # 色条（右下，顶部仅数值：年份范围）
        sm = plt.cm.ScalarMappable(cmap=cmap_years, norm=norm_years);
        sm.set_array([])
        _inset_colorbar_toplabel(ax, sm, top_label=f"{int(norm_years.vmin)} – {int(norm_years.vmax)}",
                                 where='lower right', width="18%", height="4%", pad=0.38,
                                 horizontal=True, titlesize=10)

    # ---------- 主函数 ----------
    def plot_trend_tm_vpd_4x3(
            data_dir=None,
            summary_path=None,
            dataset_label='EVI',
            seasons=('spring', 'summer', 'autumn', 'winter'),
            start_year=2001, end_year=2023,
            width_ratios=(1.2, 1.0, 1.0),
            figsize=(16, 12),
            filename_prefix='trend_TM_VPD_grid_exact',
            annotate_points=False,  # 第二/三列是否标注年份
            col_wspace=0.16,  # 列间距
            data_paths=None  # 推荐显式传入，确保与单图相同顺序
    ):
        """
        大图 4×3；第二/三列数据与拟合完全对齐单图函数。
        """
        assert (data_dir is not None) or (data_paths is not None), "必须提供 data_dir 或 data_paths"
        assert summary_path is not None, "必须提供 summary_path"

        plt.rcParams['font.family'] = 'Arial'

        # 获取 paths（不排序）
        paths = list(data_paths) if data_paths is not None else glob.glob(os.path.join(data_dir, 'SHAP_summary_*.csv'))
        if len(paths) == 0:
            print("[WARN] no SHAP_summary files found.");
            return

        groups, labels_all, years_single_all, win = _sliding_groups(start_year, end_year, len(paths))
        if len(labels_all) == 0:
            labels_all = [f"W{i + 1}" for i in range(len(paths))]
            years_single_all = list(range(len(paths)))

        # 年份着色（第二/三列）
        cmap_years = plt.get_cmap('PuBu').reversed()
        vmin_y = np.nanmin(years_single_all) if len(years_single_all) else 0
        vmax_y = np.nanmax(years_single_all) if len(years_single_all) else 1
        if vmin_y == vmax_y: vmin_y -= 1e-9
        norm_years = plt.Normalize(vmin=vmin_y, vmax=vmax_y)

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(4, 3, figure=fig, width_ratios=width_ratios,
                      wspace=col_wspace, hspace=0.22)
        axes = np.empty((4, 3), dtype=object)
        for i in range(4):
            for j in range(3):
                axes[i, j] = fig.add_subplot(gs[i, j])

        letters = [f"({chr(ord('a') + k)})" for k in range(12)]
        letter_idx = 0

        for i, season in enumerate(seasons):
            # summary 处理与单图一致（仅 winter 改 tm_chilling_）
            summ = _prepare_summary_exact(summary_path, season)

            # 第1列：onlyseason（不 merge），仅底行显示 xticks
            ratios1 = _compute_shap_ratio_nomerged(paths, season)[:len(labels_all)]
            _trend_ax_with_own_cbar(axes[i, 0], ratios1, labels_all, season, dataset_label,
                                    show_xticks=(i == 3))
            _letter_tag(axes[i, 0], letters[letter_idx]);
            letter_idx += 1

            # 第2/3列：with_SPEI_anomaly 逻辑（merge 后均值 + 逐 path anomaly，完全与单图一致）
            ratios2 = _compute_shap_ratio_merged(paths, season, summ)[:len(labels_all)]
            anom_tm = _climate_anomaly_exact(paths, summ, 'tm', season, start_year, end_year)
            anom_vpd = _climate_anomaly_exact(paths, summ, 'vpd', season, start_year, end_year)

            _scatter_reg_with_yearcolor(axes[i, 1], anom_tm, ratios2, years_single_all,
                                        cmap_years, norm_years, xlab='TM anomaly (Z-Score)',
                                        show_xlabel=(i == 3), show_xticks=True,
                                        annotate_points=annotate_points)
            _letter_tag(axes[i, 1], letters[letter_idx]);
            letter_idx += 1

            _scatter_reg_with_yearcolor(axes[i, 2], anom_vpd, ratios2, years_single_all,
                                        cmap_years, norm_years, xlab='VPD anomaly (Z-Score)',
                                        show_xlabel=(i == 3), show_xticks=True,
                                        annotate_points=annotate_points)
            _letter_tag(axes[i, 2], letters[letter_idx]);
            letter_idx += 1

            # 第二/三列不显示 y 轴标题
            axes[i, 1].set_ylabel('');
            axes[i, 2].set_ylabel('')

        plt.subplots_adjust(left=0.12, right=0.985, top=0.97, bottom=0.10)

        outdir = data_dir if data_dir is not None else os.path.dirname(paths[0])
        os.makedirs(outdir, exist_ok=True)
        out_jpg = os.path.join(outdir, f'{filename_prefix}.jpg')
        out_pdf = os.path.join(outdir, f'{filename_prefix}.pdf')
        fig.savefig(out_jpg, dpi=600)
        fig.savefig(out_pdf, dpi=600)
        print(f"Saved figure:\n{out_jpg}\n{out_pdf}")
        plt.close(fig)

    plot_trend_tm_vpd_4x3(
        data_dir=r'after_first_revision\4. Temporal trend\3. 10 years windows\4. low_elevation_evergreen\images_sif_low_elevation_everygreen',
        summary_path=r'afterfirst_revision_summary_low_elevation_everygreen.csv',
        dataset_label='SIF',
        start_year=2001, end_year=2016,
        annotate_points=True,  # 是否标注年份
        col_wspace=0.14,  # 列间距
        filename_prefix='trend_TM_VPD'
    )


def plot_biogeo_trmporal_trend_for_single(global_dir,biogeo_dirs,biogeo_mapping):
    import os, glob, re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import TwoSlopeNorm
    from scipy.stats import linregress

    # === 与单图一致：不 merge，按窗口 CSV 计算 SHAP 比例并求季节序列 ===
    def _season_ratios_onlyseason(paths, season):
        denom_cols = [f'tp_{s}_shap_value' for s in ('spring', 'summer', 'autumn', 'winter')]
        y = []
        for p in paths:  # 保持传入顺序
            df = pd.read_csv(p)
            denom = df[denom_cols].abs().sum(axis=1)
            df[f'tp_{season}_importance'] = df[f'tp_{season}_shap_value'].abs().div(denom)
            y.append(float(df[f'tp_{season}_importance'].mean()))
        return y

    def _ols_slope_ci(y):
        y = np.asarray(y, dtype=float)
        x = np.arange(len(y), dtype=float)
        m = np.isfinite(y)
        if m.sum() < 2: return np.nan, (np.nan, np.nan)
        slope, _, _, _, se = linregress(x[m], y[m])
        return slope, (slope - 1.96 * se, slope + 1.96 * se)

    def plot_season_rows_biogeo_panel(
            global_dir,
            biogeo_dirs,
            biogeo_mapping=None,
            seasons=('spring', 'summer', 'autumn', 'winter'),
            season_labels=('Spring', 'Summer', 'Autumn', 'Winter'),
            dataset_label='EVI',
            marker_size=58,
            capsize=3,
            out_path='season_rows_biogeo_panel.jpg',
            out_csv='season_rows_biogeo_slopes.csv'
    ):
        import re
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.colors import TwoSlopeNorm

        plt.rcParams['font.family'] = 'Arial'

        # --- 全欧各季节斜率 ---
        g_paths = glob.glob(os.path.join(global_dir, 'SHAP_summary_*.csv'))
        if not g_paths:
            raise FileNotFoundError(f'No SHAP_summary_*.csv in: {global_dir}')
        global_slope = {}
        for s in seasons:
            y = _season_ratios_onlyseason(g_paths, s)
            sl, (lo, hi) = _ols_slope_ci(y)
            global_slope[s] = dict(slope=sl, lo=lo, hi=hi)

        # --- 各 biogeo 斜率 + 绝对偏差 ---
        n_bio = len(biogeo_dirs)
        bio_labels_short = [f"R{i}" for i in range(1, n_bio + 1)]  # y 轴只显示 R1 R2 ...

        bio_stats = {s: [] for s in seasons}  # 每季一个列表，元素：dict(idx, slope, lo, hi, delta_abs)
        csv_rows = []
        # CSV：先写全欧
        for s in seasons:
            csv_rows.append(dict(region='Global', season=s,
                                 slope=global_slope[s]['slope'],
                                 ci_low=global_slope[s]['lo'], ci_high=global_slope[s]['hi'],
                                 global_slope=global_slope[s]['slope'],
                                 delta_abs=np.nan))

        for idx, bdir in enumerate(biogeo_dirs, start=1):
            paths = glob.glob(os.path.join(bdir, 'SHAP_summary_*.csv'))
            if not paths:
                raise FileNotFoundError(f'No SHAP_summary_*.csv in: {bdir}')

            for s in seasons:
                y = _season_ratios_onlyseason(paths, s)
                sl, (lo, hi) = _ols_slope_ci(y)
                g = global_slope[s]['slope']
                delta_abs = sl - g
                bio_stats[s].append(dict(idx=idx - 1, slope=sl, lo=lo, hi=hi, delta_abs=delta_abs, g=g))

                # CSV 行（region 用简写 R#）
                csv_rows.append(dict(region=f"R{idx}", season=s, slope=sl,
                                     ci_low=lo, ci_high=hi, global_slope=g,
                                     delta_abs=delta_abs))

        # —— 统一色标（四季共用同一个范围，按绝对偏差的最大绝对值对称到 0）——
        all_abs = [d['delta_abs'] for s in seasons for d in bio_stats[s] if np.isfinite(d['delta_abs'])]
        vmax = max(1e-12, float(np.nanmax(np.abs(all_abs)))) if all_abs else 1e-12
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap('RdBu_r')

        # x 轴范围覆盖所有季节/biogeo 的 CI 与全欧线（全局一致）
        xs = []
        for s in seasons:
            for d in bio_stats[s]:
                xs += [d['lo'], d['hi']]
            xs += [global_slope[s]['lo'], global_slope[s]['hi'], global_slope[s]['slope']]
        xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
        pad = 0.06 * (xmax - xmin if xmax > xmin else 1.0)
        xlim = (xmin - pad, xmax + pad)

        # 图尺寸：每季一行，高度随 biogeo 数量自适应
        row_h = max(2.2, 0.42 * n_bio)
        fig = plt.figure(figsize=(9.5, len(seasons) * row_h))
        gs = GridSpec(len(seasons), 1, figure=fig, hspace=0.45)

        for r, s in enumerate(seasons):
            ax = fig.add_subplot(gs[r, 0])
            ax.set_facecolor('#f5f5f5')

            # y 轴：R1..Rn，从上到下
            y_pos = np.arange(n_bio)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(bio_labels_short, fontsize=10)
            ax.invert_yaxis()

            # 0 的竖虚线 + 本季节全局 slope 的竖实线（贯穿整行）
            ax.axvline(0, color='black', linestyle='--', linewidth=1.8, zorder=1)
            g = global_slope[s]['slope']
            if np.isfinite(g):
                ax.axvline(g, color='dimgray', linestyle='-', linewidth=2.2, zorder=1)

            # 画每个 biogeo 的 slope + 95%CI（颜色=绝对偏差）
            for d in bio_stats[s]:
                xerr = np.array([[d['slope'] - d['lo']], [d['hi'] - d['slope']]])
                color = cmap(norm(d['delta_abs'])) if np.isfinite(d['delta_abs']) else 'lightgray'
                # 原：color=color, ecolor=color, mec='black', mew=0.6
                ax.errorbar(
                    d['slope'], d['idx'],
                    xerr=np.array([[d['slope'] - d['lo']], [d['hi'] - d['slope']]]),
                    fmt='o',
                    ms=marker_size / 10,
                    markerfacecolor=cmap(norm(d['delta_abs'])) if np.isfinite(d['delta_abs']) else 'lightgray',
                    markeredgecolor='black',
                    markeredgewidth=0.6,
                    ecolor='black',  # 误差棒与端帽=黑色
                    elinewidth=1.2,
                    capsize=capsize,
                    zorder=3
                )

            ax.set_xlim(xlim)
            ax.set_title(season_labels[r], loc='left', fontsize=12, pad=4)
            ax.tick_params(axis='x', labelsize=10)
            ax.set_xlabel('')  # 保持干净

        # 统一色条（右下角，水平；单位与 slope 相同）
        cax = fig.add_axes([0.60, 0.02, 0.30, 0.03])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm);
        sm.set_array([])
        cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cb.set_ticks([-vmax, 0, vmax])
        cb.set_ticklabels([f"{-vmax:.3f}", "0", f"{vmax:.3f}"])
        cb.ax.set_title('Absolute difference vs global', fontsize=10, pad=2)

        plt.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out_path, dpi=600)
        fig.savefig(out_path.replace('.jpg','.pdf'), dpi=600)
        print('Saved figure:', out_path)

        # 导出 CSV（包含 absolute difference）
        df = pd.DataFrame(csv_rows,
                          columns=['region', 'season', 'slope', 'ci_low', 'ci_high', 'global_slope', 'delta_abs'])
        df.to_csv(out_csv, index=False)
        print('Saved CSV:', out_csv)
        return df

    _ = plot_season_rows_biogeo_panel(
        global_dir=global_dir,
        biogeo_dirs=biogeo_dirs,
        biogeo_mapping=biogeo_mapping,
        dataset_label='EVI',
        out_path='season_rows_biogeo_panel_EVI.png',
        out_csv='season_rows_biogeo_slopes_EVI.csv'
    )


def plot_biogeo_trmporal_trend_for_multiscenario(scenarios,indices):
    import os, glob, re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import TwoSlopeNorm

    # 复用：与单图一致，按窗口 CSV 计算季节序列
    def _season_ratios_onlyseason(paths, season):
        denom_cols = [f'tp_{s}_shap_value' for s in ('spring', 'summer', 'autumn', 'winter')]
        y = []
        for p in paths:  # 保持传入顺序
            df = pd.read_csv(p)
            denom = df[denom_cols].abs().sum(axis=1)
            df[f'tp_{season}_importance'] = df[f'tp_{season}_shap_value'].abs().div(denom)
            y.append(float(df[f'tp_{season}_importance'].mean()))
        return y

    from scipy.stats import linregress
    def _ols_slope_ci(y):
        y = np.asarray(y, dtype=float)
        x = np.arange(len(y), dtype=float)
        m = np.isfinite(y)
        if m.sum() < 2: return np.nan, (np.nan, np.nan)
        slope, _, _, _, se = linregress(x[m], y[m])
        return slope, (slope - 1.96 * se, slope + 1.96 * se)

    def plot_season_rows_biogeo_panel_multi(
            scenarios,
            seasons=('spring', 'summer', 'autumn', 'winter'),
            season_labels=('Spring', 'Summer', 'Autumn', 'Winter'),
            dataset_label='EVI',
            biogeo_mapping=None,  # 不再展示映射文本，但参数保留以便未来需要
            # —— 紧凑度 —— #
            marker_size=52,
            capsize=3,
            wspace=0.10,
            hspace=0.20,
            margins=(0.07, 0.995, 0.95, 0.12),  # (left,right,top,bottom)
            fig_w_base=2.6,
            fig_h_per_bio=0.34,
            # —— 文本与标注 —— #
            annotate_values=False,
            season_label_xoffset=0.015,  # ← 新增：季节名向左移（figure 坐标）
            season_label_yoffset=0.004,  # ← 新增：季节名向上移（与每行顶部对齐的微调）
            cbar_y_offset=0.09,  # ← 新增：颜色条相对 bottom 再下移量
            out_path='season_rows_biogeo_panel_MULTI.jpg',
            out_csv='season_rows_biogeo_slopes_MULTI.csv'
    ):
        import re, os, glob
        import numpy as np, pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.colors import TwoSlopeNorm

        plt.rcParams['font.family'] = 'Arial'

        n_sc = len(scenarios)
        assert n_sc >= 1
        n_bio = len(scenarios[0]['biogeo_dirs'])
        y_pos = np.arange(n_bio)
        y_labels_short = [f"R{i}" for i in range(1, n_bio + 1)]

        # 1) 全局斜率
        global_slopes = []
        for sc in scenarios:
            g_paths = glob.glob(os.path.join(sc['global_dir'], 'SHAP_summary_*.csv'))
            if not g_paths:
                raise FileNotFoundError(f"No SHAP_summary_*.csv in: {sc['global_dir']}")
            gdict = {}
            for s in seasons:
                y = _season_ratios_onlyseason(g_paths, s)
                sl, (lo, hi) = _ols_slope_ci(y)
                gdict[s] = dict(slope=sl, lo=lo, hi=hi)
            global_slopes.append(gdict)

        # 2) 各 biogeo 斜率 + 绝对偏差
        stats = []
        csv_rows = []
        for j, sc in enumerate(scenarios):
            for s in seasons:
                csv_rows.append(dict(
                    scenario=sc['name'], region='Global', biogeo='',
                    season=s,
                    slope=global_slopes[j][s]['slope'],
                    ci_low=global_slopes[j][s]['lo'],
                    ci_high=global_slopes[j][s]['hi'],
                    global_slope=global_slopes[j][s]['slope'],
                    delta_abs=np.nan
                ))

        for j, sc in enumerate(scenarios):
            per_season = {s: [] for s in seasons}
            for idx, bdir in enumerate(sc['biogeo_dirs'], start=1):
                paths = glob.glob(os.path.join(bdir, 'SHAP_summary_*.csv'))
                if not paths:
                    for s in seasons:
                        per_season[s].append(dict(idx=idx - 1, slope=np.nan, lo=np.nan, hi=np.nan,
                                                  delta_abs=np.nan, g=global_slopes[j][s]['slope']))
                    continue
                for s in seasons:
                    y = _season_ratios_onlyseason(paths, s)
                    sl, (lo, hi) = _ols_slope_ci(y)
                    g = global_slopes[j][s]['slope']
                    delta_abs = sl - g
                    per_season[s].append(dict(idx=idx - 1, slope=sl, lo=lo, hi=hi,
                                              delta_abs=delta_abs, g=g))
                    csv_rows.append(dict(
                        scenario=sc['name'], region=f"R{idx}", biogeo=f"R{idx}",
                        season=s, slope=sl, ci_low=lo, ci_high=hi,
                        global_slope=g, delta_abs=delta_abs
                    ))
            stats.append(per_season)

        # 3) 统一色标（绝对偏差对称 0）
        all_abs = [d['delta_abs'] for j in range(n_sc) for s in seasons for d in stats[j][s] if
                   np.isfinite(d['delta_abs'])]
        vmax = max(1e-12, float(np.nanmax(np.abs(all_abs)))) if all_abs else 1e-12
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap('RdBu_r')

        # 4) 统一 xlim
        xs = []
        for j in range(n_sc):
            for s in seasons:
                xs += [global_slopes[j][s]['lo'], global_slopes[j][s]['hi'], global_slopes[j][s]['slope']]
                for d in stats[j][s]:
                    xs += [d['lo'], d['hi']]
        xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
        pad = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
        xlim = (xmin - pad, xmax + pad)

        # 5) 紧凑画布
        fig_w = fig_w_base * n_sc + 2.0
        fig_h = max(1.6, fig_h_per_bio * n_bio) * len(seasons)
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(len(seasons), n_sc, figure=fig, hspace=hspace, wspace=wspace)

        axes = [[None] * n_sc for _ in range(len(seasons))]
        for r, s in enumerate(seasons):
            for c in range(n_sc):
                ax = fig.add_subplot(gs[r, c])
                axes[r][c] = ax
                ax.set_facecolor('#f5f5f5')
                # 去边框
                for sp in ax.spines.values():
                    sp.set_visible(False)

                # 每个子图左侧都显示 R 标签
                ax.set_yticks(y_pos)
                ax.set_yticklabels(y_labels_short, fontsize=8.5)
                ax.invert_yaxis()

                # 0 虚线 + 本场景本季节全局实线
                ax.axvline(0, color='black', linestyle='--', linewidth=1.2, zorder=1)
                g = global_slopes[c][s]['slope']
                if np.isfinite(g):
                    ax.axvline(g, color='dimgray', linestyle='-', linewidth=1.6, zorder=1)

                # 点=色条颜色；误差棒=黑色；可选标注
                for d in stats[c][s]:
                    if not np.isfinite(d['slope']):
                        continue
                    xerr = np.array([[d['slope'] - d['lo']], [d['hi'] - d['slope']]])
                    face = cmap(norm(d['delta_abs'])) if np.isfinite(d['delta_abs']) else 'lightgray'
                    ax.errorbar(
                        d['slope'], d['idx'], xerr=xerr,
                        fmt='o', ms=marker_size / 10,
                        markerfacecolor=face, markeredgecolor='black', markeredgewidth=0.6,
                        ecolor='black', elinewidth=1.0, capsize=capsize, zorder=3
                    )
                    if annotate_values:
                        ax.annotate(f"{d['slope']:.3f}, Δ{d['delta_abs']:+.3f}",
                                    (d['slope'], d['idx']),
                                    textcoords="offset points", xytext=(3, -2),
                                    ha='left', va='center', fontsize=7)

                ax.set_xlim(xlim)
                ax.tick_params(axis='x', labelsize=8)
                ax.set_xlabel('')

        # —— 标题/文本布局 —— #
        left, right, top, bottom = margins
        plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom - 0.02)
        fig.canvas.draw()

        # 1) 顶部每列：场景名（居中）
        for c in range(n_sc):
            top_ax = axes[0][c]
            bb = top_ax.get_position()
            x_center = (bb.x0 + bb.x1) / 2
            y_above = bb.y1 + 0.004
            fig.text(x_center, y_above, scenarios[c]['name'],
                     ha='center', va='bottom', fontsize=11)

        # 2) 左侧：每一行季节名（水平，顶边对齐，可由两个参数微调）
        for r in range(len(seasons)):
            left_ax = axes[r][0]
            bb = left_ax.get_position()
            y_top = bb.y1
            fig.text(left - season_label_xoffset, y_top + season_label_yoffset,
                     season_labels[r], ha='right', va='bottom', rotation=0, fontsize=11)

        # 3) 颜色条：底部下移，不与图重叠
        cbar_w = 0.30
        # cbar_left = (left + right - cbar_w) / 2
        cbar_left = left
        cbar_y = max(0.01, bottom - cbar_y_offset)  # 防止越界
        cax = fig.add_axes([cbar_left, cbar_y, cbar_w, 0.018])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm);
        sm.set_array([])
        cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cb.set_ticks([-vmax, 0, vmax])
        cb.set_ticklabels([f"{-vmax:.3f}", "0", f"{vmax:.3f}"])
        cb.ax.set_title('Regional sensitivity trend relative to the Europe-wide trend', fontsize=10, pad=1.0)

        fig.savefig(out_path, dpi=600)
        fig.savefig(out_path.replace('.jpg', '.pdf'), dpi=600)
        print('Saved figure:', out_path)

        df = pd.DataFrame(csv_rows, columns=[
            'scenario', 'region', 'biogeo', 'season', 'slope', 'ci_low', 'ci_high', 'global_slope', 'delta_abs'
        ])
        df.to_csv(out_csv, index=False)
        print('Saved CSV:', out_csv)
        return df

    _ = plot_season_rows_biogeo_panel_multi(
        scenarios=scenarios,
        dataset_label=indices,
        out_path=f'season_rows_biogeo_panel_MULTI_{indices}.jpg',
        out_csv=f'season_rows_biogeo_slopes_MULTI_{indices}.csv',
    )

def plot_biogeo_trmporal_trend_for_multiscenario_basedon_dry_wet_gradient(scenarios,indices):
    import os, glob, re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import TwoSlopeNorm

    # 复用：与单图一致，按窗口 CSV 计算季节序列
    def _season_ratios_onlyseason(paths, season):
        denom_cols = [f'tp_{s}_shap_value' for s in ('spring', 'summer', 'autumn', 'winter')]
        y = []
        for p in paths:  # 保持传入顺序
            df = pd.read_csv(p)
            denom = df[denom_cols].abs().sum(axis=1)
            df[f'tp_{season}_importance'] = df[f'tp_{season}_shap_value'].abs().div(denom)
            y.append(float(df[f'tp_{season}_importance'].mean()))
        return y

    from scipy.stats import linregress
    def _ols_slope_ci(y):
        y = np.asarray(y, dtype=float)
        x = np.arange(len(y), dtype=float)
        m = np.isfinite(y)
        if m.sum() < 2: return np.nan, (np.nan, np.nan)
        slope, _, _, _, se = linregress(x[m], y[m])
        return slope, (slope - 1.96 * se, slope + 1.96 * se)

    def plot_season_rows_biogeo_panel_multi(
            scenarios,
            seasons=('spring', 'summer', 'autumn', 'winter'),
            season_labels=('Spring', 'Summer', 'Autumn', 'Winter'),
            dataset_label='EVI',
            biogeo_mapping=None,  # 备用
            # —— 紧凑度 —— #
            marker_size=52,
            capsize=3,
            wspace=0.10,
            hspace=0.20,
            margins=(0.07, 0.995, 0.95, 0.12),  # (left,right,top,bottom)
            fig_w_base=2.6,
            fig_h_per_bio=0.34,
            # —— 文本与标注 —— #
            annotate_values=False,
            season_label_xoffset=0.015,
            season_label_yoffset=0.004,
            cbar_y_offset=0.09,
            out_path='season_rows_biogeo_panel_MULTI.jpg',
            out_csv='season_rows_biogeo_slopes_MULTI.csv',

            # === 新增：biogeo 行顺序控制 ===
            # 原始输入顺序（你的 biogeo_dirs 默认就是 2,3,4,5,7,8）
            base_bio_ids=(2, 3, 4, 5, 7, 8),
            # 按干→湿的显示顺序（R1..R6）
            dryness_bio_ids=(8,3,5,4,2,7),
            # （可选）长标签，方便导出到 CSV 或将来显示
            biogeo_long_names={
                2: "Pannonian",
                3: "Alpine",
                4: "Atlantic",
                5: "Continental",
                7: "Mediterranean",
                8: "Boreal",
            },
            # 在左侧标注一个 “Dry → Wet”
            show_dry_to_wet=False,
            dry_to_wet_text="Dry → Wet",
    ):
        import re, os, glob
        import numpy as np, pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.colors import TwoSlopeNorm
        from scipy.stats import linregress

        plt.rcParams['font.family'] = 'Arial'

        # === 内部小函数（保持不变） ===
        def _season_ratios_onlyseason(paths, season):
            denom_cols = [f'tp_{s}_shap_value' for s in ('spring', 'summer', 'autumn', 'winter')]
            y = []
            for p in paths:
                df = pd.read_csv(p)
                denom = df[denom_cols].abs().sum(axis=1)
                df[f'tp_{season}_importance'] = df[f'tp_{season}_shap_value'].abs().div(denom)
                y.append(float(df[f'tp_{season}_importance'].mean()))
            return y

        def _ols_slope_ci(y):
            y = np.asarray(y, dtype=float)
            x = np.arange(len(y), dtype=float)
            m = np.isfinite(y)
            if m.sum() < 2:
                return np.nan, (np.nan, np.nan)
            slope, _, _, _, se = linregress(x[m], y[m])
            return slope, (slope - 1.96 * se, slope + 1.96 * se)

        # === 预处理：构造“原始序号 → 干湿序号”的映射 ===
        # base_bio_ids 的位置就是 biogeo_dirs 的原始索引（0..5）
        # dryness_bio_ids 的位置（0..5）就是最终 R1..R6 的 y 轴位置
        n_sc = len(scenarios)
        n_bio = len(base_bio_ids)
        assert n_bio == len(dryness_bio_ids), "base_bio_ids 与 dryness_bio_ids 长度必须一致"
        idx_for_bid = {bid: i for i, bid in enumerate(base_bio_ids)}
        # 原始索引 → 干湿 y 位置
        drypos_by_origidx = {idx_for_bid[bid]: k for k, bid in enumerate(dryness_bio_ids)}
        # 干湿 y 位置的 tick 标签（R1..R6）
        y_labels_short = [f"R{i}" for i in range(1, n_bio + 1)]
        y_pos = np.arange(n_bio)

        # === 1) 全局斜率 ===
        global_slopes = []
        for sc in scenarios:
            g_paths = glob.glob(os.path.join(sc['global_dir'], 'SHAP_summary_*.csv'))
            if not g_paths:
                raise FileNotFoundError(f"No SHAP_summary_*.csv in: {sc['global_dir']}")
            gdict = {}
            for s in seasons:
                y = _season_ratios_onlyseason(g_paths, s)
                sl, (lo, hi) = _ols_slope_ci(y)
                gdict[s] = dict(slope=sl, lo=lo, hi=hi)
            global_slopes.append(gdict)

        # === 2) 各 biogeo 斜率（注意：idx 用干湿序号放到 y 轴） ===
        stats = []
        csv_rows = []
        # 先把全局写入 CSV
        for j, sc in enumerate(scenarios):
            for s in seasons:
                csv_rows.append(dict(
                    scenario=sc['name'], region='Global', biogeo='',
                    season=s,
                    slope=global_slopes[j][s]['slope'],
                    ci_low=global_slopes[j][s]['lo'],
                    ci_high=global_slopes[j][s]['hi'],
                    global_slope=global_slopes[j][s]['slope'],
                    delta_abs=np.nan
                ))

        for j, sc in enumerate(scenarios):
            per_season = {s: [] for s in seasons}
            for orig_idx, bdir in enumerate(sc['biogeo_dirs']):  # 原始顺序：与 base_bio_ids 对应
                # 该 biogeo 在干湿排序中的 y 轴位置
                y_idx = drypos_by_origidx.get(orig_idx, orig_idx)
                # 取对应的 bio id & 名称（用于 CSV）
                bio_id = base_bio_ids[orig_idx]
                bio_name = biogeo_long_names.get(bio_id, f"Bio{bio_id}")

                paths = glob.glob(os.path.join(bdir, 'SHAP_summary_*.csv'))
                if not paths:
                    for s in seasons:
                        g = global_slopes[j][s]['slope']
                        per_season[s].append(dict(idx=y_idx, slope=np.nan, lo=np.nan, hi=np.nan,
                                                  delta_abs=np.nan, g=g,
                                                  bio_id=bio_id, bio_name=bio_name))
                        # CSV 也输出 R 标签：Rk 由 y_idx 决定
                        csv_rows.append(dict(
                            scenario=sc['name'], region=f"R{y_idx + 1}", biogeo=bio_name,
                            season=s, slope=np.nan, ci_low=np.nan, ci_high=np.nan,
                            global_slope=g, delta_abs=np.nan
                        ))
                    continue

                for s in seasons:
                    yvals = _season_ratios_onlyseason(paths, s)
                    sl, (lo, hi) = _ols_slope_ci(yvals)
                    g = global_slopes[j][s]['slope']
                    delta_abs = sl - g
                    per_season[s].append(dict(idx=y_idx, slope=sl, lo=lo, hi=hi,
                                              delta_abs=delta_abs, g=g,
                                              bio_id=bio_id, bio_name=bio_name))
                    csv_rows.append(dict(
                        scenario=sc['name'], region=f"R{y_idx + 1}", biogeo=bio_name,
                        season=s, slope=sl, ci_low=lo, ci_high=hi,
                        global_slope=g, delta_abs=delta_abs
                    ))
            stats.append(per_season)

        # === 3) 色标（保持不变：对称 0） ===
        all_abs = [d['delta_abs'] for j in range(n_sc) for s in seasons for d in stats[j][s] if
                   np.isfinite(d['delta_abs'])]
        vmax = max(1e-12, float(np.nanmax(np.abs(all_abs)))) if all_abs else 1e-12
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cmap = plt.get_cmap('RdBu_r')

        # === 4) 统一 xlim ===
        xs = []
        for j in range(n_sc):
            for s in seasons:
                xs += [global_slopes[j][s]['lo'], global_slopes[j][s]['hi'], global_slopes[j][s]['slope']]
                for d in stats[j][s]:
                    xs += [d['lo'], d['hi']]
        xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
        pad = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
        xlim = (xmin - pad, xmax + pad)

        # === 5) 画图（y 轴按干湿序号） ===
        fig_w = fig_w_base * n_sc + 2.0
        fig_h = max(1.6, fig_h_per_bio * n_bio) * len(seasons)
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(len(seasons), n_sc, figure=fig, hspace=hspace, wspace=wspace)
        axes = [[None] * n_sc for _ in range(len(seasons))]

        for r, s in enumerate(seasons):
            for c in range(n_sc):
                ax = fig.add_subplot(gs[r, c])
                axes[r][c] = ax
                ax.set_facecolor('#f5f5f5')
                for sp in ax.spines.values():
                    sp.set_visible(False)

                ax.set_yticks(y_pos)
                ax.set_yticklabels(y_labels_short, fontsize=8.5)
                ax.invert_yaxis()

                # 0线 + 全局线
                ax.axvline(0, color='black', linestyle='--', linewidth=1.2, zorder=1)
                g = global_slopes[c][s]['slope']
                if np.isfinite(g):
                    ax.axvline(g, color='dimgray', linestyle='-', linewidth=1.6, zorder=1)

                # 按干湿 y 位置作点+误差棒
                for d in stats[c][s]:
                    if not np.isfinite(d['slope']):
                        continue
                    xerr = np.array([[d['slope'] - d['lo']], [d['hi'] - d['slope']]])
                    face = cmap(norm(d['delta_abs'])) if np.isfinite(d['delta_abs']) else 'lightgray'
                    ax.errorbar(
                        d['slope'], d['idx'], xerr=xerr,
                        fmt='o', ms=marker_size / 10,
                        markerfacecolor=face, markeredgecolor='black', markeredgewidth=0.6,
                        ecolor='black', elinewidth=1.0, capsize=capsize, zorder=3
                    )
                    if annotate_values:
                        ax.annotate(f"{d['slope']:.3f}, Δ{d['delta_abs']:+.3f}",
                                    (d['slope'], d['idx']),
                                    textcoords="offset points", xytext=(3, -2),
                                    ha='left', va='center', fontsize=7)

                ax.set_xlim(xlim)
                ax.tick_params(axis='x', labelsize=8)
                ax.set_xlabel('')

        # —— 标题/文本布局 —— #
        left, right, top, bottom = margins
        plt.subplots_adjust(left=left, right=right, top=top, bottom=bottom - 0.02)
        fig.canvas.draw()

        # 顶部列标题
        for c in range(n_sc):
            top_ax = axes[0][c]
            bb = top_ax.get_position()
            x_center = (bb.x0 + bb.x1) / 2
            y_above = bb.y1 + 0.004
            fig.text(x_center, y_above, scenarios[c]['name'],
                     ha='center', va='bottom', fontsize=11)

        # 左侧季节标签
        for r in range(len(seasons)):
            left_ax = axes[r][0]
            bb = left_ax.get_position()
            y_top = bb.y1
            fig.text(left - season_label_xoffset, y_top + season_label_yoffset,
                     season_labels[r], ha='right', va='bottom', rotation=0, fontsize=11)

        # 左侧标一条 “Dry → Wet”
        if show_dry_to_wet:
            first_ax = axes[0][0]
            last_ax = axes[-1][0]
            bb1, bb2 = first_ax.get_position(), last_ax.get_position()
            y_mid = (bb1.y0 + bb2.y1) / 2
            fig.text(left - season_label_xoffset - 0.02, y_mid, dry_to_wet_text,
                     ha='right', va='center', rotation=90, fontsize=10)

        # 颜色条（不重叠）
        cbar_w = 0.30
        cbar_left = left
        cbar_y = max(0.01, bottom - cbar_y_offset)
        cax = fig.add_axes([cbar_left, cbar_y, cbar_w, 0.018])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm);
        sm.set_array([])
        cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
        cb.set_ticks([-vmax, 0, vmax])
        cb.set_ticklabels([f"{-vmax:.3f}", "0", f"{vmax:.3f}"])
        cb.ax.set_title('Regional sensitivity trend relative to the Europe-wide trend', fontsize=10, pad=1.0)

        fig.savefig(out_path, dpi=600)
        fig.savefig(out_path.replace('.jpg', '.pdf'), dpi=600)
        print('Saved figure:', out_path)

        # 导出 CSV：region=R1..R6 已按干湿梯度
        df = pd.DataFrame(csv_rows, columns=[
            'scenario', 'region', 'biogeo', 'season', 'slope', 'ci_low', 'ci_high', 'global_slope', 'delta_abs'
        ])
        df.to_csv(out_csv, index=False)
        print('Saved CSV:', out_csv)
        return df

    _ = plot_season_rows_biogeo_panel_multi(
        scenarios=scenarios,
        dataset_label=indices,
        out_path=f'season_rows_biogeo_panel_MULTI_{indices}.jpg',
        out_csv=f'season_rows_biogeo_slopes_MULTI_{indices}.csv',
    )

def plot_seasonal_index_maps_with_profiles(
        folders_or_mapping,
        shp_path: str,
        *,
        index: str = "evi",
        season_order=("spring", "summer", "autumn", "winter"),
        row_order=("1. low_elevation_broadforest and mixed forest",
                   "2. high_elevation_broad",
                   "4. low_elevation_evergreen",
                   "3. high_elevation_everygreen"),
        out_path: str = "seasonal_panel.jpg",

        cmap: str = "YlOrRd",          # 离散化取色的底色带
        robust: bool = True,
        percentiles=(2, 98),

        profile_mode: str = "lat",
        hist_bins: int = 40,

        figsize=(24, 14),
        dpi: int = 600,
        # 右侧剖面基准宽度（与左图宽度的成对比例）
        map_to_profile_width_ratio=(4.5, 0.6),
        line_kwargs=None,
        cb_label: str = None,
        font_family: str = "Arial",

        # Jenks 参数
        jenks_k: int = 5,

        # 左侧行描述（两行显示）
        row_descs=(
            ("Broadleaved and mixed", "(low elevation)"),
            ("Broadleaved and mixed", "(high elevation)"),
            ("Coniferous", "(low elevation)"),
            ("Coniferous", "(high elevation)"),
        ),
        row_label_x_offset: float = -0.36,

        # ★ 新增：春/夏/秋剖面略宽的比例系数 + 指定应用季节
        profile_width_scale: float = 1.07,
        profile_wider_seasons=("spring", "summer", "autumn",),

        # ★ 新增：TIF 与剖面之间的列间距（越小越紧）
        col_wspace: float = 0.005,

        # ★ 新增：颜色条向下移动的相对距离（越大越靠下）
        cbar_vshift: float = 0.07,
):
    """
    渲染：自然断点（Jenks）离散分级；色带离散，under/over 为灰度。
    其他：
      - 地图定位用各自 transform，视窗统一到“第二行第一个 TIF”的范围；
      - 剖面更窄；季节标题置于顶部（仅第一行）；
      - 每列底部（仅最后一行）显示经度刻度；
      - 左侧场景文字放在纬度轴更左侧，横向两行；
      - 轴与画布背景为浅灰；
      - 色标只显示 Low | High；
      - ★ 春/夏/秋剖面可按比例略微放宽；★ 列间距可调；★ 颜色条可向下移动。
    """
    import os
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio.plot import plotting_extent
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # ---------- 样式 ----------
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.unicode_minus'] = False

    if line_kwargs is None:
        line_kwargs = dict(linewidth=2.0, color="black")

    index_l = index.lower()
    # if cb_label is None:
    #     cb_label = "Sensitivity"

    # ---------- 工具 ----------
    def _find_tif(folder: str, season: str, index_key: str):
        pats = []
        syn = {"spring": ["spring", "spr"], "summer": ["summer", "sum"],
               "autumn": ["autumn", "fall", "aut"], "winter": ["winter", "win"]}
        pats += syn.get(season.lower(), [season.lower()])
        idx_keys = [index_key.lower()]
        cand = []
        for fn in os.listdir(folder):
            low = fn.lower()
            if not (low.endswith(".tif") or low.endswith(".tiff")):
                continue
            if any(k in low for k in idx_keys) and any(p in low for p in pats):
                cand.append(os.path.join(folder, fn))
        if not cand:
            raise FileNotFoundError(f"在目录中未找到 {season} + {index_key} 的 TIF：{folder}")
        cand.sort(key=lambda p: len(os.path.basename(p)))
        return cand[0]

    def _normalize_io(folders_or_mapping):
        if not isinstance(folders_or_mapping, dict):
            raise ValueError("folders_or_mapping 必须是 dict。")
        is_explicit = any(isinstance(v, dict) for v in folders_or_mapping.values())
        mapping = {}
        if is_explicit:
            for row_label, d in folders_or_mapping.items():
                mapping[row_label] = {}
                for ss in season_order:
                    if ss not in d:
                        raise ValueError(f"显式映射缺少 {row_label} 的 {ss} 文件路径")
                    mapping[row_label][ss] = d[ss]
        else:
            for row_label, folder in folders_or_mapping.items():
                mapping[row_label] = {}
                for ss in season_order:
                    mapping[row_label][ss] = _find_tif(folder, ss, index_l)
        return mapping

    def _jenks_breaks(vals: np.ndarray, k: int):
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return np.linspace(0, 1, k + 1)
        lo = float(np.nanmin(vals)); hi = float(np.nanmax(vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return np.array([lo, hi], dtype=float)
        try:
            import mapclassify as mc
            nb = mc.NaturalBreaks(vals, k=k)
            bounds = np.concatenate(([lo], nb.bins))
        except Exception:
            bounds = np.quantile(vals, np.linspace(0, 1, k + 1))
        bounds = np.unique(bounds)
        if bounds.size < k + 1:
            bounds = np.linspace(lo, hi, k + 1)
        return bounds.astype(float)

    # ---------- 收集 TIF ----------
    mapping = _normalize_io(folders_or_mapping)

    # ---------- 读取 shapefile ----------
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("Shapefile 为空。")

    # ---------- 参考范围 ----------
    nrows = len(row_order)
    n_seasons = len(season_order)
    ref_row_idx = 1 if nrows >= 2 else 0
    ref_row_label = row_order[ref_row_idx]
    ref_season = season_order[0]
    ref_tif_path = mapping[ref_row_label][ref_season]
    with rasterio.open(ref_tif_path) as ref_src:
        ref_arr = ref_src.read(1, masked=True).filled(np.nan)
        ref_xmin, ref_xmax, ref_ymin, ref_ymax = plotting_extent(ref_arr, ref_src.transform)
        ref_crs = ref_src.crs

    if gdf.crs is None:
        raise ValueError("Shapefile 缺少 CRS（.prj）。请先为矢量设置正确 CRS。")
    if ref_crs is None:
        raise ValueError("参考 TIF 缺少 CRS。")
    gdf_plot = gdf.to_crs(ref_crs)

    # ---------- 布局 ----------
    # ★ 针对每个季节构造成对的宽度：map 宽度保持不变；需要时放大 profile 宽度
    wr = []
    for ss in season_order:
        w_map = map_to_profile_width_ratio[0]
        w_prof = map_to_profile_width_ratio[1] * (profile_width_scale if ss.lower() in set(s.lower() for s in profile_wider_seasons) else 1.0)
        wr += [w_map, w_prof]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # fig.patch.set_facecolor("#f2f2f2")

    gs = GridSpec(
        nrows=nrows, ncols=2 * n_seasons, figure=fig,
        width_ratios=wr, height_ratios=[1] * nrows,
        # ★ 列间距可调
        wspace=col_wspace, hspace=0.14
    )

    im_last = None
    last_bounds = None
    map_axes = []

    # ---------- 逐行逐季绘制 ----------
    for r, row_label in enumerate(row_order):
        first_map_ax_this_row = None

        for c, ss in enumerate(season_order):
            tif_path = mapping[row_label][ss]
            with rasterio.open(tif_path) as src:
                arr = src.read(1, masked=True).filled(np.nan)

                # —— 左：地图 ——（各自 extent，统一视窗）
                ax_map = fig.add_subplot(gs[r, 2 * c])
                ax_map.set_facecolor("#f2f2f2")
                vals = arr[np.isfinite(arr)]
                bounds = _jenks_breaks(vals, jenks_k) if vals.size else np.linspace(0, 1, jenks_k + 1)

                base = plt.get_cmap(cmap)
                colors = [base(i / max(1, jenks_k - 1)) for i in range(jenks_k)]
                cmap_disc = ListedColormap(colors)
                try:
                    cmap_disc = cmap_disc.copy()
                except Exception:
                    pass
                cmap_disc.set_under("#DDDDDD")
                cmap_disc.set_over("#555555")
                norm_disc = BoundaryNorm(bounds, ncolors=cmap_disc.N, clip=False)

                xmin, xmax, ymin, ymax = plotting_extent(arr, src.transform)
                im_last = ax_map.imshow(
                    arr,
                    extent=[xmin, xmax, ymin, ymax],
                    origin='upper',
                    cmap=cmap_disc, norm=norm_disc,
                    interpolation='nearest', zorder=1
                )
                last_bounds = bounds
                ax_map.set_xlim(ref_xmin, ref_xmax)
                ax_map.set_ylim(ref_ymin, ref_ymax)

                gdf_plot.boundary.plot(ax=ax_map, color='k', linewidth=0.6, zorder=3)

                if c == 0:
                    ax_map.set_yticks(np.linspace(ref_ymin, ref_ymax, 4))
                    ax_map.set_ylabel("Latitude (°)")
                else:
                    ax_map.set_yticks([])

                if r == nrows - 1:
                    ax_map.set_xticks(np.linspace(ref_xmin, ref_xmax, 4))
                    ax_map.set_xlabel("Longitude (°)")
                else:
                    ax_map.set_xticks([])

                ax_map.set_aspect('equal')
                map_axes.append(ax_map)

                if r == 0:
                    ax_map.set_title(f"{ss.capitalize()} ({index_l.upper()})", pad=6, fontsize=12)

                if first_map_ax_this_row is None:
                    first_map_ax_this_row = ax_map

                # —— 右：剖面 ——（等高、统一范围）
                ax_prof = fig.add_subplot(gs[r, 2 * c + 1])

                if profile_mode.lower() == "lat":
                    finite = np.isfinite(arr)
                    means = np.where(finite.any(axis=1), np.nanmean(arr, axis=1), np.nan)
                    stds  = np.where(finite.any(axis=1), np.nanstd(arr, axis=1, ddof=1), 0.0)
                    ycoords = np.linspace(ref_ymax, ref_ymin, arr.shape[0])
                    m, s = means, np.nan_to_num(stds, nan=0.0, posinf=0.0, neginf=0.0)
                    ax_prof.fill_betweenx(ycoords, m - s, m + s, alpha=0.24, edgecolor="none")
                    ax_prof.plot(m, ycoords, **line_kwargs)
                    ax_prof.set_ylim(ref_ymin, ref_ymax)
                    ax_prof.set_yticks([])
                    ax_prof.set_ylabel("")
                    ax_prof.set_xlabel("Zonal mean (±1σ)" if r == nrows - 1 else "")
                    ax_prof.grid(alpha=0.25, linestyle='--', linewidth=0.6)

                elif profile_mode.lower() == "lon":
                    finite = np.isfinite(arr)
                    means = np.where(finite.any(axis=0), np.nanmean(arr, axis=0), np.nan)
                    stds  = np.where(finite.any(axis=0), np.nanstd(arr, axis=0, ddof=1), 0.0)
                    xcoords = np.linspace(ref_xmin, ref_xmax, arr.shape[1])
                    m, s = means, np.nan_to_num(stds, nan=0.0, posinf=0.0, neginf=0.0)
                    ax_prof.fill_between(xcoords, m - s, m + s, alpha=0.24, edgecolor="none")
                    ax_prof.plot(xcoords, m, **line_kwargs)
                    ax_prof.set_xlim(ref_xmin, ref_xmax)
                    ax_prof.set_xlabel("")
                    ax_prof.set_ylabel("Meridional mean (±1σ)" if r == nrows - 1 else "")
                    ax_prof.grid(alpha=0.25, linestyle='--', linewidth=0.6)

                elif profile_mode.lower() == "hist":
                    vals_h = arr[np.isfinite(arr)]
                    if vals_h.size:
                        ax_prof.hist(vals_h, bins=hist_bins, density=True, orientation='horizontal')
                    ax_prof.set_yticks([])
                    ax_prof.set_xlabel("Density" if r == nrows - 1 else "")
                    ax_prof.grid(alpha=0.2, linestyle='--', linewidth=0.5)
                else:
                    raise ValueError("profile_mode 仅支持 'lat' | 'lon' | 'hist'")

        # 左侧场景文字（两行）放在更左侧
        if first_map_ax_this_row is not None:
            if isinstance(row_descs, (list, tuple)) and len(row_descs) >= nrows:
                line1, line2 = row_descs[r]
                txt = f"{line1}\n{line2}"
            else:
                txt = row_order[r].replace("(", "\n(")
            first_map_ax_this_row.text(
                row_label_x_offset, 0.5, txt,
                va='center', ha='right', fontsize=16, rotation=0,
                transform=first_map_ax_this_row.transAxes
            )

    # ---------- 颜色条（离散，Low|High） ----------
    if im_last is not None and last_bounds is not None and len(map_axes) > 0:
        grid_left = min(ax.get_position().x0 for ax in map_axes)
        grid_right = max(ax.get_position().x1 for ax in map_axes)
        grid_bottom = min(ax.get_position().y0 for ax in map_axes)

        total_w = (grid_right - grid_left)
        cbar_w = 0.18 * total_w
        cbar_h = 0.012
        cbar_left = grid_right - cbar_w
        # ★ 下移：增大向下偏移量
        cbar_bottom = max(0.01, grid_bottom - float(cbar_vshift))

        cax = fig.add_axes([cbar_left, cbar_bottom, cbar_w, cbar_h])
        cb = fig.colorbar(im_last, cax=cax, orientation='horizontal',
                          boundaries=last_bounds, spacing="proportional")
        cb.set_ticks([last_bounds[0], last_bounds[-1]])
        cb.set_ticklabels(["Lower sensitivity", "Higher sensitivity"])
        # cb.set_label(cb_label, labelpad=2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=13)

    # ---------- 导出 ----------
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    import os
    root, ext = os.path.splitext(out_path)
    fig.savefig(root + ".pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存：{out_path} 和 {root + '.pdf'}")


import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

def plot_tm_vpd_shap_effects_split(
    indices: str,
    scenario_indices: str,
    seasons,
    scenarios,
    csv_name: str = r"SHAP_summary_Temporal Series.csv",
    *,
    out_path_tm: str = "TM_shap_effects.jpg",
    out_path_vpd: str = "VPD_shap_effects.jpg",
    figsize=(28, 20),         # 字体放大后，稍微加大版面
    dpi: int = 300,
    scatter_size: float = 6.0,
    line_width: float = 2.3,
    ci_alpha: float = 0.22,   # 置信区间透明度
    base_fontsize: int = 12,  # 全局字体
):
    """
    生成两张图：
      1) TM：4行×4列（行=森林组，列=季节）
      2) VPD：4行×4列（行=森林组，列=季节）

    每个子图：x = {var}_{season}_baseline，y = {var}_{season}_shap_value
    绘制浅灰散点 + 线性拟合（图例含公式与R²）+ 95% 置信区间。
    x 轴标签仅底行显示；y 轴标签仅左列显示；不使用整图标题。
    """

    # 放大全局字体
    plt.rcParams.update({
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 2,
        "axes.labelsize": base_fontsize + 1,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
    })

    def _season_key(season_label: str) -> str:
        s = season_label.strip()
        if ". " in s:
            s = s.split(". ", 1)[1]
        return s.strip().lower()

    def _find_csv_for(scn_dir: str, season_dirname: str) -> Optional[str]:
        """优先返回路径包含 images_{scenario_indices} 的 csv；否则回退任意匹配。"""
        season_dir = os.path.join(scn_dir, season_dirname)
        if not os.path.isdir(season_dir):
            return None
        candidates = glob.glob(os.path.join(season_dir, "**", csv_name), recursive=True)
        if not candidates:
            return None
        prefer_tokens = [
            "images_" + scenario_indices.lower(),
            "images_" + indices.lower(),
        ]
        norm = lambda p: os.path.normpath(p).replace("\\", "/").lower()
        for token in prefer_tokens:
            for p in candidates:
                if token in norm(p):
                    return p
        return candidates[0]

    def _linear_fit_with_ci(x: np.ndarray, y: np.ndarray):
        """
        线性回归 + 95% 置信区间（对回归线的均值预测带，正态近似 z=1.96）。
        返回 slope, intercept, r2, xfit, yfit, (yfit_lo, yfit_hi)
        """
        msk = np.isfinite(x) & np.isfinite(y)
        xv, yv = x[msk], y[msk]
        n = len(xv)
        if n < 3:
            return np.nan, np.nan, np.nan, None, None, (None, None)

        # 拟合
        slope, intercept = np.polyfit(xv, yv, 1)
        y_pred = slope * xv + intercept
        ss_res = np.sum((yv - y_pred) ** 2)
        ss_tot = np.sum((yv - np.mean(yv)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # 置信区间（均值预测带）
        xbar = np.mean(xv)
        sxx = np.sum((xv - xbar) ** 2)
        sigma2 = ss_res / (n - 2) if n > 2 else np.nan

        xfit = np.linspace(np.nanmin(xv), np.nanmax(xv), 80)
        yfit = slope * xfit + intercept

        if not np.isfinite(sigma2) or sxx <= 0:
            return slope, intercept, r2, xfit, yfit, (None, None)

        z = 1.96  # 正态近似
        se_mean = np.sqrt(sigma2 * (1.0 / n + (xfit - xbar) ** 2 / sxx))
        yfit_lo = yfit - z * se_mean
        yfit_hi = yfit + z * se_mean
        return slope, intercept, r2, xfit, yfit, (yfit_lo, yfit_hi)

    def _plot_one_variable(var_prefix: str, x_full_label: str, out_path: str):
        """
        var_prefix: 'tm' or 'vpd'
        x_full_label: 'Air temperature' or 'Vapor pressure deficit'
        out_path: 输出路径
        """
        n_rows = len(scenarios)
        n_cols = len(seasons)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, squeeze=False)
        plt.subplots_adjust(wspace=0.18, hspace=0.28)

        # 顶部列标题：按季节
        for c, s_lab in enumerate(seasons):
            axes[0, c].set_title(s_lab, pad=8)

        for r, scn in enumerate(scenarios):
            scn_name = scn.get("name", f"Scenario {r+1}")
            scn_dir = scn["global_dir"]

            for c, s_lab in enumerate(seasons):
                ax = axes[r, c]
                sk = _season_key(s_lab)
                csv_path = _find_csv_for(scn_dir, s_lab)

                if not csv_path or not os.path.exists(csv_path):
                    ax.text(0.5, 0.5, "CSV not found", ha="center", va="center", fontsize=base_fontsize-2)
                    ax.set_axis_off()
                    continue

                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    ax.text(0.5, 0.5, f"Failed to read CSV\n{os.path.basename(csv_path)}",
                            ha="center", va="center", fontsize=base_fontsize-2)
                    ax.set_axis_off()
                    continue

                xcol = f"{var_prefix}_{sk}_baseline"
                ycol = f"{var_prefix}_{sk}_shap_value"
                if xcol not in df.columns or ycol not in df.columns:
                    ax.text(0.5, 0.5, f"Missing columns:\n{xcol}\n{ycol}",
                            ha="center", va="center", fontsize=base_fontsize-2)
                    ax.set_axis_off()
                    continue

                x = df[xcol].to_numpy()
                y = df[ycol].to_numpy()

                # 观测点（浅灰）
                ax.scatter(x, y, s=scatter_size, c="0.75", alpha=0.6,
                           edgecolors="none", rasterized=True)

                # 拟合 + 置信区间
                slope, intercept, r2, xfit, yfit, (ylo, yhi) = _linear_fit_with_ci(x, y)
                if xfit is not None:
                    # 置信区间
                    if ylo is not None and yhi is not None:
                        ax.fill_between(xfit, ylo, yhi, alpha=ci_alpha, linewidth=0)

                    # 拟合线
                    ax.plot(xfit, yfit, lw=line_width,
                            label=f"y = {slope:.3g}x + {intercept:.3g};  R²={r2:.2f}")

                # 网格与图例
                ax.grid(True, linestyle=":", linewidth=0.9, alpha=0.65)
                if xfit is not None:
                    ax.legend(loc="best", frameon=False)

                # 轴标签：x 仅底行，y 仅左列
                if r == n_rows - 1:
                    ax.set_xlabel(x_full_label)
                else:
                    ax.set_xlabel("")

                if c == 0:
                    ax.set_ylabel("SHAP values")
                else:
                    ax.set_ylabel("")

                # 最左列外侧标注场景名（竖排）
                if c == 0:
                    ax.text(-0.33, 0.5, scn_name,
                            transform=ax.transAxes, rotation=90,
                            va="center", ha="right",
                            fontsize=15, fontweight="bold")

                # 刻度适度加粗
                ax.tick_params(axis="both", which="both", width=1.1, length=5)

        # 不加整图标题（按你的要求）
        fig.savefig(out_path, bbox_inches="tight")
        fig.savefig(out_path.replace('.jpg','.pdf'), bbox_inches="tight")
        return fig

    # === 分两张图输出 ===
    fig_tm  = _plot_one_variable("tm",  "Temperature",         out_path_tm)
    fig_vpd = _plot_one_variable("vpd", "Vapor pressure deficit",  out_path_vpd)
    return fig_tm, fig_vpd


if __name__ == '__main__':

    '''# 干旱区域物候指标提取'''
    legacy = legacy_effects()

    '''生成empty base'''
    # other_scenario = ''
    # country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
    #                 os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    # mask_tifs = []
    # for i,country_tif in tqdm(enumerate(country_tifs)):
    #     country_name = os.path.split(country_tif)[-1]
    #     if os.path.exists(os.path.join(country_tif,f'Wu Yong/{other_scenario}/valid_data_mask_ratio_all.tif')):
    #         data_path = os.path.join(country_tif,f'Wu Yong/{other_scenario}/valid_data_mask_ratio_all.tif')
    #         empty_base_tif = os.path.join(country_tif,f'Wu Yong/{other_scenario}/empty_base_all.tif')
    #         if not os.path.exists(empty_base_tif): generate_empty_base(data_path,empty_base_tif)
    #     else:
    #         reference_tif_paths = glob.glob(os.path.join(country_tif, f"Wu Yong/{other_scenario}/valid_data_mask*.tif"))
    #         reference_tif_paths = [item for item in reference_tif_paths if 'number' not in item and 'ratio' not in item]
    #         if len(reference_tif_paths) == 0:continue
    #         for path in reference_tif_paths:
    #             chunk_i,chunk_j = re.search(r"valid_data_mask_(\d+)_(\d+)", path).groups()
    #             data_path = os.path.join(country_tif, f'Wu Yong/{other_scenario}/valid_data_mask_ratio_{chunk_i}_{chunk_j}.tif')
    #             empty_base_tif = os.path.join(country_tif, f'Wu Yong/{other_scenario}/empty_base_{chunk_i}_{chunk_j}.tif')
    #             if not os.path.exists(empty_base_tif): generate_empty_base(data_path, empty_base_tif)

    '''# # 可视化weights'''
    # other_scenario = ''
    # country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
    #                 os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    # mask_tifs = []
    # for i,country_tif in tqdm(enumerate(country_tifs)):
    #     country_name = os.path.split(country_tif)[-1]
    #     if os.path.exists(os.path.join(country_tif,f'Wu Yong/{other_scenario}/valid_data_mask_ratio_all.tif')):
    #         data_path = os.path.join(country_tif,f'Wu Yong/{other_scenario}/valid_data_mask_ratio_all.tif')
    #         mask_tifs.append(data_path)
    #     else:
    #         reference_tif_paths = glob.glob(os.path.join(country_tif, f"Wu Yong/{other_scenario}/valid_data_mask*.tif"))
    #         reference_tif_paths = [item for item in reference_tif_paths if 'number' not in item and 'ratio' not in item]
    #         if len(reference_tif_paths) == 0:continue
    #         for path in reference_tif_paths:
    #             chunk_i,chunk_j = re.search(r"valid_data_mask_(\d+)_(\d+)", path).groups()
    #             data_path = os.path.join(country_tif, f'Wu Yong/{other_scenario}/valid_data_mask_ratio_{chunk_i}_{chunk_j}.tif')
    #             mask_tifs.append(data_path)
    #
    # legacy.merge_tifs(mask_tifs,f'{other_scenario} mask_ratio_merge.tif')

    '''随机森林结果绘制（时序）summary图或者其他图'''
    # indices_name = 'EVI'
    # data_path = f'时序建模/images {indices_name}/SHAP_summary_Temporal Series_new.csv'
    # legacy.random_forest_summary(data_path)
    '''1. EVI,SIF空间时序可视化'''
    # 生成EVI SIF sa数据
    # start_year = 2001
    # end_year = 2024
    # indices_name = r'EVI'
    # data_path = r'summary.csv'
    # legacy.indices_spatial_visualization(data_path,start_year,end_year,indices_name)

    # 合并时空可视化
    # paths_evi = glob.glob(os.path.join(r'after_first_revision\6. offset\temp\sif\merge', '*.tif'))
    # plot_indices_timeseries_with_spatial_panels_with_drought_vertical_line(
    #     tif_paths=paths_evi,
    #     output_path="spatial_temporal.jpg",
    #     shp_path="after_first_revision/data/European_no_belarus_ukrain.shp",
    #     cmap='RdYlGn',
    #     ncols=5,
    #     robust=True,
    #     percentiles=(2, 98),
    #     sample_per_image=200000,
    #     figsize=(16, 12),
    #
    #     # 上：地图每行高度；下：时序高度
    #     panel_height_ratio=(0.8, 2.2),
    #     # 颜色条外观与位置
    #     cb_label="",
    #     cbar_thickness_frac=0.35,  # 色条厚度（相对所在单元的高度）
    #     cbar_xshift_frac=0.05,  # 色条整体向右平移（相对所在单元的宽度）
    #     # 让时序轴整体右移一点，把 y 轴刻度+轴名“收进去”
    #     ts_left_shift_frac=0.08,  # 相对整张图宽度的右移比例（可微调 0.03~0.05）,
    #     index_name='SIF'
    # )
    # 可视化三个季节相对贡献的箱型图
    # path_evi = r'时序建模/images EVI/SHAP_summary_Temporal Series_new.csv'
    # path_sif = r'时序建模/images SIF/SHAP_summary_Temporal Series_new.csv'
    # output_path = r'时序建模/boxplot.jpg'
    # precipitation_item = [r'tp_spring_relative_shap_ratio_new','tp_summer_relative_shap_ratio_new','tp_autumn_relative_shap_ratio_new','tp_winter_relative_shap_ratio_new']
    # legacy.seasonal_precipitataion_shaprelatively_boxplot([path_evi,path_sif],precipitation_item,output_path)
    '''# 随机森林结果可视化'''

    '''2. 绘制季节性降水敏感性图，EVI,SIF，全局，分biogeo'''
    # data_root_path = r'D://WuYong//after_first_revision/1. importance of seasonal precipitation (with biogeo)/spatially_block_kfold'
    # scenarios = os.listdir(data_root_path)
    # scenarios = sorted(scenarios)
    # indices = ['evi','sif']
    # weights_paths = {}
    # indices_data_paths = {}
    # for scenario in scenarios:
    #     if 'low_elevation_broadforest' in scenario: weights_paths[scenario] = r'after_first_revision/2 weights/mask_ratio_merge.tif'
    #     elif 'high_elevation_broad' in scenario:weights_paths[scenario] = r'after_first_revision/2 weights/high_elevation_broad mask_ratio_merge.tif'
    #     elif 'low_elevation_evergreen' in scenario:weights_paths[scenario] = r'after_first_revision/2 weights/low_elevation_everygreen mask_ratio_merge.tif'
    #     elif 'high_elevation_everygreen' in scenario:weights_paths[scenario] = r'after_first_revision/2 weights/high_elevation_everygreen mask_ratio_merge.tif'
    #     scenario_path = os.path.join(data_root_path,scenario)
    #     scenario_indices = os.listdir(scenario_path)
    #     scenario_evi_data_path = os.path.join(scenario_path,[item for item in scenario_indices if 'evi' in item][0])
    #     scenario_sif_data_path = os.path.join(scenario_path, [item for item in scenario_indices if 'sif' in item][0])
    #     scenario_evi_data_path = os.path.join(scenario_evi_data_path,'SHAP_summary_Temporal Series.csv')
    #     scenario_sif_data_path = os.path.join(scenario_sif_data_path, 'SHAP_summary_Temporal Series.csv')
    #     indices_data_paths[scenario] = {'evi':scenario_evi_data_path,'sif':scenario_sif_data_path}
    #
    # shp_path = "after_first_revision/data/European_no_belarus_ukrain.shp"
    # biogeo_mapping = {
    #     1:'Black Sea Bio-geographical Region',2:'Pannonian Bio-geographical Region',3:'Alpine Bio-geographical Region',
    #     4:'Atlantic Bio-geographical Region',5:'Continental Bio-geographical Region',6:'Macaronesian Bio-geographical Region',
    #     7:'Mediterranean Bio-geographical Region',8:'Boreal Bio-geographical Region',9:'Steppic Bio-geographical Region',10:'Arctic Bio-geographical Region'
    # }
    #
    # plot_sensitivity_panel(weights_paths, indices_data_paths, shp_path,
    #                           out_path="sensitivity_panel_final.jpg",
    #                           show=False)

    # 测试用
    # path = os.path.join(r'D://SHAP_summary_Temporal Series.csv')
    # data = pd.read_csv(path)
    # data_abs = data[['tp_spring_shap_value', 'tp_summer_shap_value', 'tp_autumn_shap_value',
    #    'tp_winter_shap_value']].abs()
    # sum_value = data_abs.sum(axis=1)
    # data['tp_spring_shap_value_abs'] = data_abs['tp_spring_shap_value']
    # data['tp_summer_shap_value_abs'] = data_abs['tp_summer_shap_value']
    # data['tp_autumn_shap_value_abs'] = data_abs['tp_autumn_shap_value']
    # data['tp_winter_shap_value_abs'] = data_abs['tp_winter_shap_value']
    # data['sum'] = sum_value
    # print(data['tp_spring_shap_value_abs'].div(sum_value).mean())
    # print(data['tp_summer_shap_value_abs'].div(sum_value).mean())
    # print(data['tp_autumn_shap_value_abs'].div(sum_value).mean())
    # print(data['tp_winter_shap_value_abs'].div(sum_value).mean())
    # test = data[data['biogeo'] == 8]
    # print(test['tp_spring_shap_value_abs'].div(sum_value).mean() - data['tp_spring_shap_value_abs'].div(sum_value).mean())
    # print((data[data['biogeo'] == 2]['tp_spring_shap_value_abs']/data[data['biogeo'] == 2]['sum']).mean() - (data['tp_spring_shap_value_abs']/sum_value).mean())
    '''# 重新计算特征相对重要性'''
    # 序列数据
    # path = f'时序建模/images EVI/SHAP_summary_Temporal Series.csv'
    # data = pd.read_csv(path)
    # shap_columns = ['tp_spring_shap_value', 'tp_summer_shap_value', 'tp_autumn_shap_value',
    #                 'vpd_annual_shap_value', 'tm_annual_shap_value', 'spei_annual_shap_value']
    # total_shap_value = data[shap_columns].abs().sum(axis=1)
    # data['tp_spring_relative_shap_ratio_new'] = data['tp_spring_shap_value'].abs() / total_shap_value
    # data['tp_summer_relative_shap_ratio_new'] = data['tp_summer_shap_value'].abs() / total_shap_value
    # data['tp_autumn_relative_shap_ratio_new'] = data['tp_autumn_shap_value'].abs() / total_shap_value
    # data['vpd_annual_relative_shap_ratio_new'] = data['vpd_annual_shap_value'].abs() / total_shap_value
    # data['tm_annual_relative_shap_ratio_new'] = data['tm_annual_shap_value'].abs() / total_shap_value
    # data['spei_annual_relative_shap_ratio_new'] = data['spei_annual_shap_value'].abs() / total_shap_value
    # data.to_csv(path.replace('.csv','_new.csv'))
    # 窗口数据
    # paths = glob.glob(os.path.join(r'10年窗口/images window EVI/', 'SHAP_summary_*.csv'))
    # for path in paths:
    #     data = pd.read_csv(path)
    #     shap_columns = ['tp_spring_shap_value',
    #                     'vpd_annual_shap_value', 'tm_annual_shap_value', 'spei_annual_shap_value']
    #     total_shap_value = data[shap_columns].abs().sum(axis=1)
    #     data['tp_spring_relative_shap_ratio_new'] = data['tp_spring_shap_value'].abs() / total_shap_value
    #     data['vpd_annual_relative_shap_ratio_new'] = data['vpd_annual_shap_value'].abs() / total_shap_value
    #     data['tm_annual_relative_shap_ratio_new'] = data['tm_annual_shap_value'].abs() / total_shap_value
    #     data['spei_annual_relative_shap_ratio_new'] = data['spei_annual_shap_value'].abs() / total_shap_value
    #     data.to_csv(path.replace('.csv', '_new.csv'))

    '''# 基于窗口的时间趋势变化'''
    # indices_name = 'evi'
    # seasons = ['spring','summer','autumn','winter']
    # data_paths = glob.glob(os.path.join(r'D:\WuYong/after_first_revision\4. Temporal trend\2. 5 years windows\1. low_elevation_broadforest and mixed forest\biogeo\biogeo_8_temporal_trend\images_evi_lowelevation_broad','SHAP_summary_*.csv'))
    # for season in seasons:
    #     # season = r'winter'
    #     start_year = 2001
    #     end_year = 2023
    #     dataset = 'evi'
    #     legacy.temporal_window_analysis_onlyseason(data_paths,season,start_year,end_year,dataset)
    #
    '''# 分biogeo'''
    # biogeos = [2,3,4,5,7,8]
    # for biogeo in biogeos:
    #     indices_name = 'sif'
    #     seasons = ['spring','summer','autumn','winter']
    #     data_paths = glob.glob(os.path.join(f'D:\WuYong/after_first_revision/4. Temporal trend/2. 5 years windows/1. low_elevation_broadforest and mixed forest/biogeo/biogeo_{biogeo}_temporal_trend\images_{indices_name}_lowelevation_broad','SHAP_summary_*.csv'))
    #     for season in seasons:
    #         # season = r'winter'
    #         start_year = 2001
    #         end_year = 2023
    #         dataset = indices_name
    #         legacy.temporal_window_analysis_onlyseason(data_paths,season,start_year,end_year,dataset)

    '''窗口时序趋势变化和SPEI异常关系图'''
    # seasons = ['spring','summer','autumn','winter']
    # data_paths = glob.glob(os.path.join(r'after_first_revision\4. Temporal trend\2. 5 years windows\3. high_elevation_everygreen\images_evi_high_elevation_everygreen','SHAP_summary_*.csv'))
    # for season in seasons:
    # # season = r'summer'
    # #     if season =='winter':
    # #         print('test')
    #     climate_name = 'tm'
    #     start_year = 2001
    #     end_year = 2023
    #     dataset = 'EVI'
    #     summary_path = r'afterfirst_revision_summary_high_elevation_everygreen.csv'
    #     legacy.temporal_window_analysis_with_SPEI_anomaly(data_paths,climate_name,season,start_year,end_year,dataset,summary_path)

    '''同时绘制趋势和异常关系图'''
    # plot_trend_anomaly()


    '''biogeo和全局时序趋势的对比图'''
    # 单一场景
    # biogeo_mapping = {
    #     1: 'Black Sea Bio-geographical Region', 2: 'Pannonian Bio-geographical Region',
    #     3: 'Alpine Bio-geographical Region', 4: 'Atlantic Bio-geographical Region',
    #     5: 'Continental Bio-geographical Region', 6: 'Macaronesian Bio-geographical Region',
    #     7: 'Mediterranean Bio-geographical Region', 8: 'Boreal Bio-geographical Region',
    #     9: 'Steppic Bio-geographical Region', 10: 'Arctic Bio-geographical Region'
    # }
    #
    # global_dir = r'D:\WuYong\after_first_revision\4. Temporal trend\2. 5 years windows\1. low_elevation_broadforest and mixed forest\images_evi_lowelevation_broad'
    # biogeo_ids = [2, 3, 4, 5, 7, 8]
    # biogeo_dirs = [
    #     fr'D:\WuYong\after_first_revision\4. Temporal trend\2. 5 years windows\1. low_elevation_broadforest and mixed forest\biogeo\biogeo_{bid}_temporal_trend\images_evi_lowelevation_broad'
    #     for bid in biogeo_ids
    # ]
    # plot_biogeo_trmporal_trend_for_single(global_dir,biogeo_dirs,biogeo_mapping)
    #
    # 多场景
    # indices = 'EVI'
    # scenario_indices = 'evi'
    # year_windows = '3. 10'
    # scenarios = [
    #     {
    #         "name": "Broadleaved and mixed (low elevation)",
    #         "global_dir": fr"D:\WuYong\after_first_revision\4. Temporal trend\{year_windows} years windows\1. low_elevation_broadforest and mixed forest\images_{scenario_indices}_lowelevation_broad",
    #         "biogeo_dirs": [
    #             fr"D:\WuYong\after_first_revision\4. Temporal trend\{year_windows} years windows\1. low_elevation_broadforest and mixed forest\biogeo\biogeo_{bid}_temporal_trend\images_{scenario_indices}_lowelevation_broad"
    #             for bid in [2, 3, 4, 5, 7, 8]
    #         ],
    #     },
    #     {
    #         "name": "Broadleaved and mixed (high elevation)",
    #         "global_dir": fr"D:\WuYong\after_first_revision\4. Temporal trend\{year_windows} years windows\2. high_elevation_broad\images_{scenario_indices}_high_elevation_broad",
    #         "biogeo_dirs": [
    #             fr"D:\WuYong\after_first_revision\4. Temporal trend\{year_windows} years windows\2. high_elevation_broad\biogeo\biogeo_{bid}_temporal_trend\images_{scenario_indices}_high_elevation_broad"
    #             for bid in [2, 3, 4, 5, 7, 8]
    #         ],
    #     },
    #     {
    #         "name": "Coniferous (low elevation)",
    #         "global_dir": fr"D:\WuYong\after_first_revision\4. Temporal trend\{year_windows} years windows\4. low_elevation_evergreen\images_{scenario_indices}_low_elevation_everygreen",
    #         "biogeo_dirs": [
    #             fr"D:\WuYong\after_first_revision\4. Temporal trend\{year_windows} years windows\4. low_elevation_evergreen\biogeo\biogeo_{bid}_temporal_trend\images_{scenario_indices}_low_elevation_everygreen"
    #             for bid in [2, 3, 4, 5, 7, 8]
    #         ],
    #     },
    #     {
    #         "name": "Coniferous (high elevation)",
    #         "global_dir": fr"D:\WuYong\after_first_revision\4. Temporal trend\{year_windows} years windows\3. high_elevation_everygreen\images_{scenario_indices}_high_elevation_everygreen",
    #         "biogeo_dirs": [
    #             fr"D:\WuYong\after_first_revision\4. Temporal trend\{year_windows} years windows\3. high_elevation_everygreen\biogeo\biogeo_{bid}_temporal_trend\images_{scenario_indices}_high_elevation_everygreen"
    #             for bid in [2, 3, 4, 5, 7, 8]
    #         ],
    #     },
    #
    # ]
    #
    # plot_biogeo_trmporal_trend_for_multiscenario_basedon_dry_wet_gradient(scenarios,indices)
    '''基于时序建模的交互作用分析'''
    '''全局'''
    # path = r'D:\WuYong\after_first_revision\3 interactions'
    # features = [f'tp_spring_baseline',
    #         f'tp_summer_baseline',
    #         f'tp_autumn_baseline',
    #         f'tp_winter_baseline'
    #         f'vpd',
    #         f'tm',
    #         # f'spei'
    #             ]
    # seasons = ['spring','summer','autumn','winter']
    # scenarios = ['1.','2.','3.','4.']
    # scenarios_path = os.listdir(path)
    # for scenario_path in scenarios_path:
    #     paths_evi, paths_sif = {}, {}
    #     if scenario_path[0:2] not in scenarios: continue
    #     for i,season in enumerate(seasons):
    #         season_path = f'{path}/{scenario_path}/{str(i+1)}. {season}'
    #         season_evi_path = f'{season_path}/{[item for item in os.listdir(season_path) if "evi" in item ][0]}/SHAP_summary_Temporal Series.csv'
    #         season_sif_path = f'{season_path}/{[item for item in os.listdir(season_path) if "sif" in item][0]}/SHAP_summary_Temporal Series.csv'
    #         paths_evi[season] = season_evi_path
    #         paths_sif[season] = season_sif_path
    #     grid_evi_sif_tp_by_season_files(
    #         paths_evi=paths_evi,
    #         paths_sif=paths_sif,
    #         out_path=f'{scenario_path}_interactions.jpg',
    #         seasons=('spring', 'summer', 'autumn', 'winter'),
    #         condition_vars=('tm', 'vpd'),  # TM 用蓝/红；VPD 自动用 BrBG 绿/棕
    #         x_feature='tp',
    #         shap_suffix_evi='shap_value',  # 若 EVI/SIF 的 shap 后缀不同，这里分别指定
    #         shap_suffix_sif='shap_value',
    #         baseline_suffix='baseline',
    #         keep_outside_quantiles=(0.25, 0.75),
    #         line_span='axis',
    #         show_kde=True
    #     )

    '''分biogeo'''
    # biogeo_ids = [2,3,4,5,7,8]
    # for biogeo_id in biogeo_ids:
    #     path = r'D:\WuYong\after_first_revision\3 interactions'
    #     features = [f'tp_spring_baseline',
    #             f'tp_summer_baseline',
    #             f'tp_autumn_baseline',
    #             f'tp_winter_baseline'
    #             f'vpd',
    #             f'tm',
    #             # f'spei'
    #                 ]
    #     seasons = ['spring','summer','autumn','winter']
    #     scenarios = ['1.','2.','3.','4.']
    #     scenarios_path = os.listdir(path)
    #     for scenario_path in scenarios_path:
    #         paths_evi, paths_sif = {}, {}
    #         if scenario_path[0:2] not in scenarios: continue
    #         for i,season in enumerate(seasons):
    #             season_path = f'{path}/{scenario_path}/{str(i+1)}. {season}'
    #             season_evi_path = f'{season_path}/{[item for item in os.listdir(season_path) if "evi" in item ][0]}/SHAP_summary_Temporal Series.csv'
    #             season_sif_path = f'{season_path}/{[item for item in os.listdir(season_path) if "sif" in item][0]}/SHAP_summary_Temporal Series.csv'
    #             paths_evi[season] = season_evi_path
    #             paths_sif[season] = season_sif_path
    #         grid_evi_sif_tp_by_season_files_biogeo(
    #             biogeo_id,
    #             paths_evi=paths_evi,
    #             paths_sif=paths_sif,
    #             out_path=f'{scenario_path}_interactions_biogeo{biogeo_id}.jpg',
    #             seasons=('spring', 'summer', 'autumn', 'winter'),
    #             condition_vars=('tm', 'vpd'),  # TM 用蓝/红；VPD 自动用 BrBG 绿/棕
    #             x_feature='tp',
    #             shap_suffix_evi='shap_value',  # 若 EVI/SIF 的 shap 后缀不同，这里分别指定
    #             shap_suffix_sif='shap_value',
    #             baseline_suffix='baseline',
    #             keep_outside_quantiles=(0.25, 0.75),
    #             line_span='axis',
    #             show_kde=False,
    #         )

    # 测试用
    # legacy.decependence_plot_analysis(r'SHAP_summary_Temporal Series.csv','test.jpg','tm')
    '''# 随机森林结果空间可视化'''
    # scenario = r'4. low_elevation_evergreen'
    # root_path = f'D:\WuYong/after_first_revision/5. Spatial pattern/{scenario}/以vpd tm tp为因子'
    # # season = 'spring'
    # # indices = 'sif'
    # for indices in ['evi','sif']:
    #     for season in ['spring','summer','autumn','winter']:
    #         results_path = os.path.join(root_path,f'images_{indices}_low_elevation_everygreen_forspatial_{season}/SHAP_summary_Temporal Series.csv')
    #         legacy.rf_results_legacy_spatial_visualization(results_path,season)
    #         os.makedirs(f'after_first_revision/5. Spatial pattern/Mergetifs/{scenario}',exist_ok=True)
    #         legacy.merge_tifs(glob.glob(os.path.join(
    #             r'after_first_revision\5. Spatial pattern\test',
    #             '*.tif')), f'after_first_revision/5. Spatial pattern/Mergetifs/{scenario}/{indices}_spatial_pattern_{season}.tif')
    #         for single_tif in glob.glob(os.path.join(r'after_first_revision\5. Spatial pattern\test','*.tif')):
    #             os.remove(single_tif)
    '''EVI,SIF纬度分析可视化'''
 #    folders = {'1. low_elevation_broadforest and mixed forest': 'after_first_revision/5. Spatial pattern/Mergetifs/1. low_elevation_broadforest and mixed forest',
 # '2. high_elevation_broad': 'after_first_revision/5. Spatial pattern/Mergetifs/2. high_elevation_broad',
 # '4. low_elevation_evergreen': 'after_first_revision/5. Spatial pattern/Mergetifs/4. low_elevation_evergreen',
 # '3. high_elevation_everygreen': 'after_first_revision/5. Spatial pattern/Mergetifs/3. high_elevation_everygreen'}
 #    plot_seasonal_index_maps_with_profiles(
 #        folders_or_mapping=folders,
 #        shp_path=r"after_first_revision/data/European_no_belarus_ukrain.shp",
 #        index='sif',
 #        profile_mode='lat',
 #        out_path='panel_sif_lat.jpg'
 #    )

    '''计算0.1分辨率各个森林组的偏差值然后拼起来'''
    # path = r'afterfirst_revision_summary_lowelevation_broad.csv'
    # legacy.offset_calculation_spatial_visualization(path)
    # country_path = r'after_first_revision\6. offset\temp'
    # indices = ['evi','sif']
    # for indices_ in indices:
    #     indices_country_path = os.path.join(country_path,indices_)
    #     for year in range(2001,2024):
    #         indices_country_year_paths = glob.glob(os.path.join(indices_country_path,f'*_{indices_}_anomaly_{year}.tif'))
    #         if len(indices_country_year_paths) !=0:
    #             merge_path = os.path.join(indices_country_path,'merge')
    #             os.makedirs(merge_path,exist_ok=True)
    #             legacy.merge_tifs(indices_country_year_paths,os.path.join(merge_path,f'{year}.tif'))

    '''绘制TM和VPD的效应图'''
    indices = 'SIF'
    scenario_indices = 'sif'
    seasons = ['1. spring', '2. summer', '3. autumn', '4. winter']
    scenarios = [
        {
            "name": "Broadleaved and mixed (low elevation)",
            "global_dir": r"D:\WuYong\after_first_revision\3 interactions\1. low_elevation_broadforest and mixed forest",
        },
        {
            "name": "Broadleaved and mixed (high elevation)",
            "global_dir": r"D:\WuYong\after_first_revision\3 interactions\2. high_elevation_broad",
        },
        {
            "name": "Coniferous (low elevation)",
            "global_dir": r"D:\WuYong\after_first_revision\3 interactions\4. low_elevation_evergreen",
        },
        {
            "name": "Coniferous (high elevation)",
            "global_dir": r"D:\WuYong\after_first_revision\3 interactions\3. high_elevation_everygreen",
        },
    ]
    csv_name = r"SHAP_summary_Temporal Series.csv"

    plot_tm_vpd_shap_effects_split(
        indices=indices,
        scenario_indices=scenario_indices,
        seasons=seasons,
        scenarios=scenarios,
        csv_name=csv_name,
        out_path_tm="TM_shap_effects.jpg",
        out_path_vpd="VPD_shap_effects.jpg",
        figsize=(28, 20),  # 可再调大/调小
        dpi=300,
        base_fontsize=17
    )
