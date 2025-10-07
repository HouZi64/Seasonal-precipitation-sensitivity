#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 5/22/2024 15:56
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


class legacy_effects():
    def __init__(self):
       pass


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

if __name__ == '__main__':

    '''# 干旱区域物候指标提取'''
    legacy = legacy_effects()
    # 根据土地利用数据，干扰数据，DEM数据筛选标准如下：
    '''
    1.土地利用中选择 Broad-leaved forest(23)  Mixed forest(25)
    2. 高程低于800米
    扰动数据做记录
    土地利用阈值设为比例超过百分八十的点
    扰动设为扰动少于百分之20
    '''
    '''
    *****************由于有些国家的面积很大，所以下面的代码有些是分块运行的，有些是按国家运行的，建议如果复用，不需要按国家来分，直接按照矩形来切块
    *****************因为一开始已经用了国家了，所以后面就没有改成用矩形来切块，而是仍然用国家切块，并且对于国家很大的数据，单独用矩形切块
    '''
    '''
    基于筛除扰动后的数据和森林区域进行干旱遗留效应的统计建模
    1. 获取对应点的SPEI,GDD,VPD,TM
    2. 统计建模
    '''
    '''# # 1. 获取对应点的SPEI数据和GDD数据和温度'''
    country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
                    os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    gdd_path = r'D:\Data Collection\Temperature'
    SPEI_paths = [r'data collection\SPEI/spei03.nc',
                 r'Cdata collection\SPEI/spei06.nc']
    tem_path = r'D:\Data Collection\Temperature\T2M\T2M_2000to2023.nc'
    VPD_path = r'D:\Data Collection\Temperature\VPD/VPD_2000to2023.nc'
    TP_path = r'D:\Data Collection\other_factors/total_precipitation.nc'
    TP_path_2224 = r'D:\Data Collection\other_factors/total_precipitation_22_24.nc'
    SIF_path = r'data collection/SIF_tempory.nc'
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

