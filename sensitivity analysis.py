#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


    def temporal_window_analysis(self,paths,variable,start_year,end_year,dataset):
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
            variable_shap_ratios.append(data[variable].mean())

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
        variable_y_label_map = {'tp_spring_relative_shap_ratio_new':'TP_Spring','tp_summer_relative_shap_ratio_new':'TP_Summer','tp_autumn_relative_shap_ratio_new':'TP_Autumn'}
        # 绘制背景和轴
        plt.gca().set_facecolor('#f5f5f5')  # 坐标轴背景色设置为白色
        plt.xticks(ticks=x_numeric, labels=years, rotation=45, fontsize=15)
        plt.yticks(fontsize=10)
        plt.ylabel(f'{variable_y_label_map[variable]} effect on forest {dataset}', fontsize=20)
        # 添加网格
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        # 添加图例和颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # 为颜色条创建映射
        cbar = plt.colorbar(sm, aspect=30, pad=0.02)
        # cbar.set_label('SHAP relatively ratio', fontsize=20)
        plt.legend(fontsize=20, loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.split(path)[0],f'{variable_y_label_map[variable]}_temporal_fitting.jpg'),dpi = 600)
        plt.savefig(os.path.join(os.path.split(path)[0], f'{variable_y_label_map[variable]}_temporal_fitting.pdf'), dpi=600)

    def temporal_window_analysis_with_SPEI_anomaly(self,paths,variable,start_year,end_year,dataset):
        anomaly_variable = 'SPEI'     #可以替换为TM,VPD，但是要注意修改baseline和anomaly中的参数
        summary =  pd.read_csv('summary.csv').dropna()
        SPEI_baseline = summary[[col for col in summary.columns if '03_annual_' in col]].mean(axis=1)
        total_years = end_year - start_year + 1
        window_size = total_years - len(paths) + 1  # 推导窗口大小
        # 生成年份窗口
        years = [f"{start_year + i}-{start_year + i + window_size - 1}" for i in range(len(paths))]
        years_single = [int(item.split('-')[0]) for item in years]
        # 读取数据并计算变量 SHAP 比例
        variable_shap_ratios = []
        SPEI_anomaly = []
        for path in paths:
            group_year = os.path.split(path)[-1].split('_')[-2]
            data = pd.read_csv(path)
            variable_shap_ratios.append(data[variable].mean())
            SPEI_anomaly.append(((data[f'spei_03_annual_spei_group{group_year}']-SPEI_baseline).mean())/SPEI_baseline.std())   #当前group的vpd减去长期均值，然然后除以长期均值序列的标准差

        # 设置全局样式
        plt.rcParams['font.family'] = 'Arial'
        plt.figure(figsize=(10, 6))
        # 散点颜色映射（基于 SHAP 值）
        cmap = plt.get_cmap('PuBu').reversed()  # 使用 RdBu 渐变色
        norm = plt.Normalize(vmin=min(years_single), vmax=max(years_single))
        scatter_colors = [cmap(norm(val)) for val in years_single]
        # 绘制散点图
        plt.scatter(
            SPEI_anomaly,
            variable_shap_ratios,
            color=scatter_colors,
            edgecolor='black',
            s=80
        )
        for i, year in enumerate(years_single):
            plt.annotate(str(year), (SPEI_anomaly[i], variable_shap_ratios[i]),
                         textcoords="offset points", xytext=(5,5), ha='center', fontsize=12)

        # 绘制拟合线和置信区间
        sns.regplot(
            x=SPEI_anomaly,
            y=variable_shap_ratios,
            scatter=False,
            line_kws={"color": "dimgray", "linewidth": 2.5},  # 深灰色线条
            ci=95,
            label="OLS with 95% CI",
            truncate=False
        )
        variable_y_label_map = {'tp_spring_relative_shap_ratio_new':'TP_Spring','tp_summer_relative_shap_ratio_new':'TP_Summer','tp_autumn_relative_shap_ratio_new':'TP_Autumn'}
        # 绘制背景和轴
        plt.gca().set_facecolor('#f5f5f5')  # 坐标轴背景色设置为白色
        x_min, x_max = min(SPEI_anomaly), max(SPEI_anomaly)
        # 生成均匀间隔的刻度
        num_ticks = 5  # 你可以调整这个数值
        x_ticks = np.linspace(x_min, x_max, num_ticks)
        # 设置X轴刻度
        plt.xticks(ticks=x_ticks, labels=[f"{tick:.2f}" for tick in x_ticks], fontsize=15)
        plt.xlabel(f'{anomaly_variable} anomaly',fontsize=20)
        plt.yticks(fontsize=10)
        plt.ylabel(f'{variable_y_label_map[variable]} effect on forest {dataset}', fontsize=20)
        # 添加网格
        plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        # 添加图例和颜色条
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # 为颜色条创建映射
        cbar = plt.colorbar(sm, aspect=30, pad=0.02)
        # cbar.set_label('SHAP relatively ratio', fontsize=20)
        # plt.legend(loc="upper left",fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.split(path)[0],f'{variable_y_label_map[variable]}_{anomaly_variable}_analysis.jpg'),dpi = 600)
        plt.savefig(os.path.join(os.path.split(path)[0], f'{variable_y_label_map[variable]}_{anomaly_variable}_analysis.pdf'), dpi=600)
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

    def decependence_plot_analysis(self,path,features,output_path):

        data = pd.read_csv(path)
        lower_bound = data['tm'].quantile(0.05)
        upper_bound = data['tm'].quantile(0.95)
        data = data[(data['tm'] <= lower_bound) | (data['tm'] >= upper_bound)]
        X_test = data[features]

        df = pd.DataFrame({
            'TP_Spring': data['tp_spring'],
            'SHAP for TP_Spring': data['tp_spring_shap_value'],
            "TM": data['tm'],
            'year':data['year']
        })
        # 分组：低 VPD 和高 VPD
        low_vpd_threshold = np.percentile(df['TM'], 50)
        high_vpd_threshold = np.percentile(df['TM'], 50)

        low_vpd = df[df['TM'] <= low_vpd_threshold]
        high_vpd = df[df['TM'] >= high_vpd_threshold]

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
            low_vpd["TP_Spring"], low_vpd["SHAP for TP_Spring"]
        )
        low_vpd_formula = f"y = {low_vpd_slope:.3f}x + {low_vpd_intercept:.3f}"
        if low_vpd_intercept < 0:
            low_vpd_formula = f"y = {low_vpd_slope:.3f}x - {-low_vpd_intercept:.3f}"
        else:
            low_vpd_formula = f"y = {low_vpd_slope:.3f}x + {low_vpd_intercept:.3f}"

        # 计算高 VPD 的拟合公式
        high_vpd_slope, high_vpd_intercept, _, _, _ = linregress(
            high_vpd["TP_Spring"], high_vpd["SHAP for TP_Spring"]
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
            x=low_vpd["TP_Spring"],
            y=low_vpd["SHAP for TP_Spring"],
            cmap="Blues",
            fill=True,
            alpha=1,
            thresh=0.1,
            levels=10,
            zorder=1,
            linewidths=1
        )

        sns.kdeplot(
            x=high_vpd["TP_Spring"],
            y=high_vpd["SHAP for TP_Spring"],
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
            x = data["TP_Spring"]
            y = data["SHAP for TP_Spring"]
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
        plt.plot([], [], color="blue", linewidth=2.5, label=f"Low TM Fit: {low_vpd_formula}")
        plt.plot([], [], color="red", linewidth=2.5, label=f"High TM Fit: {high_vpd_formula}")
        plt.legend(loc="lower left", fontsize=20)
        # 添加标签和标题
        plt.xlabel("TP_Spring", fontsize=20)
        plt.ylabel("SHAP value for TP_Spring", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(0, np.quantile(low_vpd["TP_Spring"], 0.95))
        plt.savefig(output_path, dpi=600)
        plt.close()

    def seasonal_precipitataion_shaprelatively_boxplot(self,paths,analysis_items,output_path):
        seasons = ['Spring', 'Summer', 'Autumn']
        season_dic = {
            'Spring':'tp_spring_relative_shap_ratio_new', 'Summer':'tp_summer_relative_shap_ratio_new', 'Autumn':'tp_autumn_relative_shap_ratio_new'
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
    def plot_indices_timeseries_offset_with_colorbar(self,tif_paths, output_path, cmap='viridis', ncols=5):
        '''
        绘制折线图，折线图内部上下渐变填充，因为matplib本身不支持折线图内部渐变，只能离散颜色，所以这里的思路是
        1.全局使用上下渐变填充
        2.对折线图内部区域的反向区域进行白色填充覆盖来曲线救国
        对于2：首先需要插值时序数据，因为需要知道和x轴的交点位置，然后对各个区域内的位置进行调试
        此外，还有一个左右渐变的代码在上一个函数plot_indices_timeseries_offset_with_colorbar_left_right_gradient
        :param tif_paths:
        :param output_path:
        :param cmap:
        :param ncols:
        :return:
        '''
        def get_full_years_temporalvalues(years, temporal_series):
            # 画图时的填充需要对x轴交点区域处理
            intersections = []
            interpolated_values = []

            # 遍历 temporal_series 查找符号变化（即交点）
            for i in range(1, len(temporal_series)):
                if (temporal_series[i - 1] > 0 and temporal_series[i] < 0) or (
                        temporal_series[i - 1] < 0 and temporal_series[i] > 0):
                    # 通过线性插值估算交点位置
                    x1, y1 = years[i - 1], temporal_series[i - 1]
                    x2, y2 = years[i], temporal_series[i]

                    # 线性插值计算交点（y=0时对应的x）
                    x_intersect = x1 - (y1 * (x2 - x1)) / (y2 - y1)
                    intersections.append(x_intersect)

                    # 对应的 temporal_series 值应该是 0（交点）
                    interpolated_values.append(0)

            # 将交点和插值数据插入到 years 和 temporal_series 中
            all_years = np.concatenate((years, intersections))
            all_temporal_series = np.concatenate((temporal_series, interpolated_values))

            # 对合并后的数组进行排序
            sorted_indices = np.argsort(all_years)
            sorted_years = all_years[sorted_indices]
            sorted_temporal_series = all_temporal_series[sorted_indices]
            return sorted_years,sorted_temporal_series
        plt.rcParams['font.family'] = 'Arial'
        temporal_series = []
        years = []
        # 提取每个 TIFF 文件的均值
        for tif_path in tif_paths:
            years.append(int(os.path.split(tif_path)[-1].replace('.tif','')))
            with rasterio.open(tif_path) as src:
                data = src.read(1)
                mean_value = np.nanmean(data)
                temporal_series.append(mean_value)
        years = np.array(years)
        # 设置配色范围
        vmin, vmax = np.min(temporal_series), np.max(temporal_series)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = get_cmap(cmap)
        full_years,full_temporal_series = get_full_years_temporalvalues(years,temporal_series)
        # 创建图形
        fig, ax = plt.subplots(figsize=(15, 5))
        # 添加折线
        ax.plot(years, temporal_series, color='black', linewidth=1.5)
        res = 500  # 渐变的分辨率（越高越平滑）
        ymin, ymax = vmin, vmax
        gradient = np.linspace(ymin, ymax, res).reshape(-1, 1)  # 从 ymin 到 ymax 的渐变

        # 图表设置
        extent = [years[0], years[-1], ymin, ymax]
        ax.imshow(
            gradient,
            aspect="auto",
            extent=extent,
            origin="lower",
            cmap=cmap,
            norm=norm,
            alpha=0.5,  # 设置透明度
        )

        # 遮盖折线以下区域
        for i in range(len(full_years) - 1):
            # 在ax.fill(x_fill, y_fill, color="lightblue", edgecolor="none", zorder=2)
            # 这段代码中，我们使用四个点来定义一个矩形区域，填充折线与x轴之间的区域。解释一下这四个位置是如何构成填充区域的：
            # 解释：
            # x_fill = [x1, x2, x2, x1]：
            # x1和x2是当前折线段的两个x坐标（即full_years[i]和full_years[i + 1]）。
            # x_fill列表定义了矩形区域的x坐标，从左到右依次是：x1：折线段的左端点x坐标。 x2：折线段的右端点坐标。
            # x2：右端点的x坐标（重复），用来关闭矩形的右边。x1：左端点的x坐标（重复），用来关闭矩形的左边。这就构成了一个矩形的左右边界，起始点和结束点的x坐标是一样的。
            # y_fill = [y1, y2, 0, 0]：y1和y2是当前折线段的两个y坐标（即temporal_series[i]和temporal_series[i + 1]）。y_fill列表定义了矩形区域的y坐标，从上到下依次是：
            # y1：折线段的左端点y坐标（即该点的值）。
            # y2：折线段的右端点y坐标（即该点的值）。
            # 0：矩形的右下角y坐标，表示填充区域的底部在x轴上（y = 0）。
            # 0：矩形的左下角y坐标，也在x轴上（y = 0）。
            # 通过这四个点，形成了一个由折线、x 轴和垂直边界构成的矩形区域，矩形的顶部由折线的两端（y1和y2）定义，底部则与
            # x轴（y = 0）重合。
            # 获取当前两个点的坐标
            x1, x2 = full_years[i], full_years[i + 1]
            y1, y2 = full_temporal_series[i], full_temporal_series[i + 1]
            plot_linestatus = 'None'
            if y1 < 0 and y2 < 0: plot_linestatus = 'down'
            if y1 > 0 and y2 > 0: plot_linestatus = 'up'
            if y1 <0 and y2 == 0: plot_linestatus = 'down'
            if y1 == 0 and y2 > 0: plot_linestatus = 'up'
            if y1 > 0 and y2 == 0:plot_linestatus = 'up'
            # 判断折线段在x轴的上方还是下方
            if plot_linestatus == 'down':  # 折线段在x轴下方
                # 填充折线与底部的部分
                x_fill = [x1, x2, x2, x1]
                y_fill = [y1, y2, vmin, vmin]
                ax.fill(x_fill, y_fill, color="white", edgecolor="none", zorder=2)
                # 填充x轴上所有部分
                x_fill = [x1, x2, x2, x1]
                y_fill = [0, 0, vmax, vmax]
                ax.fill(x_fill, y_fill, color="white", edgecolor="none", zorder=2)

            else:  # 折线段在x轴上方
                # 填充折线和顶部间的区域
                x_fill = [x1, x2, x2, x1]
                y_fill = [y1, y2, vmax,vmax]
                ax.fill(x_fill, y_fill, color="white", edgecolor="none", zorder=2)
                # 填充x轴下所有区域
                x_fill = [x1, x2, x2, x1]
                y_fill = [0, 0, vmin,vmin]
                ax.fill(x_fill, y_fill, color="white", edgecolor="none", zorder=2)

        # 添加颜色条
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # cbar = plt.colorbar(sm, ax=ax, orientation="vertical", label="Grassland vitality")

        # 样式调整
        ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')  # 中心线
        ax.set_xticks(np.arange(int(years[0]), int(years[-1]), 1))  # 设置 X 轴刻度
        ax.set_xticklabels(np.arange(int(years[0]), int(years[-1]), 1), rotation=45)  # X 轴标签旋转
        ax.set_ylabel("Index Deviation from Baseline", fontsize=20)
        ax.tick_params(axis='x', labelsize=20)  # 设置 x 轴刻度字体大小
        ax.tick_params(axis='y', labelsize=20)  # 设置 y 轴刻度字体大小

        ax.set_facecolor("#f0f0f0")  # 设置背景颜色
        plt.tight_layout()
        plt.savefig(output_path,dpi=600)
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
    def rf_results_legacy_spatial_visualization(self,data_path,year):
        legacy_data = pd.read_csv(data_path)
        seasonal_items = ['tp_spring_relative_shap_ratio_new']
        countries = legacy_data['country'].unique().tolist()
        for country_name in countries:
            country_legacy_data = legacy_data[legacy_data['country'] == country_name]
            base_path = os.path.join(f'D:\Data Collection\RS\Disturbance/7080016/{country_name}/Wu Yong',f'{country_name}_2001_EVI_offset_sa.tif')
            if not os.path.exists(base_path):
                base_path = os.path.join(r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\2 forest sensitivity to water\test\5 year 空间可视化\2001',f'{country_name}.tif')

            country_basetif = gdal.Open(base_path)
            country_basetif_data = country_basetif.GetRasterBand(1).ReadAsArray()
            modified_data = np.full_like(country_basetif_data, np.nan, dtype=np.float32)
            for _, row in country_legacy_data.iterrows():
                modified_data[int(row['row']), int(row['col'])] = row[f'tp_spring_relative_shap_ratio_new']

            driver = gdal.GetDriverByName('GTiff')
            # Create a new GeoTIFF file with the modified data
            os.makedirs(f'test/{year}',exist_ok=True)
            output_path = os.path.join(f'test/{year}', f'{country_name}.tif')
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
    #    if 'ukraine' in country_tif or 'belarus' in country_tif or 'norway_plot_' in country_tif: continue
    #    if 'norway' in country_tif:
    #        country_name = os.path.split(country_tif)[-1]
    #        print(country_name)
    #        # for drought_year in drought_years:
    #        evi_path = os.path.join(country_tif,country_name+'_2001_EVI.tif')
    #        landcover_path = os.path.join(country_tif,country_name+'_land_cover.tif')
    #        disturbance_path = os.path.join(country_tif,'disturbance_year_1986-2020_{}_reprojection.tif'.format(country_name))
    #        mask_out_path = {'landcover_mask':os.path.join(country_tif,r'Wu Yong/mask_landcover.tif'),
    #                         'DEM_mask': os.path.join(country_tif, r'Wu Yong/mask_DEM.tif'),
    #                         f'combined_mask':os.path.join(country_tif,r'Wu Yong/mask_combined.tif')}
    #        legacy.generate_mask(evi_path, phenology_band,landcover_path, disturbance_path,DEM_path,mask_out_path)

    '''
    基于筛除扰动后的数据和森林区域进行干旱遗留效应的统计建模
    1. 获取对应点的SPEI,GDD,VPD,TM
    2. 统计建模
    '''
    '''# # 1. 获取对应点的SPEI数据和GDD数据和温度'''
    # country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
    #                 os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    # gdd_path = r'D:\Data Collection\Temperature'
    # SPEI_paths = [r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei03.nc',
    #              r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei06.nc']
    # tem_path = r'D:\Data Collection\Temperature\T2M\T2M_2000to2023.nc'
    # VPD_path = r'D:\Data Collection\Temperature\VPD/VPD_2000to2023.nc'
    # TP_path = r'D:\Data Collection\other_factors/total_precipitation.nc'
    # TP_path_2224 = r'D:\Data Collection\other_factors/total_precipitation_22_24.nc'
    # SIF_path = r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\2 forest sensitivity to water\data collection/SIF_tempory.nc'
    # chillingdays_data_path = r'D:\Data Collection\Temperature\chillingdays'
    # chillingdays_paths = glob.glob(os.path.join(chillingdays_data_path,'*.nc'))
    # drought_years = [2003, 2015, 2018,2019,2022]
    # norway_index = None
    # # for i, country_tif in enumerate(country_tifs):
    # #     if 'ukraine' in country_tif:
    # #         norway_index = i
    # #         break
    #
    # for i,country_tif in tqdm(enumerate(country_tifs)):
    #    # if 'ukraine' in country_tif or 'belarus' in country_tif or 'andorra' in country_tif or 'albania' in country_tif or 'austria' in country_tif or 'belgium' in country_tif or 'bosniaherzegovina' in country_tif        or 'bulgaria' in country_tif   or 'croatia' in country_tif or 'czechia' in country_tif or 'denmark' in country_tif or 'estonia' in country_tif or 'finland' in country_tif       or 'france' in country_tif or 'germany' in country_tif or 'greece' in country_tif or 'hungary' in country_tif or 'ireland' in country_tif or 'italy' in country_tif or 'latvia' in country_tif :continue
    #    # if country_tif == country_tifs[0]:
    #
    #    if ('ukraine' in country_tif or 'belarus' in country_tif or
    #            'norway' in country_tif or 'finland' in country_tif or 'france' in country_tif or 'italy' in country_tif or 'spain' in country_tif or 'sweden' in country_tif
    #           or 'andorra' in country_tif or 'liechtenstein' in country_tif or 'luxembourg' in country_tif):continue
    #    country_name = os.path.split(country_tif)[-1]
    #    # if i <= norway_index:
    #    #     print(country_name+'跳过')
    #    #     continue
    #    if 'unitedkingdom_plot_' in country_tif:
    #        print(country_name)
    #        evi_tifs = glob.glob(os.path.join(country_tif,'*EVI*.tif'))
    #        evi_tifs = sorted(evi_tifs, key=lambda x: int(x.split('_EVI')[0][-4:]))          #按照年份排序
    #        mask_tifs = [os.path.join(country_tif,'Wu Yong/mask_combined.tif')]
    #        legacy.summarize_raster_by_mask_aggregation(evi_tifs,mask_tifs,SPEI_paths,tem_path,VPD_path,SIF_path,TP_path,TP_path_2224)

    # #################重新获取冬季数据，因为冬季降雨数据之前选择的是到下一年，这是不对的，应该是当年的1，2，12月############################################################
    # country_tifs = [os.path.join(r'H:\吴勇论文暂存', item) for item in
    #                 os.listdir(r'H:\吴勇论文暂存') if '.zip' not in item]
    # gdd_path = r'D:\Data Collection\Temperature'
    # SPEI_paths = [r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei03.nc',
    #              r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei06.nc']
    # tem_path = r'D:\Data Collection\Temperature\T2M\T2M_2000to2023.nc'
    # VPD_path = r'D:\Data Collection\Temperature\VPD/VPD_2000to2023.nc'
    # TP_path = r'D:\Data Collection\other_factors/total_precipitation.nc'
    # TP_path_2224 = r'D:\Data Collection\other_factors/total_precipitation_22_24.nc'
    # SIF_path = r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\2 forest sensitivity to water\data collection/SIF_tempory.nc'
    # chillingdays_data_path = r'D:\Data Collection\Temperature\chillingdays'
    # chillingdays_paths = glob.glob(os.path.join(chillingdays_data_path,'*.nc'))
    # drought_years = [2003, 2015, 2018,2019,2022]
    # norway_index = None
    #
    # for i,country_tif in tqdm(enumerate(country_tifs)):
    #
    #    # if ('ukraine' in country_tif or 'belarus' in country_tif or 'france' in country_tif or 'italy' in country_tif or
    #    # 'norway' in country_tif or 'spain' in country_tif or 'sweden' in country_tif or 'unitedkingdom' in country_tif):continue
    #    if ('plot_' in country_tif):
    #        country_name = os.path.split(country_tif)[-1]
    #        print(country_name)
    #        evi_tifs = glob.glob(os.path.join(country_tif,'*EVI*.tif'))
    #        evi_tifs = sorted(evi_tifs, key=lambda x: int(x.split('_EVI')[0][-4:]))          #按照年份排序
    #        mask_tifs = [os.path.join(country_tif,'Wu Yong/mask_combined.tif')]
    #        legacy.summarize_raster_by_mask_aggregation_onlywinterTP(evi_tifs,mask_tifs,SPEI_paths,tem_path,VPD_path,SIF_path,TP_path,TP_path_2224)



    '''# # 汇总数据'''
    # country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
    #                 os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    # data_final = pd.DataFrame()
    # country_names = []
    # for i,country_tif in tqdm(enumerate(country_tifs)):
    #     country_name = os.path.split(country_tif)[-1]
    #     if os.path.exists(os.path.join(country_tif,f'Wu Yong/inform_sum_sa_EVI_SIF.csv')):
    #         country_names.append(country_name)
    #         data_path = os.path.join(country_tif,f'Wu Yong/inform_sum_sa_EVI_SIF.csv')
    #         data_country = pd.read_csv(data_path)
    #         data_country['country'] = country_name
    #         data_final = pd.concat([data_final,data_country])
    # data_final.to_csv(f'summary.csv')


    # @@@@@@@@@@@@@@@@@@@单独补充冬季降雨数据，因为之前的冬季选择的时间是当年的12月和下一年的1月2月，这是不对的，重新补充为当年的1月2月12月
    # data_final = pd.read_csv('summary.csv')
    # country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
    #                 os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    # country_tifs2 = [os.path.join(r'H:\吴勇论文暂存', item) for item in
    #                 os.listdir(r'H:\吴勇论文暂存') if '.zip' not in item]
    # country_tifs2.remove( 'H:\\吴勇论文暂存\\绘图暂存，带有1年的EVI数据，忘记了干啥用')
    # country_tifs = country_tifs + country_tifs2
    # for i,country_tif in tqdm(enumerate(country_tifs)):
    #     country_name = os.path.split(country_tif)[-1]
    #     if country_name == 'luxembourg':continue
    #     if os.path.exists(os.path.join(country_tif,'Wu Yong/inform_sum_sa_(only_winterTP).csv')):
    #         data = pd.read_csv(os.path.join(country_tif,'Wu Yong/inform_sum_sa_(only_winterTP).csv'))
    #         # 筛选 `data_final` 中属于当前 `country` 的数据
    #         mask = data_final['country'] == country_name
    #         data_final_country = data_final[mask]
    #         if data.shape[0]!=data_final_country.shape[0]:
    #             print('wrong')
    #         # 进行数据匹配 (以 `row` 和 `col` 为键合并)
    #         merged_data = data_final_country.merge(data, on=['row', 'col'], suffixes=('_old', ''))
    #         # 仅更新 `data_final` 中的对应数据 (不修改 `row` 和 `col`)
    #         update_cols = [col for col in data.columns if col not in ['row', 'col','weights']]
    #         data_final.loc[mask, update_cols] = merged_data[update_cols].values
    #
    # # 保存更新后的 `data_final`
    # data_final.to_csv('summary_updated_winterTP.csv', index=False)
    '''# 3. 随机森林建模'''
    # data_path = f'summary.csv'
    # indices_name = 'sif'
    # time_information = {'evi':{'start':2001,'end':2023},'sif':{'start':2001,'end':2016}}
    # legacy.rf_modelling(data_path,time_information[indices_name]['start'],time_information[indices_name]['end'],indices_name)

    '''随机森林结果绘制（时序）summary图或者其他图'''
    # indices_name = 'EVI'
    # data_path = f'时序建模/images {indices_name}/SHAP_summary_Temporal Series_new.csv'
    # legacy.random_forest_summary(data_path)
    '''EVI,SIF空间时序可视化'''
    # 生成EVI SIF sa数据
    # start_year = 2001
    # end_year = 2024
    # indices_name = r'EVI'
    # data_path = r'summary.csv'
    # legacy.indices_spatial_visualization(data_path,start_year,end_year,indices_name)

    # 可视化
    # paths_evi = glob.glob(os.path.join(r'paper plots\sensitivity map\10年窗口\EVI','*.tif'))
    # paths_sif = glob.glob(os.path.join(r'paper plots\sensitivity map\10年窗口\SIF','*.tif'))
    # evi_colormap = 'Reds'
    # sif_colormap = 'Reds'
    # legacy.plot_tifs_with_colorbar(paths_evi, "10year_sensitivity_spatial_evi.jpg", cmap=evi_colormap, ncols=3)
    # legacy.plot_tifs_with_colorbar(paths_sif, "10year_sensitivity_spatial_sif.jpg", cmap=sif_colormap, ncols=3)
    # # 可视化时序偏差图
    # paths_evi = glob.glob(os.path.join(r'paper plots\offset map\EVI','*.tif'))
    # paths_sif = glob.glob(os.path.join(r'paper plots\offset map\SIF','*.tif'))
    # evi_colormap = 'RdYlGn'
    # sif_colormap = 'RdYlBu'
    # legacy.plot_indices_timeseries_offset_with_colorbar(paths_evi, "tif_time_series_evi.jpg", cmap=evi_colormap, ncols=6)
    # legacy.plot_indices_timeseries_offset_with_colorbar(paths_sif, "tif_time_series_sif.jpg", cmap=sif_colormap, ncols=6)


    # 可视化三个季节相对贡献的箱型图
    # path_evi = r'时序建模/images EVI/SHAP_summary_Temporal Series_new.csv'
    # path_sif = r'时序建模/images SIF/SHAP_summary_Temporal Series_new.csv'
    # output_path = r'时序建模/boxplot.jpg'
    # precipitation_item = [r'tp_spring_relative_shap_ratio_new','tp_summer_relative_shap_ratio_new','tp_autumn_relative_shap_ratio_new']
    # legacy.seasonal_precipitataion_shaprelatively_boxplot([path_evi,path_sif],precipitation_item,output_path)
    '''# 随机森林结果可视化'''
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
    # data_paths = glob.glob(os.path.join(r'10年窗口/images window SIF','SHAP_summary_*_new.csv'))
    # variable_name = r'tp_spring_relative_shap_ratio_new'
    # start_year = 2001
    # end_year = 2016
    # dataset = 'SIF'
    # legacy.temporal_window_analysis(data_paths,variable_name,start_year,end_year,dataset)

    '''窗口时序趋势变化和SPEI异常关系图'''
    data_paths = glob.glob(os.path.join(r'10年窗口/images window SIF','SHAP_summary_*_new.csv'))
    variable_name = r'tp_spring_relative_shap_ratio_new'
    start_year = 2001
    end_year = 2016
    dataset = 'SIF'
    legacy.temporal_window_analysis_with_SPEI_anomaly(data_paths,variable_name,start_year,end_year,dataset)


    '''基于时序建模的交互作用分析'''
    # path = r'时序建模/images SIF/SHAP_summary_Temporal Series.csv'
    # features = [f'tp_spring',
    #         f'tp_summer',
    #         f'tp_autumn',
    #         f'vpd',
    #         f'tm',
    #         f'spei']
    # legacy.decependence_plot_analysis(path,features,output_path=r'paper plots/interaction/SIF_TM_TP.jpg')
    '''# 随机森林结果空间可视化'''
    # results_path = glob.glob(os.path.join(r'10年窗口\images window SIF/','SHAP_summary_*_new.csv'))
    # for path in results_path:
    #     year = os.path.split(path)[-1].replace('SHAP_summary_','').replace('_new.csv','')
    #     legacy.rf_results_legacy_spatial_visualization(path,year)
    '''EVI,SIF纬度分析可视化'''
    # data_path = r'paper plots\sensitivity map\SIF'
    # year_range = [2001,2013]
    # legacy.rf_results_latitude_sensitivity_visualization(data_path,year_range)
