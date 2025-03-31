#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import glob
import json
from rasterio.windows import Window
import rasterio
from matplotlib import pyplot as plt
from bs4 import BeautifulSoup
import time
from datetime import datetime,timedelta
from tqdm import tqdm
from matplotlib.dates import YearLocator,DateFormatter
import geemap
import ee
from osgeo import gdal,ogr
from multiprocessing import Pool
from functools import partial
from scipy.ndimage import zoom
import re
import subprocess
import psycopg2
import csv
import xarray as xr
import rioxarray
import cdsapi
from rasterio.mask import mask
from shapely.geometry import box
from skimage.transform import resize
from climate_indices import indices
from climate_indices import compute
from rasterio.warp import calculate_default_transform,reproject,Resampling
import math
from rasterio.merge import merge
def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Data_Visualization():
    def __init__(self):
        self.pep_data_path = r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data_analysis\station_distribution/pep_plus_2.csv'
        self.webcam_path = r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data_analysis\station_distribution/webcam_plus2.csv'
        self.phenocam_path = r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data_analysis\station_distribution/phenocam_eur_plus2.csv'

    def station_temporal_visualization(self,station):
        if station == 'pep':
            data = pd.read_csv(self.pep_data_path)
            data['start'] = pd.to_datetime(data['start'], format='%Y/%m/%d/%H%M')
            data['end'] = pd.to_datetime(data['end'], format='%Y/%m/%d/%H%M')
            fontsize = 45
        elif station == 'webcam':
            data = pd.read_csv(self.webcam_path)
            data['start'] = pd.to_datetime(data['start'], format='%Y-%m-%d-%H%M')
            data['end'] = pd.to_datetime(data['end'], format='%Y-%m-%d-%H%M')
            fontsize = 100
        elif station == 'phenocam':
            data = pd.read_csv(self.phenocam_path)
            data['start'] = pd.to_datetime(data['start'], format='%m/%d/%Y')
            data['end'] = pd.to_datetime(data['end'], format='%m/%d/%Y')
            fontsize = 45
        # 设置字体大小
        plt.rcParams.update({'font.size': fontsize})  # 根据需要调整字体大小

        # 动态调整图片大小，使其足够长以充分显示61条数据
        figsize = (data.shape[0], data.shape[0])
        plt.figure(figsize=figsize)

        # 绘图
        for i, row in tqdm(data.iterrows()):
            plt.plot([row['start'], row['end']], [i, i], marker='o', label=row['s_id'])
            # 添加标注
            label_x = pd.Timestamp(((row['start'].timestamp() + row['end'].timestamp()) / 2),unit='s')  # 取中点作为标注位置
            label_y = i
            plt.text(label_x, label_y, str(row['s_id']), ha='center', va='center', fontsize=fontsize/2)

        # 设置图形属性
        plt.xlabel('Time')
        plt.ylabel('Station ID')
        plt.title('Start and End Time for Each Station')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 将图例放在外面
        plt.grid(True)

        # 设置时间轴刻度间隔为一年，仅显示年份
        plt.gca().xaxis.set_major_locator(YearLocator())
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))

        plt.tight_layout()

        # 显示图形
        plt.savefig('{}_temporal_visualation.jpg'.format(station))


class Data_Preprocess():
    def __init__(self):
        self.phenocam_csv = r'data_analysis\station_distribution/phenocam_eur_plus2.csv'
        self.webcam_csv = r'data_analysis\station_distribution/webcam_plus2.csv'
        self.pep_csv = r'data_analysis\station_distribution/pep_plus_2.csv'

    def station_data_filtering_by_drought_event(self,drgouht_time,save_path,temperal_limitition = 1):
        '''
        given a specific drought event, filtering the station data based on the temperal limitition, e.g. given the '['2018-04','2018-12']' drought event
        the required time should include time from 2017-04 to 2019-12 at least
        :param drgouht_time:
        :param temperal_limitition:
        :return:
        '''
        start_date = datetime.strptime(drgouht_time[0],'%Y-%m')
        start_date_adjusted = start_date - timedelta(days = temperal_limitition*365)
        end_date = datetime.strptime(drgouht_time[1], '%Y-%m')
        end_date_adjusted = end_date + timedelta(days = temperal_limitition*365)
        phenocam_data = pd.read_csv(self.phenocam_csv)
        phenocam_data['start'],phenocam_data['end'] = pd.to_datetime(phenocam_data['start']),pd.to_datetime(phenocam_data['end'])
        webcam_data = pd.read_csv(self.webcam_csv)
        webcam_data['start'], webcam_data['end'] = pd.to_datetime(webcam_data['start']), pd.to_datetime(webcam_data['end'])
        pep_data = pd.read_csv(self.pep_csv)
        pep_data['start'], pep_data['end'] = pd.to_datetime(pep_data['start']), pd.to_datetime(pep_data['end'])

        # 选择时间范围
        phenocam_data_selected = phenocam_data[(phenocam_data['start']<=start_date_adjusted) & (phenocam_data['end']>=end_date_adjusted)]
        webcam_data_selected = webcam_data[(webcam_data['start'] <= start_date_adjusted) & (webcam_data['end'] >= end_date_adjusted)]
        pep_data_selected = pep_data[(pep_data['start'] <= start_date_adjusted) & (pep_data['end'] >= end_date_adjusted)]

        # filter pep, to choose phasid of 11 and 19
        pep_raw_data = []
        files = glob.glob(os.path.join(self.raw_pep_path, '*.csv'))
        for file in files:
            df = pd.read_csv(file)
            pep_raw_data.append(df)
        pep_raw_data = pd.concat(pep_raw_data, ignore_index=True)

        # 从 pep_data_selected 中提取 s_id 列
        s_ids = pep_data_selected['s_id'].unique().tolist()
        phaseids_needed = [11]
        for sid in s_ids:
            if_meet = True
            pep_raw_data_station = pep_raw_data[pep_raw_data['s_id']==sid]
            station_years = pep_raw_data_station['year']
            # 允许缺失所有年份缺少三年的数据
            allowed_lack = 1
            for year in range(station_years.min(),station_years.max()+1):

                year_station = pep_raw_data_station[pep_raw_data_station['year']==year]
                year_phaseids = year_station['phase_id'].unique().tolist()

                for id in phaseids_needed:
                    if id not in year_phaseids:allowed_lack += 1
            if allowed_lack > 3:if_meet = False
            if not if_meet:pep_data_selected = pep_data_selected[pep_data_selected['s_id'] != sid]






        # for i in range(0,s_ids.shape[0]):
        #     sid = s_ids.iloc[i]
        #     sid_phaseid = pep_raw_data[pep_raw_data['s_id'] == sid]['phase_id']
        #     contains_11_or_19 = (sid_phaseid == 11) & (sid_phaseid == 19)
        #     if contains_11_or_19.any():
        #         continue
        #     else:
        #         pep_data_selected = pep_data_selected[pep_data_selected['s_id'] != sid]
        merged = pd.concat([phenocam_data_selected,webcam_data_selected,pep_data_selected])
        merged.to_csv(save_path)

    # 根据区域边界裁剪 SPEI 数据
    def crop_spei_to_region(self,spei_data, region_bounds):
        # 获取经纬度
        lat = spei_data['lat']
        lon = spei_data['lon']

        # 获取区域范围的SPEI数据
        spei_cropped = spei_data.sel(
            lon=slice(region_bounds.left, region_bounds.right),
            lat=slice(region_bounds.top, region_bounds.bottom)
        )
        return spei_cropped

    # 使用区域掩膜进一步精确裁剪
    def mask_spei_with_region(self,spei_cropped, region_mask, region_bounds):
        # 创建shapely多边形
        region_polygon = [box(*region_bounds)]

        # 获取裁剪区域的经纬度索引
        lon_min, lon_max = spei_cropped.lon.values.min(), spei_cropped.lon.values.max()
        lat_min, lat_max = spei_cropped.lat.values.min(), spei_cropped.lat.values.max()

        # 应用掩膜
        spei_masked = np.where(region_mask == 0, np.nan, spei_cropped['spei'].values)

        # 将掩膜后的SPEI数据返回为一个新的xarray数据集
        spei_cropped['spei'].values = spei_masked

        return spei_cropped
    def align_data_to_country(self,data_input_path,align_data_path,data_type):
        file_type = os.path.splitext(align_data_path)[1]
        # tif 数据的切割
        if file_type == '.tif':
            if data_type == 'phenology':
                output_tif_path = os.path.join(os.path.split(data_input_path)[0],os.path.split(data_input_path)[-1].replace(r'.tif',r'_phenology{}.tif'.format(re.search(r'(\d{4})_\d{2}_\d{2}', align_data_path).group(1)))).replace('forestcover_','')
            if data_type == 'land_cover':
                # 土地利用所用到的输出路径
                output_tif_path = os.path.join(os.path.split(data_input_path)[0],os.path.split(os.path.split(data_input_path)[0])[-1]+'_land_cover.tif')
            if data_type == 'EVI':
                output_tif_path = os.path.join(os.path.split(data_input_path)[0],os.path.split(os.path.split(data_input_path)[0])[-1] + f'_{os.path.split(align_data_path)[-1][0:4]}_EVI.tif')
            # 读取区域的tif文件以获取掩膜
            with rasterio.open(data_input_path) as country_tif:
                country_mask = country_tif.read(1)  # 读取第一个波段作为掩膜
                country_transform = country_tif.transform  # 获取地理变换信息
                country_bounds = country_tif.bounds  # 获取国家的边界
            # 将边界框转换为shapely的Polygon对象
            country_polygon = [box(*country_bounds)]
            # 读取欧洲的tif文件
            with rasterio.open(align_data_path) as europe_tif:
                # 使用德国的边界多边形来裁剪欧洲的TIFF文件
                europe_data_cropped, europe_transform = mask(europe_tif, country_polygon, crop=True)
                # 通过掩膜获取德国区域的物候数据
                if data_type == 'phenology':
                    europe_data_cropped[np.tile(np.expand_dims(resize(country_mask,(europe_data_cropped.shape[1],europe_data_cropped.shape[2]), order=0,preserve_range=True).astype(np.uint8),axis=0),(5,1,1)) == 255] = -9999  # 设置区域外的部分为-9999
                if data_type == 'land_cover':
                    europe_data_cropped[np.expand_dims(resize(country_mask, (europe_data_cropped.shape[1], europe_data_cropped.shape[2]), order=0,preserve_range=True).astype(np.uint8), axis=0) == 255] = -128  # 设置区域外的部分为-128
                if data_type == 'EVI':
                    europe_data_cropped[np.repeat(np.expand_dims(
                        resize(country_mask, (europe_data_cropped.shape[1], europe_data_cropped.shape[2]), order=0,
                               preserve_range=True).astype(np.uint8), axis=0),2,axis=0) == 255] = np.nan  # 设置区域外的部分为-128

                # 保存裁剪后的数据为新的TIFF文件
                with rasterio.open(
                        output_tif_path,
                        'w',
                        driver='GTiff',
                        height=europe_data_cropped.shape[1],
                        width=europe_data_cropped.shape[2],
                        count=europe_tif.count,
                        dtype=europe_data_cropped.dtype,
                        crs=europe_tif.crs,
                        transform=europe_transform,
                        compress=europe_tif.compression,  # 保留压缩信息
                        photometric=europe_tif.photometric,  # 保留色彩解释
                        nodata= np.nan
                ) as dst:
                    for i in range(europe_data_cropped.shape[0]):
                        dst.write(europe_data_cropped[i], i+1)
                        dst.update_tags(i + 1, **europe_tif.tags(i + 1))  # 保留波段的元数据
                    # 保留文件级别的元数据
                    dst.update_tags(**europe_tif.tags())
        if file_type == '.nc':
            '''
            考虑到SPEI的分辨率是0.5°，物候数据是0.005°，直接裁剪范围不好确定
            把SPEI重采样到0.005°误差太大，所以以下代码没有调试好，若是重新使用，需要重新调试
            '''
            # 读取区域TIF数据
            with rasterio.open(data_input_path) as region_tif:
                region_mask = region_tif.read(1)  # 读取第一个波段
                region_bounds = region_tif.bounds  # 获取区域的边界
            spei_data = xr.open_dataset(align_data_path)
            # 根据区域边界裁剪SPEI数据
            spei_cropped = self.crop_spei_to_region(spei_data,region_bounds)
            # 使用区域掩膜进一步裁剪SPEI数据
            spei_masked = self.mask_spei_with_region(spei_cropped, region_mask, region_bounds)
            spei_masked.to_netcdf('spei_masked_region.nc')

    def stack_tif_as_nc(self,data_list,band):

        #  NetCDF 信息：
        # lat 是从 -89.75 到 89.75 递增的（南到北）。
        # lon 是从 -179.8 到 179.8 递增的（西到东）。
        # 在 GeoTIFF 中，通常：
        # 纬度（lat）通常是从北到南递减的，如果从北向南存储数据。
        # 经度（lon）通常是从西到东递增的，和 NetCDF 文件中的经度一样。
        # 区别：
        # 纬度方向：NetCDF 文件中的纬度是从南到北递增的，而通常 GeoTIFF 文件的纬度方向可能是从北到南递减的。
        # 经度方向：NetCDF 和 GeoTIFF 通常在经度方向上都是递增的（从西到东）。
        band_attr = {1:'SOS',3:'EOS'}
        # 读取第一个TIFF文件，获取维度和坐标信息
        with rasterio.open(data_list[0]) as src:
            data = src.read(1)  # 读取第一个波段
            # 计算纬度和经度数组，注意方向和长度
            lat_len = data.shape[0]
            lon_len = data.shape[1]
            transform = src.transform
            # 纬度从南到北递增
            latitude = np.arange(src.bounds.bottom, src.bounds.top, -transform[4])
            if transform[4] < 0:
                latitude = latitude[::-1]  # 如果纬度是递减的，反转使其递增
            longitude = np.arange(src.bounds.left, src.bounds.right, transform[0])

            # 修正纬度数组长度，如果它比数据的纬度维度多一个元素
            if len(latitude) > lat_len:
                latitude = latitude[:lat_len]
            # 修正经度数组长度，如果它比数据的经度维度多一个元素
            if len(longitude) > lon_len:
                longitude = longitude[:lon_len]

            crs = src.crs  # 获取投影信息
        # 创建空数组存储所有年份的数据
        all_years_data = np.zeros((len(data_list), lat_len, lon_len))
        # 创建一个xarray Dataset
        phenology_ds = xr.Dataset()
        # 遍历每一个tif文件，并将其添加到xarray Dataset中
        for i, tif_file in enumerate(tqdm(data_list)):
            year = 2001 + i  # 根据文件顺序获取年份
            with rasterio.open(tif_file) as src:
                data = src.read(band)  # 读取第band个波段数据
                # 如果纬度是递减的，需要对数据进行翻转
                # if transform[4] < 0:
                #     data = np.flipud(data)
                all_years_data[i, :, :] = data
        # 创建DataArray并添加到Dataset中
        phenology_da = xr.DataArray(
            all_years_data,
            dims=["time","lat", "lon"],
            coords={"time": np.arange(2001, 2001 + len(data_list)),"lat": latitude, "lon": longitude},
            attrs={"units": band_attr[band], "description": f"Phenology {band_attr[band]} from 2001 to {2001 + len(data_list) - 1}","crs":crs.to_string()},
        )

        # 为每个变量（即每年的数据）添加一个时间坐标
        # phenology_ds = phenology_ds.assign_coords(time=("time", np.arange(2001, 2021)))
        phenology_ds = xr.Dataset({band_attr[band]: phenology_da})

        # 保存为NetCDF文件
        phenology_ds.to_netcdf(os.path.join(save_path,"phenology_{}_2001_2021.nc".format(band_attr[band])))

    def CSIF_annual_calculation(self,root_path,output_path):
        '''
        计算CSIF数据（来源于https://osf.io/8xqy6/中allsky数据）的年均值，由于这个数据的单个nc里面没有时间维度的数据，所以遍历每个nc求和，然后计算均值
        :param root_path:
        :return:
        '''
        results_tempory = []    #所有年份汇总结果
        years = np.array([int(item) for item in os.listdir(root_path)])
        for year_item in os.listdir(root_path):
            paths = glob.glob(os.path.join(root_path,year_item,'*.nc'))
            result_sum = 0

            number_minus = 0
            for i,path in enumerate(paths):
                data = xr.open_dataset(path)['all_daily_SIF'].values
                if np.unique(data).shape[0] == 1:
                    # 在2002年的数据中发现了一个全为nan的数据，导致最后的数据是空的，所以在这里把他排除掉，同时减去一个数据表示正确的均值
                    number_minus += 1
                    continue
                result_sum += data
                print(f'processing {year_item} year, {path}')
            result_avg = result_sum / (i+1 - number_minus)
            results_tempory.append(result_avg)
        example_data = xr.open_dataset(path)['all_daily_SIF']   #读取示例数据以获取lats，lons

        ds = xr.Dataset(
            {
                "annual_SIF": (("year", "lat", "lon"), np.array(results_tempory))
            },
            coords={
                "year": years,
                "lat": example_data['lat'].values,
                "lon": example_data['lon'].values,
            }
        )
        ds.to_netcdf(os.path.join(output_path,'SIF_tempory_new.nc'))

    def split_tif_dynamic(self,input_tif, output_dir, num_blocks=4):
        """
        将 TIFF 文件动态切割为 num_blocks 块，保留地理信息。
        """
        country_name = os.path.split(os.path.split(input_tif)[0])[-1]
        file_name = os.path.split(input_tif)[-1]
        # 打开输入 TIFF 文件
        with rasterio.open(input_tif) as src:
            # 获取 TIFF 的宽度和高度
            width = src.width
            height = src.height
            # 计算行列分块数（假设 num_blocks 是一个完全平方数）
            grid_size = int(math.sqrt(num_blocks))
            if grid_size ** 2 != num_blocks:
                raise ValueError("num_blocks 必须是一个完全平方数，例如 4, 9, 16 等")
            # 计算每个块的宽度和高度
            tile_width = width // grid_size
            tile_height = height // grid_size

            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 遍历行和列生成窗口
            plot_id = 0
            for row in range(grid_size):
                for col in range(grid_size):
                    plot_id+=1
                    # 定义窗口
                    window = Window(
                        col * tile_width,
                        row * tile_height,
                        tile_width,
                        tile_height
                    )

                    # 生成输出文件路径

                    if not os.path.exists(os.path.join(output_dir,f'{country_name}_plot_{plot_id}')):
                        os.makedirs(os.path.join(output_dir,f'{country_name}_plot_{plot_id}'))
                    output_path = os.path.join(os.path.join(output_dir,f'{country_name}_plot_{plot_id}'), file_name)

                    # 更新元数据
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "width": tile_width,
                        "height": tile_height,
                        "transform": rasterio.windows.transform(window, src.transform)
                    })

                    # 写入新的 TIFF 文件
                    with rasterio.open(output_path, "w", **out_meta) as dst:
                        dst.write(src.read(window=window))
                    print(f"Saved {output_path}")


class PEP725():
    def __init__(self):

    def reorganize_data(self,path):
        df = pd.read_csv(path,sep=';')
        file_name = os.path.split(path)[-1][:-4]+'_reorganized.csv'
        df.to_csv(os.path.join(self.root_path,'reorganized/'+file_name))
        return df
    def station_distribution(self):
        '''
        确定每个站点的地理位置（经纬度和高程），每个站点的时间范围
        :return:
        '''
        files = glob.glob(os.path.join(self.reorganized_root_path,'*.csv'))
        # 创建空列表存储结果
        results = []    #空间分布
        temporal_results = []   #时间分布
        for file in tqdm(files):
            df = pd.read_csv(file)
            temporal_results.append(df.groupby('s_id')['date'].agg(['min', 'max']))
            # 获取 s_id 的唯一值
            unique_s_ids = df['s_id'].unique()
            # 遍历每个唯一的 s_id
            for s_id in unique_s_ids:
                # 获取对应 s_id 的行
                subset = df[df['s_id'] == s_id]
                # 获取该 s_id 对应的第一行的 lat 和 lon
                lat = subset.iloc[0]['lat']
                lon = subset.iloc[0]['lon']
                alt = subset.iloc[0]['alt']
                alt_dem = subset.iloc[0]['alt_dem']
                # 添加到结果列表中
                results.append({'s_id': s_id, 'lat': lat, 'lon': lon, 'alt': alt, 'alt_dem': alt_dem})

        results_df = pd.DataFrame(results)
        results_df_unique = results_df.drop_duplicates(subset='s_id',keep='first',inplace=False)

        temporal_combined = pd.concat([temporal_result for temporal_result in temporal_results])
        temporal_combined_results = temporal_combined.groupby('s_id').agg({'min': 'min', 'max': 'max'})

        results_merged = pd.merge(temporal_combined_results, results_df_unique,on='s_id')
        new_columns = {'min':'start', 'max':'end'}
        results_merged = results_merged.rename(columns=new_columns)

        time_difference = pd.to_datetime(results_merged['end']) - pd.to_datetime(results_merged['start'])
        time_difference = round(time_difference.dt.days / 365)
        results_merged['last'] = time_difference
        results_merged[results_merged['last'] != 0.0].to_csv(os.path.join(self.root_path,'station_distribution.csv'))
        return results_df_unique

class Database():
    def __init__(self):

        self.password_postgre = 85354
        self.port_postgre = 5432
        self.user_postgre = 'postgres'

    def csv_to_postgre(self,dbname,csv_path):
        conn = psycopg2.connect(
            dbname = dbname,
            user = self.user_postgre,
            password = self.password_postgre,
            host = 'localhost',
            port = self.port_postgre
        )
        cur = conn.cursor()
        table_columns = ['plot', 'row', 'col', '2000_03_05',
       '2000_03_21', '2000_04_06', '2000_04_22', '2000_05_08',
       '2000_05_24', '2000_06_09', '2000_06_25', '2000_07_11',
       '2000_07_27', '2000_08_12', '2000_08_28', '2000_09_13',
       '2000_09_29', '2000_10_15', '2000_10_31', '2000_11_16',
       '2000_12_02', '2000_12_18', '2001_01_01', '2001_01_17',
       '2001_02_02', '2001_02_18', '2001_03_06', '2001_03_22',
       '2001_04_07', '2001_04_23', '2001_05_09', '2001_05_25',
       '2001_06_10', '2001_06_26', '2001_07_12', '2001_07_28',
       '2001_08_13', '2001_08_29', '2001_09_14', '2001_09_30',
       '2001_10_16', '2001_11_01', '2001_11_17', '2001_12_03',
       '2001_12_19', '2002_01_01', '2002_01_17', '2002_02_02',
       '2002_02_18', '2002_03_06', '2002_03_22', '2002_04_07',
       '2002_04_23', '2002_05_09', '2002_05_25', '2002_06_10',
       '2002_06_26', '2002_07_12', '2002_07_28', '2002_08_13',
       '2002_08_29', '2002_09_14', '2002_09_30', '2002_10_16',
       '2002_11_01', '2002_11_17', '2002_12_03', '2002_12_19',
       '2003_01_01', '2003_01_17', '2003_02_02', '2003_02_18',
       '2003_03_06', '2003_03_22', '2003_04_07', '2003_04_23',
       '2003_05_09', '2003_05_25', '2003_06_10', '2003_06_26',
       '2003_07_12', '2003_07_28', '2003_08_13', '2003_08_29',
       '2003_09_14', '2003_09_30', '2003_10_16', '2003_11_01',
       '2003_11_17', '2003_12_03', '2003_12_19', '2004_01_01',
       '2004_01_17', '2004_02_02', '2004_02_18', '2004_03_05',
       '2004_03_21', '2004_04_06', '2004_04_22', '2004_05_08',
       '2004_05_24', '2004_06_09', '2004_06_25', '2004_07_11',
       '2004_07_27', '2004_08_12', '2004_08_28', '2004_09_13',
       '2004_09_29', '2004_10_15', '2004_10_31', '2004_11_16',
       '2004_12_02', '2004_12_18', '2005_01_01', '2005_01_17',
       '2005_02_02', '2005_02_18', '2005_03_06', '2005_03_22',
       '2005_04_07', '2005_04_23', '2005_05_09', '2005_05_25',
       '2005_06_10', '2005_06_26', '2005_07_12', '2005_07_28',
       '2005_08_13', '2005_08_29', '2005_09_14', '2005_09_30',
       '2005_10_16', '2005_11_01', '2005_11_17', '2005_12_03',
       '2005_12_19', '2006_01_01', '2006_01_17', '2006_02_02',
       '2006_02_18', '2006_03_06', '2006_03_22', '2006_04_07',
       '2006_04_23', '2006_05_09', '2006_05_25', '2006_06_10',
       '2006_06_26', '2006_07_12', '2006_07_28', '2006_08_13',
       '2006_08_29', '2006_09_14', '2006_09_30', '2006_10_16',
       '2006_11_01', '2006_11_17', '2006_12_03', '2006_12_19',
       '2007_01_01', '2007_01_17', '2007_02_02', '2007_02_18',
       '2007_03_06', '2007_03_22', '2007_04_07', '2007_04_23',
       '2007_05_09', '2007_05_25', '2007_06_10', '2007_06_26',
       '2007_07_12', '2007_07_28', '2007_08_13', '2007_08_29',
       '2007_09_14', '2007_09_30', '2007_10_16', '2007_11_01',
       '2007_11_17', '2007_12_03', '2007_12_19', '2008_01_01',
       '2008_01_17', '2008_02_02', '2008_02_18', '2008_03_05',
       '2008_03_21', '2008_04_06', '2008_04_22', '2008_05_08',
       '2008_05_24', '2008_06_09', '2008_06_25', '2008_07_11',
       '2008_07_27', '2008_08_12', '2008_08_28', '2008_09_13',
       '2008_09_29', '2008_10_15', '2008_10_31', '2008_11_16',
       '2008_12_02', '2008_12_18', '2009_01_01', '2009_01_17',
       '2009_02_02', '2009_02_18', '2009_03_06', '2009_03_22',
       '2009_04_07', '2009_04_23', '2009_05_09', '2009_05_25',
       '2009_06_10', '2009_06_26', '2009_07_12', '2009_07_28',
       '2009_08_13', '2009_08_29', '2009_09_14', '2009_09_30',
       '2009_10_16', '2009_11_01', '2009_11_17', '2009_12_03',
       '2009_12_19', '2010_01_01', '2010_01_17', '2010_02_02',
       '2010_02_18', '2010_03_06', '2010_03_22', '2010_04_07',
       '2010_04_23', '2010_05_09', '2010_05_25', '2010_06_10',
       '2010_06_26', '2010_07_12', '2010_07_28', '2010_08_13',
       '2010_08_29', '2010_09_14', '2010_09_30', '2010_10_16',
       '2010_11_01', '2010_11_17', '2010_12_03', '2010_12_19',
       '2011_01_01', '2011_01_17', '2011_02_02', '2011_02_18',
       '2011_03_06', '2011_03_22', '2011_04_07', '2011_04_23',
       '2011_05_09', '2011_05_25', '2011_06_10', '2011_06_26',
       '2011_07_12', '2011_07_28', '2011_08_13', '2011_08_29',
       '2011_09_14', '2011_09_30', '2011_10_16', '2011_11_01',
       '2011_11_17', '2011_12_03', '2011_12_19', '2012_01_01',
       '2012_01_17', '2012_02_02', '2012_02_18', '2012_03_05',
       '2012_03_21', '2012_04_06', '2012_04_22', '2012_05_08',
       '2012_05_24', '2012_06_09', '2012_06_25', '2012_07_11',
       '2012_07_27', '2012_08_12', '2012_08_28', '2012_09_13',
       '2012_09_29', '2012_10_15', '2012_10_31', '2012_11_16',
       '2012_12_02', '2012_12_18', '2013_01_01', '2013_01_17',
       '2013_02_02', '2013_02_18', '2013_03_06', '2013_03_22',
       '2013_04_07', '2013_04_23', '2013_05_09', '2013_05_25',
       '2013_06_10', '2013_06_26', '2013_07_12', '2013_07_28',
       '2013_08_13', '2013_08_29', '2013_09_14', '2013_09_30',
       '2013_10_16', '2013_11_01', '2013_11_17', '2013_12_03',
       '2013_12_19', '2014_01_01', '2014_01_17', '2014_02_02',
       '2014_02_18', '2014_03_06', '2014_03_22', '2014_04_07',
       '2014_04_23', '2014_05_09', '2014_05_25', '2014_06_10',
       '2014_06_26', '2014_07_12', '2014_07_28', '2014_08_13',
       '2014_08_29', '2014_09_14', '2014_09_30', '2014_10_16',
       '2014_11_01', '2014_11_17', '2014_12_03', '2014_12_19',
       '2015_01_01', '2015_01_17', '2015_02_02', '2015_02_18',
       '2015_03_06', '2015_03_22', '2015_04_07', '2015_04_23',
       '2015_05_09', '2015_05_25', '2015_06_10', '2015_06_26',
       '2015_07_12', '2015_07_28', '2015_08_13', '2015_08_29',
       '2015_09_14', '2015_09_30', '2015_10_16', '2015_11_01',
       '2015_11_17', '2015_12_03', '2015_12_19', '2016_01_01',
       '2016_01_17', '2016_02_02', '2016_02_18', '2016_03_05',
       '2016_03_21', '2016_04_06', '2016_04_22', '2016_05_08',
       '2016_05_24', '2016_06_09', '2016_06_25', '2016_07_11',
       '2016_07_27', '2016_08_12', '2016_08_28', '2016_09_13',
       '2016_09_29', '2016_10_15', '2016_10_31', '2016_11_16',
       '2016_12_02', '2016_12_18', '2017_01_01', '2017_01_17',
       '2017_02_02', '2017_02_18', '2017_03_06', '2017_03_22',
       '2017_04_07', '2017_04_23', '2017_05_09', '2017_05_25',
       '2017_06_10', '2017_06_26', '2017_07_12', '2017_07_28',
       '2017_08_13', '2017_08_29', '2017_09_14', '2017_09_30',
       '2017_10_16', '2017_11_01', '2017_11_17', '2017_12_03',
       '2017_12_19', '2018_01_01', '2018_01_17', '2018_02_02',
       '2018_02_18', '2018_03_06', '2018_03_22', '2018_04_07',
       '2018_04_23', '2018_05_09', '2018_05_25', '2018_06_10',
       '2018_06_26', '2018_07_12', '2018_07_28', '2018_08_13',
       '2018_08_29', '2018_09_14', '2018_09_30', '2018_10_16',
       '2018_11_01', '2018_11_17', '2018_12_03', '2018_12_19',
       '2019_01_01', '2019_01_17', '2019_02_02', '2019_02_18',
       '2019_03_06', '2019_03_22', '2019_04_07', '2019_04_23',
       '2019_05_09', '2019_05_25', '2019_06_10', '2019_06_26',
       '2019_07_12', '2019_07_28', '2019_08_13', '2019_08_29',
       '2019_09_14', '2019_09_30', '2019_10_16', '2019_11_01',
       '2019_11_17', '2019_12_03', '2019_12_19', '2020_01_01',
       '2020_01_17', '2020_02_02', '2020_02_18', '2020_03_05',
       '2020_03_21', '2020_04_06', '2020_04_22', '2020_05_08',
       '2020_05_24', '2020_06_09', '2020_06_25', '2020_07_11',
       '2020_07_27', '2020_08_12', '2020_08_28', '2020_09_13',
       '2020_09_29', '2020_10_15', '2020_10_31', '2020_11_16',
       '2020_12_02', '2020_12_18', '2021_01_01', '2021_01_17',
       '2021_02_02', '2021_02_18', '2021_03_06', '2021_03_22',
       '2021_04_07', '2021_04_23', '2021_05_09', '2021_05_25',
       '2021_06_10', '2021_06_26', '2021_07_12', '2021_07_28',
       '2021_08_13', '2021_08_29', '2021_09_14', '2021_09_30',
       '2021_10_16', '2021_11_01', '2021_11_17', '2021_12_03',
       '2021_12_19', '2022_01_01', '2022_01_17', '2022_02_02',
       '2022_02_18', '2022_03_06', '2022_03_22', '2022_04_07',
       '2022_04_23', '2022_05_09', '2022_05_25', '2022_06_10',
       '2022_06_26', '2022_07_12', '2022_07_28', '2022_08_13',
       '2022_08_29', '2022_09_14', '2022_09_30', '2022_10_16',
       '2022_11_01', '2022_11_17', '2022_12_03', '2022_12_19',
       '2023_01_01', '2023_01_17', '2023_02_02', '2023_02_18',
       '2023_03_06', '2023_03_22', '2023_04_07', '2023_04_23',
       '2023_05_09', '2023_05_25', '2023_06_10', '2023_06_26',
       '2023_07_12', '2023_07_28', '2023_08_13', '2023_08_29',
       '2023_09_14', '2023_09_30', '2023_10_16', '2023_11_01',
       '2023_11_17', '2023_12_03', '2023_12_19']
        table_columns_quoted = [f'"{col}"' for col in table_columns]
        copy_command = f"""
        COPY spei_3_average12_evi ({','.join(table_columns_quoted)}) FROM '{csv_path}' WITH (FORMAT CSV, HEADER);
        """
        cur.execute(copy_command)
        conn.commit()
        cur.close()
        conn.close()
class RS_GEE():
    def __init__(self):

        ee.Authenticate()
        ee.Initialize(project='ee-')
        geemap.ee_initialize()
        self.european_shp = r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection/European_ExportFeatures_ExportFeatures.shp'
        self.MODIS_save_path = r'D:\Data Collection\RS\MODIS\NDVI'
        self.MODIS_GEE = 'MODIS/061/MOD13Q1'
        self.MODIS_EVI_save_path = r'D:\Data Collection\RS\MODIS\EVI'
        self.MODIS_phenology = 'MODIS/061/MCD12Q2'
        self.MODIS_phenology_save_path = r'D:\Data Collection\RS\MODIS\phenology_new'
        self.MODIS_phenology_land_cover_path = r'D:\Data Collection\RS\MODIS\land_cover_type'
        self.DEM_path = r'D:\Data Collection\DEM'
        self.MODIS_land_cover_type = 'MODIS/061/MCD12Q1'
        self.DEM = 'COPERNICUS/DEM/GLO30'

    def split_rectangle(self,coords, num_cols=5, num_rows=5):
        """
        将矩形框等分成指定数量的部分，并返回每个部分的边界坐标。

        参数：
        coords (list): 矩形框的边界坐标，格式为[[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x1, y1]]。
        num_cols (int): 划分的列数，默认为6。
        num_rows (int): 划分的行数，默认为6。

        返回：
        sliced_boundaries (list): 包含划分后每个部分的边界坐标的列表。
        """
        # 将坐标转换为 NumPy 数组
        coords_np = np.array(coords)

        # 计算边界框的长宽
        width = np.abs(coords_np[1][0] - coords_np[0][0])
        height = np.abs(coords_np[3][1] - coords_np[0][1])

        # 确定水平和垂直方向上的步长
        step_x = width / num_cols
        step_y = height / num_rows

        # 存储切分后的边界
        sliced_boundaries = []

        # 切分边界框
        for i in range(num_cols):
            for j in range(num_rows):
                min_x = coords_np[0][0] + i * step_x
                max_x = coords_np[0][0] + (i + 1) * step_x
                min_y = coords_np[0][1] + j * step_y
                max_y = coords_np[0][1] + (j + 1) * step_y
                sliced_boundaries.append(
                    [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y], [min_x, min_y]])

        return sliced_boundaries

    # def download_roi(self,index, roi_split, collection, out_dir):
    #     geemap.ee_export_image_collection(collection, out_dir=out_dir, region=roi_split)
    #     print(f"Downloaded images for ROI {index}")

    def download_MODIS_NDVI(self,start_date,end_date):
        collection = (
            ee.ImageCollection(self.MODIS_GEE)
            .select('NDVI')
            .filterDate(start_date, end_date)
        )
        roi = geemap.shp_to_ee(self.european_shp).geometry().bounds().coordinates().getInfo()[0]
        roi_splits = self.split_rectangle(roi)
        for index,roi_split in tqdm(enumerate(roi_splits)):
            out_dir = os.path.join(self.MODIS_save_path,str(index))
            geemap.ee_export_image_collection(collection, out_dir=out_dir,region=roi_split)
    # def download_MODIS_NDVI(self,start_date,end_date):
    #     collection = (
    #         ee.ImageCollection(self.MODIS_NDVI_GEE)
    #         .select('NDVI')
    #         .filterDate(start_date, end_date)
    #     )
    #     roi = geemap.shp_to_ee(self.european_shp).geometry().bounds().coordinates().getInfo()[0]
    #     roi_splits = self.split_rectangle(roi)
    #     # 使用多进程处理下载
    #     with Pool(processes=4) as pool:  # 设置进程数为4，根据系统资源进行调整
    #         for index, roi_split in enumerate(roi_splits):
    #             out_dir = os.path.join(self.MODIS_save_path, str(index))
    #             pool.apply_async(self.download_roi, (index, roi_split, collection, out_dir))

        # pool.close()
        # pool.join()
        # print("All processes completed.")
    def download_MODIS_EVI(self,start_date,end_date):
        collection = (
            ee.ImageCollection(self.MODIS_GEE)
            .select('EVI')
            .filterDate(start_date, end_date)
        )
        roi = geemap.shp_to_ee(self.european_shp).geometry().bounds().coordinates().getInfo()[0]
        roi_splits = self.split_rectangle(roi)
        for index,roi_split in tqdm(enumerate(roi_splits)):
            out_dir = os.path.join(self.MODIS_EVI_save_path,str(index))
            geemap.ee_export_image_collection(collection, out_dir=out_dir,region=roi_split)
    def download_MODIS_phenology(self,start_date,end_date):
        collection = (
            ee.ImageCollection(self.MODIS_phenology)
            .select(['Greenup_1','Greenup_2','Dormancy_1','Dormancy_2'])
            .filterDate(start_date, end_date)
        )
        roi = geemap.shp_to_ee(self.european_shp).geometry().bounds().coordinates().getInfo()[0]
        roi_splits = self.split_rectangle(roi)
        for index,roi_split in tqdm(enumerate(roi_splits)):
            out_dir = os.path.join(self.MODIS_phenology_save_path,str(index))
            geemap.ee_export_image_collection(collection, out_dir=out_dir,region=roi_split)

    def download_MODIS_land_cover(self,start_date,end_date):
        collection = (
            ee.ImageCollection(self.MODIS_land_cover_type)
            .select(['LC_Type1','LC_Type2'])
            .filterDate(start_date, end_date)
        )
        roi = geemap.shp_to_ee(self.european_shp).geometry().bounds().coordinates().getInfo()[0]
        roi_splits = self.split_rectangle(roi)
        for index,roi_split in tqdm(enumerate(roi_splits)):
            out_dir = os.path.join(self.MODIS_phenology_land_cover_path,str(index))
            geemap.ee_export_image_collection(collection, out_dir=out_dir,region=roi_split)

    def download_DEM(self,):
        collection = (
            ee.ImageCollection(self.DEM)
            .select(['DEM'])
        )
        roi = geemap.shp_to_ee(self.european_shp).geometry().bounds().coordinates().getInfo()[0]
        roi_splits = self.split_rectangle(roi,num_cols=100,num_rows=100)
        for index,roi_split in tqdm(enumerate(roi_splits)):
            out_dir = os.path.join(self.DEM_path,str(index))
            geemap.ee_export_image_collection(collection, out_dir=out_dir,region=roi_split)

    def day_of_year(self,date):
        return (date - datetime(date.year, 1, 1)).days + 1
    def convert_to_date(self,days_since_1970):
        date = datetime(1970, 1, 1) + timedelta(days=int(days_since_1970))
        days = self.day_of_year(date)
        return days


    def MODIS_phenology_to_normal_date(self,tif_path,basic_date = datetime(1970, 1, 1)):

        data_profile = None
        with rasterio.open(tif_path) as data:
            greenup_1 = data.read(1)
            greenup_2 = data.read(2)
            dormancy_1 = data.read(3)
            dormancy_2 =  data.read(4)
            data_profile = data.profile
        dates_greenup1 = np.vectorize(self.convert_to_date)(greenup_1)
        dates_greenup2 = np.vectorize(self.convert_to_date)(greenup_2)
        dates_dormancy1 = np.vectorize(self.convert_to_date)(dormancy_1)
        dates_dormancy2 = np.vectorize(self.convert_to_date)(dormancy_2)

        tif_path_days_pre = os.path.split(tif_path)[0]
        tif_path_days_post = os.path.split(tif_path)[1][:-4]+'_days'+'.tif'
        tif_path_days = os.path.join(tif_path_days_pre,tif_path_days_post)

        with rasterio.open(tif_path_days,'w',**data_profile) as dst:
            dst.write(dates_greenup1,1)
            dst.write(dates_greenup2,2)
            dst.write(dates_dormancy1,3)
            dst.write(dates_dormancy2,4)
    def MODIS_phenology_spatial_interpolation(self,original_tif_path):

        tif_areas = os.listdir(original_tif_path)
        original_tif_paths = []
        for area in tif_areas:
            area_tifs = glob.glob(os.path.join(os.path.join(original_tif_path, area), '*days.tif'))
            for area_tif_path in area_tifs:
                original_tif_paths.append(area_tif_path)

        for tif_path in tqdm(original_tif_paths):
            target_tif_path = os.path.join(
                r'D:\Data Collection\RS\MODIS\EVI/' + re.search(r'phenology\\(\d+)', tif_path).group(1), '2000_12_18.tif')
            target_tif = gdal.Open(target_tif_path)
            target_cols = target_tif.RasterXSize
            target_rows = target_tif.RasterYSize
            t_proj = target_tif.GetProjection()
            t_geotransform = target_tif.GetGeoTransform()
            ttif_x_min = t_geotransform[0]
            ttif_y_max = t_geotransform[3]
            ttif_x_res = t_geotransform[1]
            ttif_y_res = t_geotransform[5]

            original_tif = gdal.Open(tif_path)
            proj = original_tif.GetProjection()
            geotransform = original_tif.GetGeoTransform()

            output_tif_pre = os.path.split(tif_path)[0]
            output_tif_post = os.path.split(tif_path)[1][:-4]+'_interpolation'+'.tif'
            output_tif = os.path.join(output_tif_pre,output_tif_post)
            # 创建输出
            driver = gdal.GetDriverByName('GTiff')
            dst_ds = driver.Create(output_tif,target_cols,target_rows,original_tif.RasterCount,original_tif.GetRasterBand(1).DataType)
            dst_transform = (ttif_x_min,ttif_x_res,0,ttif_y_max,0,ttif_y_res)
            dst_ds.SetGeoTransform(dst_transform)
            dst_ds.SetProjection(t_proj)
            gdal.ReprojectImage(original_tif,dst_ds,t_proj,t_proj,gdal.GRA_NearestNeighbour)
            # for i in range(1,original_tif.RasterCount + 1):
            #     original_data = original_tif.GetRasterBand(i).ReadAsArray()
            #     # dst_data = gdal.RegenerateOverview(original_data, (target_cols, target_rows),gdal.GRA_Bilinear)
            #     # 使用scipy.ndimage.zoom进行双线性插值
            #     # order=1 表示使用双线性插值
            #     dst_data = zoom(original_data,2,1)
            #     # 将插值后的数据写入输出数据集
            #     dst_ds.GetRasterBand(i).WriteArray(dst_data)
            original_tif = None
            dst_ds = None


    def merge_tifs(self,file_path_list,output_path):
        # 用于合并tif文件的命令
        vrt_options = gdal.BuildVRTOptions(addAlpha=True)  # 去掉resampleAlg参数
        vrt = gdal.BuildVRT('/vsimem/merged.vrt', file_path_list, options=vrt_options)

        # 将虚拟VRT文件转换为实际的tif文件
        gdal.Translate(output_path, vrt)

    def merge_tifs_with_nan_handling(self,tif_paths, output_path):
        """
        合并多个 TIFF 文件，并在重叠区域优先保留有效数据。
        如果所有 TIFF 在某区域都为 NaN，则合并后该区域也为 NaN。
        # 注意：！！！！！！！！虽然能用，但是不知道怎么用的，未来只能用于可视化，不能用于分析数据！！！！！！！！！！！！！
        参数：
        tif_paths: list of str
            要合并的 TIFF 文件路径列表。
        output_path: str
            输出合并后的 TIFF 文件路径。
        """
        # 读取所有 TIFF 文件
        src_files_to_mosaic = []
        for fp in tif_paths:
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)

        # 使用 rasterio 的 merge 函数进行初步合并
        mosaic, out_trans = merge(src_files_to_mosaic, nodata=np.nan)

        # 处理重叠区域，确保仅在所有输入为 NaN 时，输出为 NaN
        combined_data = np.full_like(mosaic[0], np.nan)  # 初始化为全 NaN
        for layer in mosaic:
            # 对每一层数据，用 `np.where` 替换 NaN 区域的值
            combined_data = np.where(np.isnan(combined_data), layer, combined_data)

        # 获取第一个栅格文件的元数据作为基础
        out_meta = src_files_to_mosaic[0].meta.copy()

        # 更新输出元数据，以匹配合并后的数据集
        out_meta.update({
            "driver": "GTiff",
            "height": combined_data.shape[0],
            "width": combined_data.shape[1],
            "transform": out_trans,
            "nodata": np.nan
        })

        # 输出合并后的文件
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(combined_data, 1)

        # 关闭所有输入文件
        for src in src_files_to_mosaic:
            src.close()

        print(f"合并后的文件已保存到: {output_path}")
    def reproject_modis(self,input_file, output_file, target_projection='EPSG:4326'):
        subprocess.run([
            'gdalwarp',
            '-t_srs', target_projection,
            input_file, output_file
        ])
        print(f"Reprojected file saved as {output_file}")
    def MODIS_data_merge(self,path,data_type):

        if data_type == 'indices': file_post = '*reprojection.tif'
        elif data_type == 'phenology_days_reprojection': file_post = '*days_reprojection.tif'
        else: file_post = '*days.tif'

        plot_list = os.listdir(path)

        plot_path = os.path.join(path,plot_list[0])
        tiffs_current_plot = glob.glob(os.path.join(plot_path,file_post))
        for tiff_current_plot in tqdm(tiffs_current_plot):
            tiff_current_plot_name = os.path.split(tiff_current_plot)[-1]
            tiff_ID_paths = [os.path.join(path, plot_) for plot_ in plot_list]
            tiff_ID_paths_full = [os.path.join(item,tiff_current_plot_name) for item in tiff_ID_paths]

            outputpath = os.path.split(os.path.split(tiff_current_plot)[0])[0]
            make_dir(outputpath + '_merge/')
            outputpath = outputpath + '_merge/' + tiff_current_plot_name
            if os.path.exists(outputpath):continue
            self.merge_tifs(tiff_ID_paths_full,outputpath)

    def MODIS_reprojection(self,path,data_type):
        if data_type == 'indices': file_post = '*.tif'
        elif data_type == 'land cover': file_post = '*.tif'
        else: file_post = '*days.tif'

        plot_list = os.listdir(path)

        for plot in plot_list:
            plot_path = os.path.join(path,plot)
            tiffs_current_plot = glob.glob(os.path.join(plot_path,file_post))
            for tif_plot in tqdm(tiffs_current_plot):
                # if 'interpolation' in tif_plot or 'reprojection' in tif_plot or 'days' in tif_plot: continue
                # output_file_path = os.path.join(os.path.split(tif_plot)[0],
                #                                 os.path.split(tif_plot)[-1].replace('.tif',
                #                                                                     '_reprojection240611.tif'))
                # output_file_path = os.path.join(os.path.split(tif_plot)[0],os.path.split(tif_plot)[-1].replace('days.tif','days_reprojection.tif'))
                output_file_path = os.path.join(os.path.split(tif_plot)[0],os.path.split(tif_plot)[-1].replace('.tif','_reprojection.tif'))
                if not os.path.exists(output_file_path):
                    self.reproject_modis(tif_plot,output_file_path)

    def tif_to_nc(self,path,file_back,plot):

        output_nc_file = os.path.join(path, file_back + '_summary.nc')
        if not os.path.exists(output_nc_file):
            files = glob.glob(os.path.join(path,'*'+file_back+'.tif'))
            # 定义存储波段数据的列表
            band1_data = []
            band3_data = []
            for tiff in files:
                raster = rioxarray.open_rasterio(tiff)
                band1_data.append(raster.sel(band=1).drop_vars('band'))
                band3_data.append(raster.sel(band=3).drop_vars('band'))
            # 将所有波段数据组合到一个xarray数据集中
            combined_band1 = xr.concat(band1_data, dim='time')
            combined_band3 = xr.concat(band3_data, dim='time')
            # 创建一个包含两个波段的xarray数据集
            ds = xr.Dataset({
                'sos': combined_band1,
                'eos': combined_band3
            })

            ds.to_netcdf(output_nc_file)

    def indices_annual_aggregation(self,files,year):
        files_year = [f for f in files if str(year) in f]
        year_data = []
        for file in files_year:
            with rasterio.open(file) as src:
                data = src.read(1)*0.0001
                data[data == -0.3] = np.nan
                year_data.append(data)
        year_array = np.array(year_data)
        mean_array = np.nanmean(year_array,axis=0)
        output_path = os.path.join(os.path.split(file)[0],f'{year}_annualmean.tif')
        with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=mean_array.shape[0],
                width=mean_array.shape[1],
                count=1,
                dtype=mean_array.dtype,
                crs=src.crs,
                transform=src.transform
        ) as dst:
            dst.write(mean_array, 1)


class Meteorological_Data:
    def __init__(self):
        pass

    def copernicus_temperature_download(self):
        c = cdsapi.Client()
        years = [str(year) for year in range(2022, 2024)]
        for year in tqdm(years):
            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': '2m_temperature',
                    'year': year,
                    'month': ['01','02','03','04','05','06','07','08','09','10','11','12'],
                    'day': [
                        '01', '02', '03',
                        '04', '05', '06',
                        '07', '08', '09',
                        '10', '11', '12',
                        '13', '14', '15',
                        '16', '17', '18',
                        '19', '20', '21',
                        '22', '23', '24',
                        '25', '26', '27',
                        '28', '29', '30',
                        '31',
                    ],
                    'time': [
                        '00:00', '01:00', '02:00',
                        '03:00', '04:00', '05:00',
                        '06:00', '07:00', '08:00',
                        '09:00', '10:00', '11:00',
                        '12:00', '13:00', '14:00',
                        '15:00', '16:00', '17:00',
                        '18:00', '19:00', '20:00',
                        '21:00', '22:00', '23:00',
                    ],
                    'format': 'netcdf.zip',
                    'area': [
                        71.11, -11.98, 35.42,
                        32.24,
                    ],
                },
                '{}.netcdf.zip'.format(year))
    def copernicus_precip_pev_download2(self):
        c = cdsapi.Client()
        years = [str(year) for year in range(1950, 2000)]
        variables = ['potential_evaporation', 'total_precipitation']
        for variable in variables:
            c.retrieve(
                'reanalysis-era5-land-monthly-means',
                {
                    'product_type':'monthly_averaged_reanalysis',
                    'variable':  variable,
                    'year': years,
                    'month': ['01','02','03','04','05','06','07','08','09','10','11','12'],
                    'format': 'netcdf.zip',
                    'time': '00:00',
                    'area': [
                        71.11, -11.98, 35.42,
                        32.24,
                    ],
                },
                os.path.join(r'D:\Data Collection\SPEI_variables','{}.netcdf.zip'.format(variable)))
    def copernicus_precip_pev_download(self):
        c = cdsapi.Client()
        years = [str(year) for year in range(1950, 2000)]
        variables = ['potential_evaporation', 'total_precipitation']
        for variable in variables:
            for year in tqdm(years):
                c.retrieve(
                    'reanalysis-era5-land-monthly-means',
                    {
                        'product_type':'monthly_averaged_reanalysis',
                        'variable':  variable,
                        'year': year,
                        'month': ['01','02','03','04','05','06','07','08','09','10','11','12'],
                        'format': 'netcdf.zip',
                        'time': '00:00',
                        'area': [
                            71.11, -11.98, 35.42,
                            32.24,
                        ],
                    },
                    os.path.join(r'D:\Data Collection\SPEI_variables','{}_{}.netcdf.zip'.format(variable,year)))
    def copernicus_temperaturemonthly_download(self):
        years = list(range(2023, 2024))
        years_str = [str(year) for year in years]
        c = cdsapi.Client()
        c.retrieve(
            'reanalysis-era5-land-monthly-means',
            {
                'product_type':'monthly_averaged_reanalysis',
                'variable':  '2m_temperature',
                'year': years_str,
                'month': ['01','02','03','04','05','06','07','08','09','10','11','12'],
                'format': 'netcdf.zip',
                'time': '00:00',
                'area': [
                    71.11, -11.98, 35.42,
                    32.24,
                ],
            },
            'monthly_temperature.netcdf.zip')

    def merge_t2m_data(self,path):
        years = os.listdir(path)
        for year in years:
            files = os.listdir(os.path.join(path,year))
            data_lists = [os.path.join(os.path.join(path,year),file+'/data_0.nc') for file in files]
            data = xr.open_mfdataset(data_lists)
            data = data.rename({'valid_time': 'time'})
            data.to_netcdf(f'{year}_t2m.nc')
    def copernicus_dewpoint_temperaturemonthly_download(self):
        c = cdsapi.Client()
        c.retrieve(
            'reanalysis-era5-land-monthly-means',
            {
                'product_type':'monthly_averaged_reanalysis',
                'variable':  '2m_dewpoint_temperature',
                # 'year': ['2000', '2001', '2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022'],
                'year': ['2023'],
                'month': ['01','02','03','04','05','06','07','08','09','10','11','12'],
                'format': 'netcdf.zip',
                'time': '00:00',
                'area': [
                    71.11, -11.98, 35.42,
                    32.24,
                ],
            },
            'monthly_dewpoint_temperature.netcdf.zip')
    def GDD_calculation(self,path, base_temperature=5.0):

        temperature_data = xr.open_dataset(path)
        # 提取时间、纬度和经度
        time = temperature_data['time']
        latitude = temperature_data['latitude']
        longitude = temperature_data['longitude']

        # 提取温度数据（单位为K，需要转换为摄氏度）
        temperature_k = temperature_data['t2m']
        temperature_c = temperature_k - 273.15  # 转换为摄氏度

        # 将日期转换为 pandas DatetimeIndex 以确保正确的 dtype
        unique_dates = pd.to_datetime(time.dt.date.values).unique()
        # 创建一个空的 DataArray 用于存储每日 GDD
        daily_gdd = xr.DataArray(
            np.zeros((len(unique_dates), len(latitude), len(longitude))),
            dims=["day", "latitude", "longitude"],
            coords={"day": unique_dates, "latitude": latitude, "longitude": longitude},
            name="GDD"
        )

        # 按天分组数据
        for date, daily_temp in temperature_c.groupby(time.dt.date):
            # 转换 date 为 Timestamp 以确保与坐标一致
            date = pd.Timestamp(date)
            # 每日最大温度和最小温度
            daily_max_temp = daily_temp.max(dim="time")
            daily_min_temp = daily_temp.min(dim="time")

            # 计算每日平均温度
            daily_mean_temp = (daily_max_temp + daily_min_temp) / 2

            # 计算 GDD
            daily_gdd_value = daily_mean_temp - base_temperature

            # 设置低于基准温度的 GDD 为 0
            daily_gdd_value = daily_gdd_value.where(daily_gdd_value > 0, 0)

            # 存储结果
            daily_gdd.loc[date] = daily_gdd_value
        pattern = r'(?<=\\|/)\d{4}(?=\.netcdf)'
        year = re.search(pattern, path).group()

        daily_gdd.to_netcdf('{}_GDD.nc'.format(year))

    def chillingdays_calculation(self,path, base_temperature=5.0):

        temperature_data = xr.open_dataset(path)
        # 提取时间、纬度和经度
        time = temperature_data['time']
        latitude = temperature_data['latitude']
        longitude = temperature_data['longitude']

        # 提取温度数据（单位为K，需要转换为摄氏度）
        temperature_k = temperature_data['t2m']
        temperature_c = temperature_k - 273.15  # 转换为摄氏度

        # 将日期转换为 pandas DatetimeIndex 以确保正确的 dtype
        unique_dates = pd.to_datetime(time.dt.date.values).unique()
        # 创建一个空的 DataArray 用于存储每日 GDD
        daily_chilling = xr.DataArray(
            np.zeros((len(unique_dates), len(latitude), len(longitude))),
            dims=["day", "latitude", "longitude"],
            coords={"day": unique_dates, "latitude": latitude, "longitude": longitude},
            name="chilling"
        )

        # 按天分组数据
        for date, daily_temp in temperature_c.groupby(time.dt.date):
            # 转换 date 为 Timestamp 以确保与坐标一致
            date = pd.Timestamp(date)
            # 每日最大温度和最小温度
            daily_max_temp = daily_temp.max(dim="time")
            daily_min_temp = daily_temp.min(dim="time")

            # 计算每日平均温度
            daily_mean_temp = (daily_max_temp + daily_min_temp) / 2
            # 计算 chilling
            daily_chilling_value = xr.where(daily_mean_temp.isnull(), np.nan, xr.where(daily_mean_temp < base_temperature, 1, 0))
            # 存储结果
            daily_chilling.loc[date] = daily_chilling_value

        pattern = r'(?<=\\|/)\d{4}(?=\.netcdf)'
        year = re.search(pattern, path).group()

        daily_chilling.to_netcdf('{}_chilling.nc'.format(year))
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
        return data_aggregation
    def VPD_calculation(self,Ta_path,Td_path,DEM_path):
        Ta_data =xr.open_dataset(Ta_path)
        Td_data = xr.open_dataset(Td_path)
        DEM_data = self.tif_tonc(dem_path)
        # DEM_data =gdal.Open(DEM_path)
        # DEM_proj = DEM_data.GetProjection()
        # # # 使用rioxarray扩展打开数据集
        # Ta_data = Ta_data.rio.write_crs(DEM_proj)
        # Td_data = Td_data.rio.write_crs(DEM_proj)
        # Ta_data_reprojected = Ta_data.rio.reproject(DEM_proj)
        # Td_data_reprojected = Td_data.rio.reproject(DEM_proj)
        # Ta_data_reprojected.to_netcdf(os.path.join(os.path.split(Ta_path)[0],"reprojected_Ta_data.nc"))
        # Td_data_reprojected.to_netcdf(os.path.join(os.path.split(Td_path)[0],"reprojected_Td_data.nc"))
        temperature_k = Ta_data['t2m']
        temperature_c = temperature_k - 273.15  # 转换为摄氏度

        dewpoint_temperature_k = Td_data['d2m']
        dewpoint_temperature_c = dewpoint_temperature_k - 273.15  # 转换为摄氏度
        temperature_c = temperature_c.rename({'latitude': 'lat', 'longitude': 'lon'})
        dewpoint_temperature_c = dewpoint_temperature_c.rename({'latitude': 'lat', 'longitude': 'lon'})
        DEM_SA = self.spatial_aggregation(DEM_data,temperature_c,'mean')
        DEM_sa_data = DEM_SA['data']
        DEM_sa_data_padding = DEM_sa_data.reindex(lat=temperature_c.coords['lat'], lon=temperature_c.coords['lon'], method='nearest')
        DEM_sa_data_padding = DEM_sa_data_padding.expand_dims(time=temperature_c['time'])

        P_mst = 1013.25 * ((273.16 + temperature_c) / (273.16 + temperature_c + (0.0065*DEM_sa_data_padding))) ** 5.625
        AVP = 6.112 * (1 + (7e-4) + (3.46 * (1e-6) * P_mst)) * (np.exp((17.67*dewpoint_temperature_c)/(dewpoint_temperature_c+243.5)))
        SVP = 6.112 * (1 + (7e-4) + (3.46 * (1e-6) * P_mst)) * (np.exp((17.67*temperature_c)/(temperature_c+243.5)))
        VPD = SVP - AVP
        VPD.name = 'VPD'
        VPD.to_netcdf('VPD.nc')

    def compute_spei_for_grid(self,tp_series, pev_series):
        return indices.spei(
            precips_mm=tp_series,
            pet_mm=pev_series,
            scale=3,
            distribution=indices.Distribution.gamma,
            periodicity=compute.Periodicity.monthly,
            data_start_year=1950,
            calibration_year_initial=1950,
            calibration_year_final=2021
        )
    def compute_pet(self,ds,):
        # 提取温度数据和时间
        t2m = ds['t2m'] - 273.15  # 将温度从K转为摄氏度
        time = ds['time']
        temps = t2m.copy()
        latitudes = ds['latitude']
        times = time.copy()
        PET = xr.zeros_like(temps)  # 创建一个与温度数据相同大小的PET变量
        years = np.unique(times.dt.year)  # 获取唯一的年份
        lat_rad = np.deg2rad(latitudes)  # 将纬度转换为弧度，用于日照修正

        for year in years:
            # 选取当前年份的数据
            temps_year = temps.sel(time=str(year))
            months = np.arange(1, 13)

            # 计算每年的温度指数 I
            I = ((temps_year.groupby('time.month').mean(dim='time') / 5) ** 1.514).sum(dim='month')

            # 计算经验参数 a
            a = 6.75e-7 * I ** 3 - 7.71e-5 * I ** 2 + 1.792e-2 * I + 0.49239

            # 计算每个月的PET
            for month in months:
                temp_month = temps_year.sel(time=f'{year}-{month:02d}')  # 选择每个月的数据
                PET_month = 16 * ((10 * temp_month / I) ** a)
                PET_month = PET_month.where(temp_month > 0, 0)  # 如果温度<=0, PET = 0

                # 将每个月的PET结果放入PET矩阵中
                PET.loc[{'time': f'{year}-{month:02d}'}] = PET_month

        return PET.rename('pet')
    def SPEI_calculation(self,precips_path,tm_path):

        years = list(range(1950,2022))
        sorted_tp =[]
        sorted_pev = []    #按时间排序
        for year in years:
            sorted_tp.append(os.path.join(precips_path,f'total_precipitation_{year}.netcdf/data.nc'))
            sorted_pev.append(os.path.join(pet_path, f'potential_evaporation_{year}.netcdf/data.nc'))
        tp = xr.open_mfdataset(sorted_tp).load()
        tm = xr.open_dataset(tm_path)
        pet = self.compute_pet(tm)
        # tp_ = (tp['tp'] * 1000).rename({'latitude': 'lat', 'longitude': 'lon'}).transpose('lat', 'lon', 'time')
        # pet_ = pet.rename({'latitude': 'lat', 'longitude': 'lon'}).transpose('lat', 'lon', 'time')
        # tp_.attrs['units'] = 'mm'
        # pet_.attrs['units'] = 'mm'
        # lons = tp['longitude'].values
        # lats = tp['latitude'].values
        # ds_tp = xr.Dataset(
        #     {
        #         "tp": (("lat", "lon", 'time'), np.zeros((len(lats), len(lons),len(tp['time'].values))))
        #     },
        #     coords={
        #         "lat": lats,
        #         "lon": lons,
        #         "time": tp['time'].values
        #     }
        # )
        #
        # for i, time_item in enumerate(tp['time'].values):
        #         new_data = (tp.sel(time=time_item)['tp'].data * 1000).copy()
        #         ds_tp['tp'].loc[dict(time=tp['time'].values[i])] = new_data
        #
        # ds_pev = xr.Dataset(
        #     {
        #         "pev": (("lat", "lon", 'time'), np.zeros((len(lats), len(lons),len(tp['time'].values))))
        #     },
        #     coords={
        #         "lat": lats,
        #         "lon": lons,
        #         "time": tp['time'].values
        #     }
        # )
        # for i, time_item in enumerate(pev['time'].values):
        #         new_data = (pev.sel(time=time_item)['pev'].data * 1000).copy()
        #         ds_pev['pev'].loc[dict(time=pev['time'].values[i])] = new_data
        # ds_tp.to_netcdf('D:\Data Collection\SPEI_variables\SPEI_calculation/tp.nc')
        # ds_pev.to_netcdf('D:\Data Collection\SPEI_variables\SPEI_calculation/pev.nc')
        tp_data = tp['tp'] * 1000

         # Initialize an array to store SPEI values
        deficit_surplus = tp_data - pet
        spei_data = xr.full_like(deficit_surplus, np.nan)
        for lat_idx in tqdm(range(tp.latitude.size)):
            for lon_idx in tqdm(range(tp.longitude.size)):
                # tp_point = tp_data[:, lat_idx, lon_idx].values  # Get precipitation at this grid point
                # pev_point = pet[:, lat_idx, lon_idx].values  # Get PET at this grid point
                # 提取每个格点的时间序列
                time_series = deficit_surplus[:, lat_idx, lon_idx].values
                # 忽略NaN值
                if np.isnan(time_series).all():
                    continue
                spei_point = indices.spei(
                    precips_mm=time_series,
                    pet_mm=np.zeros_like(time_series),
                    scale=3,
                    distribution=indices.Distribution.gamma,
                    periodicity=compute.Periodicity.monthly,
                    data_start_year=1951,  # 数据起始年份
                    calibration_year_initial=1951,  # 校准期起始年份
                    calibration_year_final=2021  # 校准期结束年份
                )

                spei_data[:, lat_idx, lon_idx] = spei_point  # Store SPEI value
        spei_data = xr.DataArray(spei_data, coords=[deficit_surplus.time, deficit_surplus.latitude, deficit_surplus.longitude], dims=["time", "latitude", "longitude"])
        spei_data.name = "spei"


if __name__ == '__main__':
    # # 遥感数据下载,处理
    rs_gee = RS_GEE()
    # pep = PEP725()
    # # 重新组织csv变成结构化数据
    # # files = glob.glob(os.path.join(pep.root_path,'*.csv'))
    # # for file in files:
    # #     pep.reorganize_data(file)
    # # 获取所有站点的分布
    # results_df_unique = pep.station_distribution()


    # 数据可视化和预处理
    # dv = Data_Visualization()
    # dv.station_temporal_visualization('webcam')

    dp = Data_Preprocess()
    # dp.station_data_filtering_by_drought_event(['2018-04','2018-12'],'station_filter_PEP11.csv')

    # 各国数据切割
    # 1. 统一投影
    # countries_path = r'D:\Data Collection\RS\Disturbance\7080016'
    # for country_path in os.listdir(countries_path):
    #     if r'.zip' in country_path:continue
    #     country_path = os.path.join(countries_path, country_path)
    #     country_tifs = glob.glob(os.path.join(country_path, '*.tif'))
    #     country_tif = [item for item in country_tifs if 'forestcover' in item][0]
    #     country_tif_reprojection_path = country_tif.replace('.tif', '_reprojection.tif')
    #     rs_gee.reproject_modis(country_tif,country_tif_reprojection_path)

    # 各国干扰数据重投影
    # country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
    #                 os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    # country_tifs = [f for path in country_tifs for f in glob.glob(f"{path}\\*disturbance_year*")]
    # country_tifs = [file for file in country_tifs if file.endswith('.tif')]
    # for country_tif in tqdm(country_tifs):
    #     rs_gee.reproject_modis(country_tif,country_tif.replace('.tif','_reprojection.tif'))

    # 计算年度植被指数均值
    # data_path = r'E:\TUM\D\Data Collection\1st drought legacy\MODIS\EVI'
    # paths = [os.path.join(data_path,item) for item in os.listdir(data_path)]
    # for path in tqdm(paths):
    #     indices_paths = glob.glob(os.path.join(path,'*.tif'))
    #     filtered_files = [f for f in indices_paths if 'reprojection' not in f]
    #     for year in range(2001,2024):
    #         rs_gee.indices_annual_aggregation(filtered_files,year)
    # 年度植被指数均值重投影
    # data_path = r'E:\TUM\D\Data Collection\1st drought legacy\MODIS\EVI'
    # paths = [os.path.join(data_path,item) for item in os.listdir(data_path)]
    # paths = [f for f in paths if 'annual' not in f]
    # path = paths[0]
    # for path in paths:
    #     indices_paths = glob.glob(os.path.join(path,'*.tif'))
    #     filtered_files = [f for f in indices_paths if 'annual' in f]
    #     for annualmean_path in tqdm(filtered_files):
    #         if not os.path.exists(annualmean_path.replace('annualmean.tif','annualmean_reprojection.tif')):
    #             rs_gee.reproject_modis(annualmean_path,annualmean_path.replace('annualmean.tif','annualmean_reprojection.tif'))
    # 合并所有地块的年度植被指数均值
    # data_path = r'E:\TUM\D\Data Collection\1st drought legacy\MODIS\EVI'
    # paths = [os.path.join(data_path,item) for item in os.listdir(data_path)]
    # paths = [f for f in paths if 'annual' not in f]
    # path = paths[0]
    # indices_paths = glob.glob(os.path.join(path,'*.tif'))
    # filtered_files = [f for f in indices_paths if 'annualmean_reprojection' in f]
    # for year_item in tqdm(filtered_files):
    #     plot_items = [os.path.join(os.path.split(os.path.split(year_item)[0])[0],plot_id+'//'+os.path.split(year_item)[-1]) for plot_id in sorted([f for f in os.listdir(data_path) if 'annual' not in f], key=lambda x: int(x))]
    #     if not os.path.exists(os.path.join(r'E:\TUM\D\Data Collection\1st drought legacy\MODIS\EVI\merge_annual',os.path.split(year_item)[-1]).replace('reprojection.tif','reprojection2.tif')):
    #         rs_gee.merge_tifs(plot_items, os.path.join(r'E:\TUM\D\Data Collection\1st drought legacy\MODIS\EVI\merge_annual',os.path.split(year_item)[-1]).replace('reprojection.tif','reprojection2.tif'))
    # 所有地块合并后的年度植被指数均值重投影
    # merge_data = r'E:\TUM\D\Data Collection\1st drought legacy\MODIS\EVI\merge_annual'
    # merge_data_paths = glob.glob(os.path.join(merge_data,'*.tif'))
    # for path in merge_data_paths:
    #     rs_gee.reproject_modis(path,os.path.join(merge_data,os.path.split(path)[-1].replace('.tif','_reprojection.tif')))
    # 各国年度均值tif切割
    # country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
    #                 os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    #
    # country_tifs = [f for path in country_tifs for f in glob.glob(f"{path}\\*forestcover*reprojection*")]
    # country_tifs = [file for file in country_tifs if file.endswith('.tif')]
    # europe_tif_paths = glob.glob(os.path.join(r'E:\TUM\D\Data Collection\1st drought legacy\MODIS\EVI\merge_annual','*.tif'))
    # europe_tif_paths = [item for item in europe_tif_paths if 'reprojection' in item]
    # for europe_tif_path in tqdm(europe_tif_paths):
    #     for country_tif in country_tifs:
    #         dp.align_data_to_country(country_tif,europe_tif_path,'EVI')
    # CSIF 计算年度均值与时序汇总
    # csif_path = r'D:\Data Collection\other_factors\all-sky'
    # output_path = r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\2 forest sensitivity to water\data collection'
    # dp.CSIF_annual_calculation(csif_path,output_path)
    # 切割tif为不同的块
    # path = r'D:\Data Collection\RS\Disturbance\7080016\unitedkingdom/Wu Yong'
    # paths = glob.glob(os.path.join(path,'mask_combined.tif'))
    # for path in paths:
    #     input_tif = path
    #     output_dir = "test"  # 替换为保存分块的目录
    #     dp.split_tif_dynamic(input_tif, output_dir)

    # 合并tif
    paths = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
             os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    indices_name = 'SIF'
    for year in tqdm(range(2001,2017)):
        files = [os.path.join(item,f'Wu Yong/{os.path.split(item)[-1]}_{year}_{indices_name}_offset_sa.tif') for item in paths if os.path.exists(os.path.join(item,f'Wu Yong/{os.path.split(item)[-1]}_{year}_{indices_name}_offset_sa.tif'))]
        os.makedirs(f'paper plots/offset map/{indices_name}', exist_ok=True)
        rs_gee.merge_tifs_with_nan_handling(files, f'paper plots/offset map/{indices_name}/{year}.tif')
    # 2. 切割物候tif，先合并所有子区域tif，然后在合并完以后进行各国tif的切割
    # path = r'D:\Data Collection\RS\MODIS\phenology'
    # rs_gee.MODIS_data_merge(path,'phenology_days_reprojection')
    # 各国物候数据tif的切割
    # country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
    #                 os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    #
    # country_tifs = [f for path in country_tifs for f in glob.glob(f"{path}\\*forestcover*reprojection*")]
    # country_tifs = [file for file in country_tifs if file.endswith('.tif')]
    # europe_tif_paths = glob.glob(os.path.join(r'D:\Data Collection\RS\MODIS\phenology_merge','*.tif'))
    # europe_tif_paths = [item for item in europe_tif_paths if '2022' in item or '2023' in item]
    # for europe_tif_path in tqdm(europe_tif_paths):
    #     for country_tif in country_tifs:
    #         dp.align_data_to_country(country_tif,europe_tif_path,'phenology')
    # 各国土地利用数据tif的切割
    # country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
    #                 os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    # country_tifs = [f for path in country_tifs for f in glob.glob(f"{path}\\*forestcover*reprojection*")]
    # country_tifs = [file for file in country_tifs if file.endswith('.tif')]
    # land_cover_tif = r'D:\Data Collection\Land Cover\land cover_100m\Results\u2018_clc2018_v2020_20u1_raster100m\u2018_clc2018_v2020_20u1_raster100m\DATA/U2018_CLC2018_V2020_20u1_reprojection_arcgispro.tif'
    # for country_tif in country_tifs:
    #     dp.align_data_to_country(country_tif,land_cover_tif,'land_cover')
    # 各国SPEI数据的切割
    '''
    2024-08-13，暂时不切割了，因为SPEI和MODIS数据分辨率差的太大，后续直接在建模的时候获取对应位置的SPEI就好了
    
    # country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
    #                 os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    # country_tifs = [f for path in country_tifs for f in glob.glob(f"{path}\\*phenology*")]
    # country_tifs = [file for file in country_tifs if file.endswith('.tif')]
    # SPEI_paths = [r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei03.nc',
    #               r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei06.nc']
    # for SPEI_path in SPEI_paths:
    #     for country_tif in country_tifs:
    #         dp.align_data_to_country(country_tif, SPEI_path)
    '''
    # 各国物候tif生成nc格式
    '''
    代码已经调试好了，可以直接用，但是通过生成结果后发现nc会存在一定的位置偏离，这个位置偏离很小，但是还是有偏离，因此暂时决定不用nc，还是使用直接读取tif的方法
    偏差大概如下
    Tif 文件地理范围: 18.928347457884325 42.696722458844 21.404179468482617 39.5682652739039
    Nc 文件地理范围: 18.930937240322187 42.699312241281866 21.40676925092039 39.57085505633973
    Tif 文件坐标系: GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]
    Nc 文件坐标系: EPSG:4326
    Tif 文件分辨率: (0.005179564875728643, -0.005179564875728643)
    Nc 文件分辨率: (0.00517956487572846, -0.005179564875732012)
    '''
    '''
    country_tifs = [os.path.join(r'D:\Data Collection\RS\Disturbance\7080016', item) for item in
                    os.listdir(r'D:\Data Collection\RS\Disturbance\7080016') if '.zip' not in item]
    for country_tif in country_tifs:
        pattern = re.compile(r"phenology\d{4}")
        matched_files = [f for f in glob.glob(f"{country_tif}\\*") if pattern.search(f)]
        matched_files = [file for file in matched_files if file.endswith('.tif')]
        dp.stack_tif_as_nc(matched_files,1)
    '''



    # # rs_gee.download_MODIS_NDVI('2000-03-01', '2024-01-01')
    # rs_gee.download_MODIS_phenology('2001-01-01', '2024-01-01')
    # rs_gee.download_MODIS_land_cover('2001-01-01', '2023-01-01')
    # rs_gee.download_DEM()
    # 物候数据处理，转为真实天数，并且插值到250m分辨率
    # 转为天数
    # tif_path = r'D:\Data Collection\RS\MODIS\phenology_new'
    # tif_areas = os.listdir(tif_path)
    # for area in tqdm(tif_areas):
    #     print('*'*10+'Processing '+area+'*'*10)
    #     area_tifs = glob.glob(os.path.join(os.path.join(tif_path,area),'*.tif'))
    #     for area_tif_path in area_tifs:
    #         if os.path.split(area_tif_path)[-1][:4] not in ['2022','2023']: continue
    #         rs_gee.MODIS_phenology_to_normal_date(area_tif_path)
    # 插值
    # rs_gee.MODIS_phenology_spatial_interpolation(r'C:\CMZ\PhD\2024\paper writing\01 Legacy effect of drought\data\phenology')


    # 合并tif
    # path = r'D:\Data Collection\RS\MODIS\phenology_new'
    # rs_gee.MODIS_data_merge(path,'phenology_days_reprojection')

    # 合并dem
    # path = r'D:\Data Collection\DEM'
    # paths = os.listdir(path)
    # paths = list(filter(lambda file: not file.endswith('.tar'), paths))
    # paths = [os.path.join(path, item) for item in paths]
    # paths = [os.path.join(item, os.listdir(item)[0]) for item in paths]
    # paths = [os.path.join(item, 'DEM') for item in paths]
    # files = [os.path.join(item, os.listdir(item)[0]) for item in paths]
    # rs_gee.merge_tifs(files,
    #                   r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\DEM')
    # 重投影
    # rs_gee.MODIS_reprojection(r'D:\Data Collection\RS\MODIS\phenology_new','phenology')

    # modis500m物候数据转成numpy
    # phenology_path = r'D:\Data Collection\RS\MODIS\phenology'
    # plots = os.listdir(phenology_path)
    # for plot in tqdm(plots):
    #     plot_path = os.path.join(phenology_path,plot)
    #     rs_gee.tif_to_nc(plot_path,'days_interpolation_reprojection',plot)
    # 数据库
    # db = Database()
    #
    # db.csv_to_postgre('paper01_legacy',r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\phenology_results\MODIS250\indices\spei_3_average12/drought_result_summary.csv')


    # 气象数据
    meteorology_data = Meteorological_Data()
    # 温度数据下载
    # meteorology_data.copernicus_temperature_download()
    # meteorology_data.copernicus_temperaturemonthly_download()
    # 露点温度下载
    # meteorology_data.copernicus_dewpoint_temperaturemonthly_download()
    # 总降水量和潜在蒸散发数据下载
    # meteorology_data.copernicus_precip_pev_download2()
    # 2023年2m温度数据合并，因为下载的时候用api太慢，所以手动从网站https://cds-beta.climate.copernicus.eu/requests?tab=all上下载，这个是单月单月下载，所以把他们合并成统一的格式
    # meteorology_data.merge_t2m_data(r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\temp\T2M')
    # GDD计算
    # t2m_path = r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\2000.netcdf/data.nc'
    # t2m_path = r'E:\TUM\D\Data Collection\1st drought legacy\Temperature'
    # for sub_path in os.listdir(t2m_path):
    #
    #     if '.zip' in sub_path: continue
    #     if '2022' in sub_path or '2023' in sub_path:
    #         sub_path = os.path.join(os.path.join(t2m_path,sub_path),'data.nc')
    #         meteorology_data.GDD_calculation(sub_path)
    # 寒冷天数计算
    t2m_path = r'E:\TUM\D\Data Collection\1st drought legacy\Temperature'
    # for sub_path in os.listdir(t2m_path):
    #
    #     if '.zip' in sub_path: continue
    #     # if '2022' in sub_path or '2023' in sub_path:
    #     sub_path = os.path.join(os.path.join(t2m_path,sub_path),'data.nc')
    #     meteorology_data.chillingdays_calculation(sub_path)
    # VPD计算
    # 合并2000-2022和2023的数据
    # old_path = r'D:\Data Collection\Temperature\T2M\monthly_temperature.netcdf_2000_2022/data.nc'
    # new_path = r'D:\Data Collection\Temperature\T2M/T2M2023.nc'
    # old,new = xr.open_dataset(old_path),xr.open_dataset(new_path)
    # new = new.rename({'valid_time': 'time'})
    # new['latitude'] = new['latitude'].astype(old.latitude.dtype)
    # new['longitude'] = new['longitude'].astype(old.latitude.dtype)
    # combined_ds = xr.concat([old, new], dim='time')
    # combined_ds.to_netcdf(r'D:\Data Collection\Temperature\T2M/T2M_2000to2023.nc')
    # VPD计算
    # t2m_path = r'D:\Data Collection\Temperature\T2M\T2M_2000to2023.nc'
    # dewpoint_path = r'D:\Data Collection\Temperature\Dewpoint_temperature\D2M_2000to2023.nc'
    # dem_path = r'D:\Data Collection\DEM/Europe_DEM.tif'
    # meteorology_data.VPD_calculation(t2m_path,dewpoint_path,dem_path)

    # precips_path = r'D:\Data Collection\SPEI_variables/SPEI_calculation\tp'
    # pet_path = r'D:\Data Collection\SPEI_variables/SPEI_calculation\pev'
    # tm_path = r'D:\Data Collection\SPEI_variables\SPEI_calculation\monthly_temperature.netcdf/data.nc'
    # meteorology_data.SPEI_calculation(precips_path,tm_path)





