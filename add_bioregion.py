#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 5/9/2025 10:06
# @Author :
import pandas as pd
import xarray as xr

from tqdm import tqdm
import glob
import os
os.environ["OMP_NUM_THREADS"] = '7'
import re
import rasterio
from scipy.ndimage import distance_transform_edt

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
        r'data collection\SPEI/spei03.nc',
        r'data collection\SPEI/spei06.nc']
    tem_path = r'D:\Data Collection\Temperature\T2M\T2M_2000to2023.nc'
    VPD_path = r'D:\Data Collection\Temperature\VPD/VPD_2000to2023.nc'
    TP_path = r'D:\Data Collection\other_factors/total_precipitation.nc'
    TP_path_2224 = r'D:\Data Collection\other_factors/total_precipitation_22_24.nc'
    SIF_path = r'data collection/SIF_tempory.nc'
    chillingdays_data_path = r'D:\Data Collection\Temperature\chillingdays'
    chillingdays_paths = glob.glob(os.path.join(chillingdays_data_path, '*.nc'))
    drought_years = [2003, 2015, 2018, 2019, 2022]
    norway_index = None
    geobioregion_path = r'Biogeo_region/Biogeoregion_Europe.tif'
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
                    x, y = ref_src.xy(r, c)
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
                        x, y = ref_src.xy(r, c)
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
