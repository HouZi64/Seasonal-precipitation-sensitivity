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
os.environ["OMP_NUM_THREADS"] = "8"       # 5850U：8 线程最稳（对应8物理核）
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
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
from joblib import Parallel, delayed
spei_path = r'C:\CMZ\pycharm_projects\pythonProject\TUM\paper projects\1 legacy effects of drought\data collection\SPEI/spei03.nc'

EVI_path = r'D:\Data Collection\RS\MODIS\EVI_merge'
tqdm.pandas()
plt.rcParams['font.family'] = 'Arial'


class legacy_effects():
    def __init__(self):
       self.biogeo_id_mapping = {
           1: "Black Sea Bio-geographical Region",
           2: "Pannonian Bio-geographical Region",
           3: "Alpine Bio-geographical Region",
           4: "Atlantic Bio-geographical Region",
           5: "Continental Bio-geographical Region",
           6: "Macaronesian Bio-geographical Region",
           7: "Mediterranean Bio-geographical Region",
           8: "Boreal Bio-geographical Region",
           9: "Steppic Bio-geographical Region",
           10: "Arctic Bio-geographical Region",
       }

    def random_forest(self,target,features,df,year):
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
            output_path = 'after_first_revision/images'
        else:
            output_path = 'after_first_revision/images window'
        os.makedirs(output_path,exist_ok=True)
        # 载入数据
        df_new = df.copy()
        features_new = features.copy()[0:-5]
        y = df_new[target]
        X = df_new[features]

        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, df_new['weights'], test_size=0.2, random_state=42
        )
        # 训练模型
        rf_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=10)
        rf_model.fit(X_train[features_new], y_train, sample_weight=X_train['weights'])

        # 测试集预测
        y_pred = rf_model.predict(X_test[features_new])

        # 计算指标
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # 绘制散点密度图
        scatter_density_plot(y_test, y_pred, mse, mae, r2, output_path, year)
        # 保存预测结果
        pd.DataFrame({'True_Values': y_test, 'Predicted_Values': y_pred}).to_csv(
            os.path.join(output_path, f'y_val_y_pred_{year}.csv')
        )

        # 计算 SHAP (基于测试集)

        # # explainer = shap.TreeExplainer(rf_model)
        explainer = shap.TreeExplainer(rf_model,approximate=True, feature_perturbation="tree_path_dependent")

        # 定义分块函数
        def compute_shap(chunk):
            return explainer.shap_values(chunk, check_additivity=False)

        # 分块
        n_jobs = 10  # 并行核数
        X_splits = np.array_split(X_test[features_new], n_jobs)

        # 并行计算
        shap_values_parts = Parallel(n_jobs=n_jobs)(
            delayed(compute_shap)(chunk) for chunk in X_splits
        )
        #
        # # 拼接
        shap_values = np.vstack(shap_values_parts)
        #
        # shap_values = explainer.shap_values(X_test[features_new])
        # explainer = shap.TreeExplainer(rf_model)  # 创建SHAP解释器
        # shap_values = explainer.shap_values(X_test[features_new])  # 计算验证集的SHAP值
        shap_data_all = df_new.loc[X_test.index, ['row',
                                                  'col',
                                                  'country',
                                                  # 'year',
                                                  'biogeo']].copy()
        shap_data_all['tp_spring_shap_value'] = shap_values[:, 0]
        shap_data_all['tp_summer_shap_value'] = shap_values[:, 1]
        shap_data_all['tp_autumn_shap_value'] = shap_values[:, 2]
        shap_data_all['tp_winter_shap_value'] = shap_values[:, 3]
        shap_data_all['vpd_annual_shap_value'] = shap_values[:, 4]
        shap_data_all['tm_annual_shap_value'] = shap_values[:, 5]
        shap_data_all['spei_annual_shap_value'] = shap_values[:, 6]
        shap_data_all.to_csv(os.path.join(output_path, f'SHAP_summary_{year}.csv'))
        # 绘制 SHAP summary
        shap.summary_plot(shap_values, X_test[features_new], show=False)
        plt.savefig(os.path.join(output_path, f'Shap_summary_{year}.jpg'), bbox_inches='tight', dpi=300)
        print(f'saved shap summary in ' + os.path.join(output_path, f'Shap_summary_{year}.jpg'))
        plt.close()

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
    def random_forest_co(self,target,features,df,year,indices_name,scenario):
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
            output_path = f'after_first_revision/images_{indices_name}_{scenario}'
        else:
            output_path = 'after_first_revision/images window'
        os.makedirs(output_path,exist_ok=True)
        # 载入数据
        df_new = df.copy()
        features_new = features.copy()[0:-6]
        y = df_new[target]
        X = df_new[features]
        season_name = features_new[0].split('_')[1]
        if season_name not in ['spring','summer','autumn','winter']:assert 'wrong'
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
            data_shap = X.iloc[val_index][['row',
                                           'col',
                                           'country',
                                           # 'year'
                                           'biogeo',
                                           'country_chunk'
                                           ]]
            data_shap[f'tp_{season_name}_relative_shap_ratio'] = shap_values[:,0] / (np.sum(np.abs(shap_values),axis=1))
            data_shap[f'vpd_{season_name}_relative_shap_ratio'] = shap_values[:, 1] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'tm_{season_name}_relative_shap_ratio'] = shap_values[:, 2] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'tp_{season_name}_shap_value'] = shap_values[:,0]
            data_shap[f'vpd_{season_name}_shap_value'] = shap_values[:, 1]
            data_shap[f'tm_{season_name}_shap_value'] = shap_values[:, 2]
            # data_shap[f'tp_spring_relative_shap_ratio'] = shap_values[:,0] / (np.sum(np.abs(shap_values),axis=1))
            # data_shap[f'tp_summer_relative_shap_ratio'] = shap_values[:, 1] / (np.sum(np.abs(shap_values), axis=1))
            # data_shap[f'tp_autumn_relative_shap_ratio'] = shap_values[:, 2] / (np.sum(np.abs(shap_values), axis=1))
            # data_shap[f'tp_winter_relative_shap_ratio'] = shap_values[:, 3] / (np.sum(np.abs(shap_values), axis=1))
            # data_shap[f'vpd_annual_relative_shap_ratio'] = shap_values[:, 4] / (np.sum(np.abs(shap_values), axis=1))
            # data_shap[f'tm_annual_relative_shap_ratio'] = shap_values[:, 5] / (np.sum(np.abs(shap_values), axis=1))
            # data_shap[f'spei_annual_relative_shap_ratio'] = shap_values[:,6] / (np.sum(np.abs(shap_values), axis=1))
            # data_shap[f'tp_spring_shap_value'] = shap_values[:,0]
            # data_shap[f'tp_summer_shap_value'] = shap_values[:, 1]
            # data_shap[f'tp_autumn_shap_value'] = shap_values[:, 2]
            # data_shap[f'tp_winter_shap_value'] = shap_values[:, 3]
            # data_shap[f'vpd_annual_shap_value'] = shap_values[:, 4]
            # data_shap[f'tm_annual_shap_value'] = shap_values[:, 5]
            # data_shap[f'spei_annual_shap_value'] = shap_values[:, 6]
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
        spatial_data_all = pd.concat([spatial_data_all, data_all['y_val'].squeeze('columns').rename('ground_truth')], axis=1)
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
        # data_all['x_val'].rename(columns=rename_mapping, inplace=True)
        shap.summary_plot(data_all['shap'], data_all['x_val'][data_all['x_val'].keys().tolist()], show=False)
        plt.savefig(os.path.join(output_path,f'Shap_summary_{year}.jpg'), bbox_inches='tight', dpi=600)
        plt.savefig(os.path.join(output_path, f'Shap_summary_{year}.pdf'), bbox_inches='tight', dpi=600)
        plt.close()  # 关闭图形以释放内存
        # # SHAP Dependence Plot for each feature
        for i, feature1 in enumerate(features_new):
            for j, feature2 in enumerate(features_new):
                if i == j: continue
                plt.rcParams['font.family'] = 'Arial'
                plt.figure()
                shap.dependence_plot(feature1, data_all['shap'], data_all['x_val'], interaction_index=feature2,
                                     show=False, cmap=plt.get_cmap("RdBu").reversed())
                # plt.savefig(os.path.join(output_path, f'shap_dependence_plot_{year}_{feature}.pdf'),
                #             bbox_inches='tight', dpi=300)
                plt.savefig(
                    os.path.join(output_path, f'shap_dependence_plot_{feature1}_vs_{feature2}.jpg'),
                    bbox_inches='tight', dpi=600)
                plt.savefig(os.path.join(output_path, f'shap_dependence_plot_{feature1}_vs_{feature2}.pdf'),
                                         bbox_inches='tight', dpi=600)
                plt.close()
    def random_forest_co_group(self,target,features,df,year,indices_name,scenario):

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

        def fast_shap_rf_5850U(rf_model, X,
                               approximate=True,  # 快速近似（强烈推荐先用）
                               batch=500,  # 分批；内存紧就 2000
                               n_jobs=3,  # 进程并行数；16GB内存上 3~4 比较稳
                               shrink_trees=None,  # 如 100：只用前100棵树近似（再提速）
                               seed=42):
            # 模型内部别再并行，避免抢核
            if hasattr(rf_model, "n_jobs"):
                try:
                    rf_model.set_params(n_jobs=1)
                except Exception:
                    pass

            X = np.asarray(X, dtype=np.float32)  # 降到 float32 省内存/带宽

            # 可选：抽样子树（进一步提速），使用固定随机种子
            model_used = rf_model
            if (shrink_trees is not None) and hasattr(rf_model, "estimators_"):
                from copy import deepcopy
                ests = np.array(rf_model.estimators_)
                k = shrink_trees
                if k >= len(ests):
                    return rf_model
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(ests), size=k, replace=False)
                model_used = deepcopy(rf_model)
                model_used.estimators_ = [ests[i] for i in idx]
                model_used.n_estimators = shrink_trees

            # 分块索引
            idx = [(i, min(i + batch, len(X))) for i in range(0, len(X), batch)]

            def run(i0, i1):
                # 在子进程里新建 explainer 更稳（避免对象序列化问题）
                exp = shap.TreeExplainer(
                    model_used,
                    feature_perturbation="tree_path_dependent",
                    model_output="raw",
                    approximate=approximate
                )
                return exp.shap_values(X[i0:i1], check_additivity=False)

            t0 = time.perf_counter()
            if len(idx) == 1:
                parts = [run(*idx[0])]
            else:
                parts = Parallel(n_jobs=n_jobs, backend="loky")(delayed(run)(i0, i1) for i0, i1 in idx)

            # 拼接
            if isinstance(parts[0], list):
                shap_values = [np.vstack([p[k] for p in parts]) for k in range(len(parts[0]))]
            else:
                shap_values = np.vstack(parts)

            info = {
                "time_sec": round(time.perf_counter() - t0, 3),
                "threads_omp": int(os.environ.get("OMP_NUM_THREADS", "8")),
                "n_jobs": n_jobs,
                "batch": batch,
                "mode": "approximate" if approximate else "exact",
                "n_trees_used": len(getattr(model_used, "estimators_", [])),
                "dtype": str(X.dtype)
            }
            return shap_values, info

        # 定义保存图片路径
        if year == 'Temporal Series':
            output_path = f'after_first_revision/images_{indices_name}_{scenario}'
        else:
            output_path = f'after_first_revision/images_{indices_name}_{scenario}'
        os.makedirs(output_path,exist_ok=True)
        # if os.path.exists(os.path.join(output_path, f'SHAP_summary_{year}.csv')): return None
        # 载入数据
        df_new = df.copy()
        features_new = features.copy()[0:-6]
        y = df_new[target]
        X = df_new[features]
        season_name = features_new[0].split('_')[1]
        if season_name not in ['spring','summer','autumn','winter']:assert 'wrong'
        rf_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=10)
        all_idx = np.arange(len(X))
        train_index, val_index = train_test_split(
            all_idx, test_size=0.2, shuffle=True, random_state=42
        )

        # 构造 train / val
        X_train_fold, X_val_fold = X.iloc[train_index][features_new], X.iloc[val_index][features_new]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        weights_k = X.iloc[train_index]['weights']

        # 训练
        rf_model.fit(X_train_fold, y_train_fold, sample_weight=weights_k)

        # 预测
        y_pred = rf_model.predict(X_val_fold)
        y_pred_all = y_pred.copy()
        y_val_all = y_val_fold.values.copy()

        # 评估
        r2  = r2_score(y_val_fold, y_pred)
        mse = mean_squared_error(y_val_fold, y_pred)
        mae = mean_absolute_error(y_val_fold, y_pred)
        # 绘制散点图,保存真值和预测值为csv
        scatter_density_plot(y_val_all, y_pred_all, mse,mae, r2, output_path,year)
        pd.DataFrame({'True_Values': y_val_all,'Predicted_Values': y_pred_all}).to_csv(os.path.join(output_path,f'y_val_y_pred_{year}.csv'))
        # explainer = shap.TreeExplainer(
        #     rf_model,
        #     feature_perturbation="tree_path_dependent",
        #     model_output="raw",
        #     approximate=True
        # )
        X_val_for_shap = pd.DataFrame(X_val_fold, columns=features_new)
        val_index_picked = val_index
        # shap_values = explainer.shap_values(X_val_for_shap, check_additivity=False)  # (m, n_features)
        shap_values,shap_info = fast_shap_rf_5850U(rf_model, X_val_for_shap, approximate=False, batch=500, n_jobs=4, shrink_trees=300)
        # 组装空间信息
        data_shap = X.iloc[val_index_picked][[
            'row', 'col', 'country', 'biogeo', 'country_chunk'
        ]].copy()

        denom = np.sum(np.abs(shap_values), axis=1)
        denom[denom == 0] = np.finfo('float32').eps

        # 假设 features_new 顺序为 [tp_*, vpd_*, tm_*]
        # data_shap[f'tp_{season_name}_relative_shap_ratio']  = shap_values[:, 0] / denom
        # data_shap[f'vpd_{season_name}_relative_shap_ratio'] = shap_values[:, 1] / denom
        # data_shap[f'tm_{season_name}_relative_shap_ratio']  = shap_values[:, 2] / denom
        # data_shap[f'tp_{season_name}_shap_value']  = shap_values[:, 0]
        # data_shap[f'vpd_{season_name}_shap_value'] = shap_values[:, 1]
        # data_shap[f'tm_{season_name}_shap_value']  = shap_values[:, 2]

        data_shap[f'tp_spring_relative_shap_ratio'] = shap_values[:,0] / (np.sum(np.abs(shap_values),axis=1))
        data_shap[f'tp_summer_relative_shap_ratio'] = shap_values[:, 1] / (np.sum(np.abs(shap_values), axis=1))
        data_shap[f'tp_autumn_relative_shap_ratio'] = shap_values[:, 2] / (np.sum(np.abs(shap_values), axis=1))
        data_shap[f'tp_winter_relative_shap_ratio'] = shap_values[:, 3] / (np.sum(np.abs(shap_values), axis=1))
        data_shap[f'tp_spring_shap_value'] = shap_values[:,0]
        data_shap[f'tp_summer_shap_value'] = shap_values[:, 1]
        data_shap[f'tp_autumn_shap_value'] = shap_values[:, 2]
        data_shap[f'tp_winter_shap_value'] = shap_values[:, 3]
        spatial_data_all = pd.DataFrame()
        data_all = {
            'shap':np.empty((0,len(features_new))),
            'y_val':pd.DataFrame(),
            'x_val':pd.DataFrame()}     #汇总所有的shap值和对应的X_test
        spatial_data_all = pd.concat((spatial_data_all, data_shap))
        data_all['shap'] = np.vstack((data_all['shap'], shap_values))
        data_all['y_val'] = pd.concat((data_all['y_val'], y_val_fold))
        data_all['x_val'] = pd.concat((data_all['x_val'], X_val_fold))
        data_all['x_val'].add_suffix('_xValue')
        data_all['y_val'].add_suffix('_yValue')
        spatial_data_all = pd.concat([spatial_data_all, data_all['x_val']], axis=1)
        spatial_data_all = pd.concat([spatial_data_all, data_all['y_val'].squeeze('columns').rename('ground_truth')],
                                     axis=1)
        spatial_data_all.to_csv(os.path.join(output_path, f'SHAP_summary_{year}.csv'))
        # 可视化shapsummary图
        shap.summary_plot(data_all['shap'], data_all['x_val'][data_all['x_val'].keys().tolist()], show=False)
        plt.savefig(os.path.join(output_path, f'Shap_summary_{year}.jpg'), bbox_inches='tight', dpi=600)
        plt.savefig(os.path.join(output_path, f'Shap_summary_{year}.pdf'), bbox_inches='tight', dpi=600)
        plt.close()  # 关闭图形以释放内存

        # # # SHAP Dependence Plot for each feature
        # for i, feature1 in enumerate(features_new):
        #     for j, feature2 in enumerate(features_new):
        #         if i == j: continue
        #         plt.rcParams['font.family'] = 'Arial'
        #         plt.figure()
        #         shap.dependence_plot(feature1, data_all['shap'], data_all['x_val'], interaction_index=feature2,
        #                              show=False, cmap=plt.get_cmap("RdBu").reversed())
        #         # plt.savefig(os.path.join(output_path, f'shap_dependence_plot_{year}_{feature}.pdf'),
        #         #             bbox_inches='tight', dpi=300)
        #         plt.savefig(
        #             os.path.join(output_path, f'shap_dependence_plot_{feature1}_vs_{feature2}.jpg'),
        #             bbox_inches='tight', dpi=600)
        #         plt.savefig(os.path.join(output_path, f'shap_dependence_plot_{feature1}_vs_{feature2}.pdf'),
        #                                  bbox_inches='tight', dpi=600)
        #         plt.close()
    def random_forest_co_group_biogeo(self, target, features, df, year, indices_name, scenario,biogeo):

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
            plt.text(0.05, 0.90, f'R²={r2:.3f}', transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top')
            plt.text(0.05, 0.85, f'MSE={mse:.3f}', transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top')
            plt.text(0.05, 0.80, f'MAE={mae:.3f}', transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top')

            # 设置坐标轴和标题
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.grid(True)
            # 保存图片
            plt.savefig(os.path.join(output_path, f'Density_Scatter_Plot_{year}.png'), dpi=300)
            plt.close()  # 关闭图形以释放内存

        def fast_shap_rf_5850U(rf_model, X,
                               approximate=True,  # 快速近似（强烈推荐先用）
                               batch=500,  # 分批；内存紧就 2000
                               n_jobs=3,  # 进程并行数；16GB内存上 3~4 比较稳
                               shrink_trees=None,  # 如 100：只用前100棵树近似（再提速）
                               seed=42):
            # 模型内部别再并行，避免抢核
            if hasattr(rf_model, "n_jobs"):
                try:
                    rf_model.set_params(n_jobs=1)
                except Exception:
                    pass

            X = np.asarray(X, dtype=np.float32)  # 降到 float32 省内存/带宽

            # 可选：抽样子树（进一步提速），使用固定随机种子
            model_used = rf_model
            if (shrink_trees is not None) and hasattr(rf_model, "estimators_"):
                from copy import deepcopy
                ests = np.array(rf_model.estimators_)
                k = shrink_trees
                if k >= len(ests):
                    return rf_model
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(ests), size=k, replace=False)
                model_used = deepcopy(rf_model)
                model_used.estimators_ = [ests[i] for i in idx]
                model_used.n_estimators = shrink_trees

            # 分块索引
            idx = [(i, min(i + batch, len(X))) for i in range(0, len(X), batch)]

            def run(i0, i1):
                # 在子进程里新建 explainer 更稳（避免对象序列化问题）
                exp = shap.TreeExplainer(
                    model_used,
                    feature_perturbation="tree_path_dependent",
                    model_output="raw",
                    approximate=approximate
                )
                return exp.shap_values(X[i0:i1], check_additivity=False)

            t0 = time.perf_counter()
            if len(idx) == 1:
                parts = [run(*idx[0])]
            else:
                parts = Parallel(n_jobs=n_jobs, backend="loky")(delayed(run)(i0, i1) for i0, i1 in idx)

            # 拼接
            if isinstance(parts[0], list):
                shap_values = [np.vstack([p[k] for p in parts]) for k in range(len(parts[0]))]
            else:
                shap_values = np.vstack(parts)

            info = {
                "time_sec": round(time.perf_counter() - t0, 3),
                "threads_omp": int(os.environ.get("OMP_NUM_THREADS", "8")),
                "n_jobs": n_jobs,
                "batch": batch,
                "mode": "approximate" if approximate else "exact",
                "n_trees_used": len(getattr(model_used, "estimators_", [])),
                "dtype": str(X.dtype)
            }
            return shap_values, info

        # 定义保存图片路径
        if year == 'Temporal Series':
            output_path = f'after_first_revision/images_{indices_name}_{scenario}'
        else:
            output_path = f'after_first_revision/biogeo_{biogeo}_temporal_trend/images_{indices_name}_{scenario}'
        os.makedirs(output_path, exist_ok=True)
        # if os.path.exists(os.path.join(output_path, f'SHAP_summary_{year}.csv')): return None
        # 载入数据
        df_new = df.copy()
        features_new = features.copy()[0:-6]
        y = df_new[target]
        X = df_new[features]
        season_name = features_new[0].split('_')[1]
        if season_name not in ['spring', 'summer', 'autumn', 'winter']: assert 'wrong'
        rf_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=10)
        all_idx = np.arange(len(X))
        train_index, val_index = train_test_split(
            all_idx, test_size=0.2, shuffle=True, random_state=42
        )

        # 构造 train / val
        X_train_fold, X_val_fold = X.iloc[train_index][features_new], X.iloc[val_index][features_new]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        weights_k = X.iloc[train_index]['weights']

        # 训练
        rf_model.fit(X_train_fold, y_train_fold, sample_weight=weights_k)

        # 预测
        y_pred = rf_model.predict(X_val_fold)
        y_pred_all = y_pred.copy()
        y_val_all = y_val_fold.values.copy()

        # 评估
        r2 = r2_score(y_val_fold, y_pred)
        mse = mean_squared_error(y_val_fold, y_pred)
        mae = mean_absolute_error(y_val_fold, y_pred)
        # 绘制散点图,保存真值和预测值为csv
        scatter_density_plot(y_val_all, y_pred_all, mse, mae, r2, output_path, year)
        pd.DataFrame({'True_Values': y_val_all, 'Predicted_Values': y_pred_all}).to_csv(
            os.path.join(output_path, f'y_val_y_pred_{year}.csv'))

        X_val_for_shap = pd.DataFrame(X_val_fold, columns=features_new)
        val_index_picked = val_index
        shap_values, shap_info = fast_shap_rf_5850U(rf_model, X_val_for_shap, approximate=False, batch=500,
                                                    n_jobs=4, shrink_trees=300)
        # 组装空间信息
        data_shap = X.iloc[val_index_picked][[
            'row', 'col', 'country', 'biogeo', 'country_chunk'
        ]].copy()

        denom = np.sum(np.abs(shap_values), axis=1)
        denom[denom == 0] = np.finfo('float32').eps

        # 假设 features_new 顺序为 [tp_*, vpd_*, tm_*]
        # data_shap[f'tp_{season_name}_relative_shap_ratio']  = shap_values[:, 0] / denom
        # data_shap[f'vpd_{season_name}_relative_shap_ratio'] = shap_values[:, 1] / denom
        # data_shap[f'tm_{season_name}_relative_shap_ratio']  = shap_values[:, 2] / denom
        # data_shap[f'tp_{season_name}_shap_value']  = shap_values[:, 0]
        # data_shap[f'vpd_{season_name}_shap_value'] = shap_values[:, 1]
        # data_shap[f'tm_{season_name}_shap_value']  = shap_values[:, 2]

        data_shap[f'tp_spring_relative_shap_ratio'] = shap_values[:, 0] / (np.sum(np.abs(shap_values), axis=1))
        data_shap[f'tp_summer_relative_shap_ratio'] = shap_values[:, 1] / (np.sum(np.abs(shap_values), axis=1))
        data_shap[f'tp_autumn_relative_shap_ratio'] = shap_values[:, 2] / (np.sum(np.abs(shap_values), axis=1))
        data_shap[f'tp_winter_relative_shap_ratio'] = shap_values[:, 3] / (np.sum(np.abs(shap_values), axis=1))
        data_shap[f'tp_spring_shap_value'] = shap_values[:, 0]
        data_shap[f'tp_summer_shap_value'] = shap_values[:, 1]
        data_shap[f'tp_autumn_shap_value'] = shap_values[:, 2]
        data_shap[f'tp_winter_shap_value'] = shap_values[:, 3]
        spatial_data_all = pd.DataFrame()
        data_all = {
            'shap': np.empty((0, len(features_new))),
            'y_val': pd.DataFrame(),
            'x_val': pd.DataFrame()}  # 汇总所有的shap值和对应的X_test
        spatial_data_all = pd.concat((spatial_data_all, data_shap))
        data_all['shap'] = np.vstack((data_all['shap'], shap_values))
        data_all['y_val'] = pd.concat((data_all['y_val'], y_val_fold))
        data_all['x_val'] = pd.concat((data_all['x_val'], X_val_fold))
        data_all['x_val'].add_suffix('_xValue')
        data_all['y_val'].add_suffix('_yValue')
        spatial_data_all = pd.concat([spatial_data_all, data_all['x_val']], axis=1)
        spatial_data_all = pd.concat(
            [spatial_data_all, data_all['y_val'].squeeze('columns').rename('ground_truth')],
            axis=1)
        spatial_data_all.to_csv(os.path.join(output_path, f'SHAP_summary_{year}.csv'))
        # 可视化shapsummary图
        shap.summary_plot(data_all['shap'], data_all['x_val'][data_all['x_val'].keys().tolist()], show=False)
        plt.savefig(os.path.join(output_path, f'Shap_summary_{year}.jpg'), bbox_inches='tight', dpi=600)
        plt.savefig(os.path.join(output_path, f'Shap_summary_{year}.pdf'), bbox_inches='tight', dpi=600)
        plt.close()  # 关闭图形以释放内存
    def random_forest_co_forspatial(self,target,features,df,year,indices_name,scenario,season):

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

        def fast_shap_rf_5850U(rf_model, X,
                               approximate=True,  # 快速近似（强烈推荐先用）
                               batch=500,  # 分批；内存紧就 2000
                               n_jobs=3,  # 进程并行数；16GB内存上 3~4 比较稳
                               shrink_trees=None,  # 如 100：只用前100棵树近似（再提速）
                               seed=42):
            # 模型内部别再并行，避免抢核
            if hasattr(rf_model, "n_jobs"):
                try:
                    rf_model.set_params(n_jobs=1)
                except Exception:
                    pass

            X = np.asarray(X, dtype=np.float32)  # 降到 float32 省内存/带宽

            # 可选：抽样子树（进一步提速），使用固定随机种子
            model_used = rf_model
            if (shrink_trees is not None) and hasattr(rf_model, "estimators_"):
                from copy import deepcopy
                ests = np.array(rf_model.estimators_)
                k = shrink_trees
                if k >= len(ests):
                    return rf_model
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(ests), size=k, replace=False)
                model_used = deepcopy(rf_model)
                model_used.estimators_ = [ests[i] for i in idx]
                model_used.n_estimators = shrink_trees

            # 分块索引
            idx = [(i, min(i + batch, len(X))) for i in range(0, len(X), batch)]

            def run(i0, i1):
                # 在子进程里新建 explainer 更稳（避免对象序列化问题）
                exp = shap.TreeExplainer(
                    model_used,
                    feature_perturbation="tree_path_dependent",
                    model_output="raw",
                    approximate=approximate
                )
                return exp.shap_values(X[i0:i1], check_additivity=False)

            t0 = time.perf_counter()
            if len(idx) == 1:
                parts = [run(*idx[0])]
            else:
                parts = Parallel(n_jobs=n_jobs, backend="loky")(delayed(run)(i0, i1) for i0, i1 in idx)

            # 拼接
            if isinstance(parts[0], list):
                shap_values = [np.vstack([p[k] for p in parts]) for k in range(len(parts[0]))]
            else:
                shap_values = np.vstack(parts)

            info = {
                "time_sec": round(time.perf_counter() - t0, 3),
                "threads_omp": int(os.environ.get("OMP_NUM_THREADS", "8")),
                "n_jobs": n_jobs,
                "batch": batch,
                "mode": "approximate" if approximate else "exact",
                "n_trees_used": len(getattr(model_used, "estimators_", [])),
                "dtype": str(X.dtype)
            }
            return shap_values, info

        # 定义保存图片路径
        if year == 'Temporal Series':
            output_path = f'after_first_revision/images_{indices_name}_{scenario}_forspatial_{season}'
        else:
            output_path = 'after_first_revision/images window'
        os.makedirs(output_path, exist_ok=True)
        if os.path.exists(os.path.join(output_path,'SHAP_summary_Temporal Series.csv')):
            print(f'{season} completed')
            return None
        # 载入数据
        df_new = df.copy()
        features_new = features.copy()[0:-6]
        y = df_new[target]
        X = df_new[features]
        season_name = features_new[0].split('_')[1]
        if season_name not in ['spring', 'summer', 'autumn', 'winter']: assert 'wrong'
        # 定义 K 折交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        # 创建随机森林模型
        rf_model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=10)
        # 使用交叉验证评估 R²
        r2_scores = []
        mse_scores = []
        mae_scores = []
        # 设立权重
        spatial_data_all = pd.DataFrame()
        data_all = {
            'shap': np.empty((0, len(features_new))),
            'y_val': pd.DataFrame(),
            'x_val': pd.DataFrame()}  # 汇总所有的shap值和对应的X_test
        y_pred_all = np.array([])
        y_val_all = np.array([])
        i = 1
        for train_index, val_index in kf.split(X):
            X_train_fold, X_val_fold = X.iloc[train_index][features_new], X.iloc[val_index][features_new]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
            weights_k = X.iloc[train_index]['weights']
            rf_model.fit(X_train_fold, y_train_fold, sample_weight=weights_k)
            # 在验证集上进行预测
            y_pred = rf_model.predict(X_val_fold)
            y_pred_all = np.concatenate([y_pred_all, y_pred])
            y_val_all = np.concatenate([y_val_all, y_val_fold])
            # 计算验证集的 R² 分数
            r2 = r2_score(y_val_fold, y_pred)
            r2_scores.append(r2)
            mse = mean_squared_error(y_val_fold, y_pred)
            mse_scores.append(mse)
            mae = mean_absolute_error(y_val_fold, y_pred)
            mae_scores.append(mae)
            # 计算验证集的 SHAP 值
            X_val_for_shap = pd.DataFrame(X_val_fold, columns=features_new)
            shap_values, shap_info = fast_shap_rf_5850U(rf_model, X_val_for_shap, approximate=True, batch=500,
                                                        n_jobs=4, shrink_trees=100)
            data_shap = X.iloc[val_index][['row',
                                           'col',
                                           'country',
                                           # 'year'
                                           'biogeo',
                                           'country_chunk'
                                           ]]
            # data_shap[f'tp_spring_relative_shap_ratio'] = shap_values[:,0] / (np.sum(np.abs(shap_values),axis=1))
            # data_shap[f'tp_summer_relative_shap_ratio'] = shap_values[:, 1] / (np.sum(np.abs(shap_values), axis=1))
            # data_shap[f'tp_autumn_relative_shap_ratio'] = shap_values[:, 2] / (np.sum(np.abs(shap_values), axis=1))
            # data_shap[f'tp_winter_relative_shap_ratio'] = shap_values[:, 3] / (np.sum(np.abs(shap_values), axis=1))
            denom = np.sum(np.abs(shap_values), axis=1)
            denom[denom == 0] = np.finfo('float32').eps
            data_shap[f'tp_{season_name}_relative_shap_ratio']  = shap_values[:, 0] / denom
            data_shap[f'vpd_{season_name}_relative_shap_ratio'] = shap_values[:, 1] / denom
            data_shap[f'tm_{season_name}_relative_shap_ratio']  = shap_values[:, 2] / denom
            # data_shap[f'tp_spring_shap_value'] = shap_values[:,0]
            # data_shap[f'tp_summer_shap_value'] = shap_values[:, 1]
            # data_shap[f'tp_autumn_shap_value'] = shap_values[:, 2]
            # data_shap[f'tp_winter_shap_value'] = shap_values[:, 3]
            data_shap[f'tp_{season_name}_shap_value']  = shap_values[:, 0]
            data_shap[f'vpd_{season_name}_shap_value'] = shap_values[:, 1]
            data_shap[f'tm_{season_name}_shap_value']  = shap_values[:, 2]
            # 保存每次验证集的 SHAP 值
            spatial_data_all = pd.concat((spatial_data_all, data_shap))
            data_all['shap'] = np.vstack((data_all['shap'], shap_values))
            data_all['y_val'] = pd.concat((data_all['y_val'], y_val_fold))
            data_all['x_val'] = pd.concat((data_all['x_val'], X_val_fold))
            print(f'计算完{i}折')
            i += 1
        r2 = np.mean(r2_scores)
        mse = np.mean(mse_scores)
        mae = np.mean(mae_scores)
        # 绘制散点图,保存真值和预测值为csv
        scatter_density_plot(y_val_all, y_pred_all, mse, mae, r2, output_path, year)
        pd.DataFrame({'True_Values': y_val_all, 'Predicted_Values': y_pred_all}).to_csv(
            os.path.join(output_path, f'y_val_y_pred_{year}.csv'))
        # 保存shap信息为csv
        data_all['x_val'].add_suffix('_xValue')
        data_all['y_val'].add_suffix('_yValue')
        spatial_data_all = pd.concat([spatial_data_all, data_all['x_val']], axis=1)
        spatial_data_all = pd.concat(
            [spatial_data_all, data_all['y_val'].squeeze('columns').rename('ground_truth')], axis=1)
        spatial_data_all.to_csv(os.path.join(output_path, f'SHAP_summary_{year}.csv'))
        # data_all['x_val'].rename(columns=rename_mapping, inplace=True)
        shap.summary_plot(data_all['shap'], data_all['x_val'][data_all['x_val'].keys().tolist()], show=False)
        plt.savefig(os.path.join(output_path, f'Shap_summary_{year}.jpg'), bbox_inches='tight', dpi=600)
        plt.savefig(os.path.join(output_path, f'Shap_summary_{year}.pdf'), bbox_inches='tight', dpi=600)
        plt.close()  # 关闭图形以释放内存


    def rf_modelling(self,data_path,start_year,end_year,indices_name):
        # 基于滑动窗口的窗口分析建模
        df = pd.read_csv(data_path).dropna()

        filtered_df = df.copy()

        # 计算anomaly
        indices_cols = [item for item in filtered_df.columns if indices_name in item]
        indices_df =  filtered_df[indices_cols]
        indices_baseline = indices_df.mean(axis=1)
        for indices_col in indices_cols:
            filtered_df[indices_col.replace(indices_name,f'{indices_name}_anomaly')] = filtered_df[indices_col] - indices_baseline
        window_length = 3
        # 生成年份列表
        years = np.arange(start_year, end_year + 1)
        # 使用滑动窗口分组
        groups = [years[i:i + window_length] for i in range(len(years) - window_length + 1)]
        variables = [col.replace('_2005', '') for col in filtered_df.columns if '2005' in col]
        variables = [x for x in variables if x not in ('evi,sif')]
        filtered_df_new = filtered_df[['row','col','weights','country']]
        # if end_year > 2016:variables.remove('sif')  # 如果 end_year > 2017，则移除 'sif' 变量
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

    def stratified_cap_equal(self,df, strat_col='biogeo', total_n=10000, random_state=42):
        """
        分层随机抽样（按 biogeo）：
        - 小样本的 biogeo 组【全部抽样】；
        - 大样本的 biogeo 组按“每组不超过 t 条”的统一上限抽样；
        - 自动找到上限 t，使总样本数恰好为 total_n（若总体不足则全取）；
        - 组内随机、结果可复现。
        """
        if len(df) <= total_n:
            return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        key = df[strat_col].fillna('__NA__')
        sizes = key.value_counts(dropna=False)

        # --- 用二分法找统一上限 t: sum(min(size_g, t)) <= total_n，且尽量接近 total_n ---
        lo, hi = 0, int(sizes.max())

        def total_if_cap(t: int) -> int:
            return int((sizes.clip(upper=t)).sum())

        while lo < hi:
            mid = (lo + hi + 1) // 2  # 取上中位，逼近最大可行 t
            if total_if_cap(mid) <= total_n:
                lo = mid
            else:
                hi = mid - 1
        t = lo

        # 先按 min(size, t) 分配
        alloc = sizes.clip(upper=t).astype(int)
        # 还差多少补齐到 total_n（只从仍有富余的组里 +1）
        remain = total_n - int(alloc.sum())
        if remain > 0:
            spare = (sizes - alloc).sort_values(ascending=False)  # 有富余的在前
            bump_idx = spare.index[:remain]
            alloc.loc[bump_idx] += 1

        # —— 按 alloc 抽样（小组会被“全取”）——
        rng = np.random.RandomState(random_state)
        parts = []
        for g, k in alloc.items():
            grp = df[key == g]
            if k >= len(grp):
                take = grp
            else:
                take = grp.sample(n=k, random_state=random_state)
            parts.append(take)

        out = (pd.concat(parts, axis=0)
               .sample(frac=1, random_state=random_state)  # 最后整体打乱
               .reset_index(drop=True))
        return out

    def rf_modelling_all(self,data_path,start_year,end_year,indices_name,scenario):
        # 基于滑动窗口的窗口分析建模
        df = pd.read_csv(data_path).dropna()

        filtered_df = df.copy()

        # 计算anomaly
        indices_cols = [item for item in filtered_df.columns if indices_name in item]
        indices_df =  filtered_df[indices_cols]
        indices_baseline = indices_df.mean(axis=1)
        filtered_df[f'{indices_name}_baseline'] = indices_baseline
        # filtered_df = filtered_df[filtered_df[f'{indices_name}_baseline']>0.2]
        for indices_col in indices_cols:
            filtered_df[indices_col.replace(indices_name,f'{indices_name}_anomaly')] = filtered_df[indices_col] - indices_baseline

        # 计算气象要素baseline
        unique_base_names = ['tp_spring','tp_summer','tp_autumn','tp_winter','vpd_annual','tm_annual','spei_03_annual_spei',
                             'vpd_spring','vpd_summer','vpd_autumn','vpd_winter',
                             'tm_spring','tm_summer','tm_autumn','tm_winter']  # 太多了，指定一下
        for unique_base_name in tqdm(unique_base_names):
            unique_base_name_list = []
            climate_columns = [col for col in filtered_df.columns if unique_base_name in col]
            unique_base_df = filtered_df[climate_columns]
            unique_base_baseline = unique_base_df.mean(axis=1)
            filtered_df[f'{unique_base_name}_baseline'] = unique_base_baseline

        # 分层抽样
        # n_total = 10000
        # groups = filtered_df['biogeo'].unique()
        # n_groups = len(groups)
        # base_n = n_total // n_groups
        # # 多出来的样本数（平均分配给部分 biogeo）
        # remainder = n_total % n_groups
        # sampled_dfs = []
        # for i, g in enumerate(groups):
        #     group_df = filtered_df[filtered_df['biogeo'] == g]
        #     # 当前组要抽的样本数
        #     n_samples = base_n + (1 if i < remainder else 0)
        #     # 如果该组数量不足，就抽全量
        #     if len(group_df) < n_samples:
        #         sampled = group_df.sample(len(group_df), random_state=42)
        #     else:
        #         sampled = group_df.sample(n_samples, random_state=42)
        #     sampled_dfs.append(sampled)
        # # 合并所有抽样结果
        # sampled_df = pd.concat(sampled_dfs).reset_index(drop=True)
        # 生成年份列表
        years = np.arange(start_year, end_year + 1)
        # data_stack = self.stack_data(filtered_df,start_year,end_year,indices_name)
        target = f'{indices_name}_baseline'
        # 干旱指数建模
        season_name = 'winter'
        features = [f'tp_{season_name}_baseline',
                    # f'tp_spring_baseline',
                    # f'tp_summer_baseline',
                    # f'tp_autumn_baseline',
                    # f'tp_winter_baseline',
                    f'vpd_{season_name}_baseline',
                    f'tm_{season_name}_baseline',
                    # f'spei_03_annual_spei_anomaly',
                    'weights',
                    'row','col','country',
                    # 'year',
                    'biogeo',
                    'country_chunk']
        filtered_df_sub = self.stratified_cap_equal(filtered_df)
        self.random_forest_co(target,features,filtered_df_sub,'Temporal Series',indices_name,scenario)
    def rf_modelling_group(self, data_path, start_year, end_year, indices_name,scenario):

        # 基于池化的窗口分析
        df = pd.read_csv(data_path).dropna()
        filtered_df = df.copy()
        window_length = 10
        # 生成年份列表
        years = np.arange(start_year, end_year + 1)
        # 使用滑动窗口分组
        groups = [years[i:i + window_length] for i in range(len(years) - window_length + 1)]
        variables = [col.replace('_2005', '') for col in filtered_df.columns if '2005' in col]
        filtered_df_new = filtered_df[['row', 'col', 'weights', 'country','biogeo','country_chunk']]
        if end_year > 2017: variables.remove('sif')  # 如果 end_year > 2017，则移除 'sif' 变量
        pairs = {}
        for group_id, group in enumerate(groups):
            for variable in variables:
                colname = f'{variable}_baseline_group{group_id}'
                pairs[colname] = filtered_df[[f'{variable}_{y}' for y in group]].mean(axis=1)
        new_data = pd.DataFrame(pairs)
        # # 设置新列名
        # new_data.columns = new_columns_name
        filtered_df_new = pd.concat([filtered_df_new, new_data], axis=1)
        filtered_df_sub = self.stratified_cap_equal(filtered_df_new)
        for group_id, group in enumerate(groups):
            target = f'{indices_name}_baseline_group{group_id}'
            # 干旱指数建模
            season_name = 'spring'
            features = [
                        # f'tp_{season_name}_baseline_group{group_id}',
                        # f'vpd_{season_name}_baseline_group{group_id}',
                        # f'tm_{season_name}_baseline_group{group_id}',
                        f'tp_spring_baseline_group{group_id}',
                        f'tp_summer_baseline_group{group_id}',
                        f'tp_autumn_baseline_group{group_id}',
                        f'tp_winter_baseline_group{group_id}',
                        'weights',
                        'row', 'col', 'country',
                        'biogeo',
                        'country_chunk']
            self.random_forest_co_group(target, features, filtered_df_sub, group[0], indices_name,scenario)
            print(f'已完成{group_id+1} 组')


    def rf_modelling_group_biogeo(self, data_path, start_year, end_year, indices_name,scenario,biogeo):

        # 基于池化的窗口分析
        df = pd.read_csv(data_path).dropna()
        filtered_df = df.copy()
        filtered_df = filtered_df[filtered_df['biogeo']==biogeo]
        window_length = 10
        # 生成年份列表
        years = np.arange(start_year, end_year + 1)
        # 使用滑动窗口分组
        groups = [years[i:i + window_length] for i in range(len(years) - window_length + 1)]
        variables = [col.replace('_2005', '') for col in filtered_df.columns if '2005' in col]
        filtered_df_new = filtered_df[['row', 'col', 'weights', 'country','biogeo','country_chunk']]
        if end_year > 2017: variables.remove('sif')  # 如果 end_year > 2017，则移除 'sif' 变量
        pairs = {}
        for group_id, group in enumerate(groups):
            for variable in variables:
                colname = f'{variable}_baseline_group{group_id}'
                pairs[colname] = filtered_df[[f'{variable}_{y}' for y in group]].mean(axis=1)
        new_data = pd.DataFrame(pairs)
        # # 设置新列名
        # new_data.columns = new_columns_name
        filtered_df_new = pd.concat([filtered_df_new, new_data], axis=1)
        filtered_df_sub = self.stratified_cap_equal(filtered_df_new)
        for group_id, group in enumerate(groups):
            target = f'{indices_name}_baseline_group{group_id}'
            # 干旱指数建模
            season_name = 'spring'
            features = [
                        # f'tp_{season_name}_baseline_group{group_id}',
                        # f'vpd_{season_name}_baseline_group{group_id}',
                        # f'tm_{season_name}_baseline_group{group_id}',
                        f'tp_spring_baseline_group{group_id}',
                        f'tp_summer_baseline_group{group_id}',
                        f'tp_autumn_baseline_group{group_id}',
                        f'tp_winter_baseline_group{group_id}',
                        'weights',
                        'row', 'col', 'country',
                        'biogeo',
                        'country_chunk']
            self.random_forest_co_group_biogeo(target, features, filtered_df_sub, group[0], indices_name,scenario,biogeo)
            print(f'已完成{biogeo},{group_id+1} 组')

    def rf_modelling_all_forspatial(self, data_path, start_year, end_year, indices_name,scenario,season):

        # 基于池化的窗口分析
        df = pd.read_csv(data_path).dropna()
        filtered_df = df.copy()
        indices_cols = [item for item in filtered_df.columns if indices_name in item]
        indices_df =  filtered_df[indices_cols]
        indices_baseline = indices_df.mean(axis=1)
        filtered_df[f'{indices_name}_baseline'] = indices_baseline
        # 计算气象要素baseline
        unique_base_names = ['tp_spring','tp_summer','tp_autumn','tp_winter','vpd_annual','tm_annual','spei_03_annual_spei',
                             'vpd_spring','vpd_summer','vpd_autumn','vpd_winter',
                             'tm_spring','tm_summer','tm_autumn','tm_winter']  # 太多了，指定一下
        for unique_base_name in tqdm(unique_base_names):
            unique_base_name_list = []
            climate_columns = [col for col in filtered_df.columns if unique_base_name in col]
            unique_base_df = filtered_df[climate_columns]
            unique_base_baseline = unique_base_df.mean(axis=1)
            filtered_df[f'{unique_base_name}_baseline'] = unique_base_baseline

        target = f'{indices_name}_baseline'
        season_name = season
        # 干旱指数建模
        features = [
                    f'tp_{season_name}_baseline',
                    f'vpd_{season_name}_baseline',
                    f'tm_{season_name}_baseline',
                    # f'tp_spring_baseline',
                    # f'tp_summer_baseline',
                    # f'tp_autumn_baseline',
                    # f'tp_winter_baseline',
                    'weights',
                    'row', 'col', 'country',
                    'biogeo',
                    'country_chunk']
        self.random_forest_co_forspatial(target, features, filtered_df,'Temporal Series', indices_name,scenario,season)
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
                new_row["biogeo"] = row['biogeo']
                new_row[f"{indices_name}"] = row[f"{indices_name}_{year}"]
                new_row[f"{indices_name}_anomaly"] = row[f"{indices_name}_anomaly_{year}"]
                new_row["tp_spring"] = row[f'tp_spring_{year}']
                new_row["tp_summer"] = row[f'tp_summer_{year}']
                new_row["tp_autumn"] = row[f'tp_autumn_{year}']
                new_row[f'tp_winter'] = row[f'tp_winter_{year}']
                new_row[f'vpd'] = row[f'vpd_annual_{year}']
                new_row["tm"] = row[f'tm_annual_{year}']
                new_row["spei"] = row[f'spei_03_annual_spei_{year}']

                new_row["tp_spring_anomaly"] = row[f'tp_spring_anomaly_{year}']
                new_row["tp_summer_anomaly"] = row[f'tp_summer_anomaly_{year}']
                new_row["tp_autumn_anomaly"] = row[f'tp_autumn_anomaly_{year}']
                new_row[f'tp_winter_anomaly'] = row[f'tp_winter_anomaly_{year}']
                new_row[f'vpd_annual_anomaly'] = row[f'vpd_annual_anomaly_{year}']
                new_row["tm_annual_anomaly"] = row[f'tm_annual_anomaly_{year}']
                new_row["spei_03_annual_spei_anomaly"] = row[f'spei_03_annual_spei_anomaly_{year}']
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


if __name__ == '__main__':

    legacy = legacy_effects()

    '''# 3. 随机森林建模,对所有数据'''
    # data_path = f'afterfirst_revision_summary_high_elevation_broad.csv'
    # indices_name = 'sif'
    # time_information = {'evi':{'start':2001,'end':2023},'sif':{'start':2001,'end':2016}}
    # legacy.rf_modelling_all(data_path,
    #                         time_information[indices_name]['start'],
    #                         time_information[indices_name]['end'],
    #                         indices_name,
    #                         'high_elevation_broad')


    '''# 随机森林建模，采用group进行'''
    # data_path = f'afterfirst_revision_summary_low_elevation_everygreen.csv'
    # indices_name = 'sif'
    # time_information = {'evi':{'start':2001,'end':2023},'sif':{'start':2001,'end':2016}}
    # legacy.rf_modelling_group(data_path,
    #                         time_information[indices_name]['start'],
    #                         time_information[indices_name]['end'],
    #                         indices_name,
    #                         'low_elevation_everygreen')
    '''# 随机森林建模，采用group进行,分biogeo'''
    # biogeos = [2,3,4,5,7,8]
    # for biogeo in biogeos:
    #     data_path = f'afterfirst_revision_summary_high_elevation_broad.csv'
    #     indices_name = 'sif'
    #     time_information = {'evi':{'start':2001,'end':2023},'sif':{'start':2001,'end':2016}}
    #     legacy.rf_modelling_group_biogeo(data_path,
    #                             time_information[indices_name]['start'],
    #                             time_information[indices_name]['end'],
    #                             indices_name,
    #                             'high_elevation_broad',biogeo)


    '''# 随机森林建模，空间化'''
    # data_path = f'afterfirst_revision_summary_high_elevation_everygreen.csv'
    # indices_name = 'sif'
    # time_information = {'evi':{'start':2001,'end':2023},'sif':{'start':2001,'end':2016}}
    # for season in ['spring','summer','autumn','winter']:
    #     legacy.rf_modelling_all_forspatial(data_path,
    #                             time_information[indices_name]['start'],
    #                             time_information[indices_name]['end'],
    #                             indices_name,

    #                             'high_elevation_everygreen',season)
