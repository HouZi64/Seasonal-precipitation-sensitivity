#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 5/22/2024 15:56
# @Author :
import time,math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
os.environ["OMP_NUM_THREADS"] = "8"       # 5850U：8 线程最稳（对应8物理核）
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import shap
from sklearn.metrics import make_scorer, r2_score
from matplotlib.colors import LogNorm
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



    def random_forest_co_kfold(self,target,features,df,year,indices_name,scenario):
        def scatter_density_plot(y_test, y_pred, mse, mae, r2, output_path, year):
            plt.figure(figsize=(8, 6), constrained_layout=True)  # 使用 constrained_layout
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
            output_path = 'after_first_revision/images window'
        os.makedirs(output_path, exist_ok=True)
        # 载入数据
        df_new = df.copy()
        features_new = features.copy()[0:-6]
        y = df_new[target]
        X = df_new[features]
        season_name = features_new[0].split('_')[1]
        if season_name not in ['spring', 'summer', 'autumn', 'winter']: assert 'wrong'
        # 定义 K 折交叉验证
        groups = df['gridid'].to_numpy()
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=5)
        # kf = KFold(n_splits=10, shuffle=True, random_state=42)
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
        for train_index, val_index in gkf.split(X, y, groups=groups):
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
                                           # 'year',
                                           'biogeo',
                                           'country_chunk'
                                           ]]
            data_shap[f'tp_spring_relative_shap_ratio'] = shap_values[:,0] / (np.sum(np.abs(shap_values),axis=1))
            data_shap[f'tp_summer_relative_shap_ratio'] = shap_values[:, 1] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'tp_autumn_relative_shap_ratio'] = shap_values[:, 2] / (np.sum(np.abs(shap_values), axis=1))
            data_shap[f'tp_winter_relative_shap_ratio'] = shap_values[:, 3] / (np.sum(np.abs(shap_values), axis=1))
            denom = np.sum(np.abs(shap_values), axis=1)
            denom[denom == 0] = np.finfo('float32').eps
            # data_shap[f'tp_{season_name}_relative_shap_ratio']  = shap_values[:, 0] / denom
            data_shap[f'vpd_{season_name}_relative_shap_ratio'] = shap_values[:, 4] / denom
            data_shap[f'tm_{season_name}_relative_shap_ratio']  = shap_values[:, 5] / denom
            data_shap[f'tp_spring_shap_value'] = shap_values[:,0]
            data_shap[f'tp_summer_shap_value'] = shap_values[:, 1]
            data_shap[f'tp_autumn_shap_value'] = shap_values[:, 2]
            data_shap[f'tp_winter_shap_value'] = shap_values[:, 3]
            # data_shap[f'tp_{season_name}_shap_value']  = shap_values[:, 0]
            data_shap[f'vpd_{season_name}_shap_value'] = shap_values[:, 4]
            data_shap[f'tm_{season_name}_shap_value']  = shap_values[:, 5]
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

    def stratified_cap_equal(self,df, strat_col='biogeo', total_n=30000, random_state=42):
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

        # 生成年份列表
        years = np.arange(start_year, end_year + 1)
        # seasons = ['spring', 'summer', 'autumn', 'winter']
        # from sklearn.linear_model import LinearRegression
        # # 1) 对每个季节的降水做正交化：P_s ~ T_s + VPD_s (+ 慢变量)
        # P_resid = {}
        # for s in seasons:
        #     Xs = filtered_df[[f'tm_{s}_baseline', f'vpd_{s}_baseline']]  # 可按需添加 'elev'、辐射、光周期等
        #     ys = filtered_df[f'tp_{s}_baseline']
        #
        #     # 线性版（解释性强）；也可换成非线性随机森林版（对非线性更稳健）
        #     g = LinearRegression()  # or RandomForestRegressor(n_estimators=400, max_depth=None, random_state=42)
        #     g.fit(Xs, ys)
        #     yhat = g.predict(Xs)
        #
        #     P_resid[s] = ys - yhat
        # filtered_df = pd.concat([pd.DataFrame({f'Pres_{s}': P_resid[s] for s in seasons}),filtered_df],axis=1)
        # filtered_df = filtered_df[
        #     (filtered_df['biogeo'] == 5)
        #     &(filtered_df['country'].isin(['france','germany','belgium','austria','croatia','slovenia']))
        #       ]
        filtered_df_sub = self.stratified_cap_equal(filtered_df)
        data_stack = self.stack_data(filtered_df_sub,start_year,end_year,indices_name)
        df = data_stack.copy()
        # === 拆 gridid ===
        df[['country', 'row', 'col']] = df['gridid'].str.split('_', expand=True)
        df['row'] = df['row'].astype(int);
        df['col'] = df['col'].astype(int)

        # === 1) 抽栅格（分层采样；这里只示范按经纬度粗分层） ===
        # 用更精细的分层：可加 elevation/多年均降水的 quantile bins
        lat_bin = pd.qcut(df['row'], q=20, duplicates='drop')
        lon_bin = pd.qcut(df['col'], q=20, duplicates='drop')
        strata = lat_bin.astype(str) + '_' + lon_bin.astype(str)

        # 每层抽固定数量的 gridid（例如总计 ~3000 个栅格）
        # 先把格点分层 → 每层抽配额 → 合成最终格点集合。
        target_cells = 7000 # 1000=快跑版；3000-5000=稳健版
        per_stratum = max(1, target_cells // strata.nunique())
        chosen_cells = (df
        .drop_duplicates('gridid')
        .groupby(strata)
        .apply(lambda g: g.sample(n=min(per_stratum, len(g)), random_state=42))
        .reset_index(drop=True)['gridid'])

        # === 2) 抽年份（10 个） ===
        years = np.sort(df['year'].unique())
        years_subset = np.linspace(years.min(), years.max(), num=10, dtype=int)

        # === 3) 得到子集 ===
        sub = df[df['gridid'].isin(chosen_cells) & df['year'].isin(years_subset)].copy()

        # data_stack = data_stack.sample(frac=1, ignore_index=True, random_state=42)
        # indices_ = {}
        # # Xs = data_stack[[f'tm_spring', f'tm_summer', f'tm_autumn', f'tm_winter',
        # #                   f'vpd_spring', f'vpd_summer', f'vpd_autumn',
        # #                   f'vpd_winter', ]]  # 可按需添加 'elev'、辐射、光周期等
        # Xs = data_stack[[f'tm',f'vpd']]  # 可按需添加 'elev'、辐射、光周期等
        # ys = data_stack[f'{indices_name}']
        #
        # from sklearn.linear_model import LinearRegression
        # g = LinearRegression()  # or RandomForestRegressor(n_estimators=400, max_depth=None, random_state=42)
        # g.fit(Xs, ys)
        # yhat = g.predict(Xs)
        # indices_[f'{indices_name}_'] = ys - yhat
        # data_stack = pd.concat([pd.DataFrame({f'{indices_name}_': indices_[f'{indices_name}_']}), data_stack], axis=1)
        target = f'{indices_name}_anomaly'
        # 干旱指数建模
        season_name = 'winter'
        # features = [
        #     # 'tp_{season_name}_baseline',
        #             f'tp_spring_baseline',
        #             f'tp_summer_baseline',
        #             f'tp_autumn_baseline',
        #             f'tp_winter_baseline',
        #
        #             # f'vpd_{season_name}_baseline',
        #             # f'tm_{season_name}_baseline',
        #             # f'spei_03_annual_spei_anomaly',
        #             'weights',
        #             'row','col','country',
        #             # 'year',
        #             'biogeo',
        #             'country_chunk']
        season = 'all'
        features = [
            # 'tp_{season_name}_baseline',
                    f'tp_spring',
                    f'tp_summer',
                    f'tp_autumn',
                    f'tp_winter',
                    # 'biogeo',
                    # f'vpd_spring_baseline',
                    # f'vpd_summer_baseline',
                    # f'vpd_autumn_baseline',
                    # f'vpd_winter_baseline',
                    #
                    # f'tm_spring_baseline',
                    # f'tm_summer_baseline',
                    # f'tm_autumn_baseline',
                    # f'tm_winter_baseline',
                    f'vpd',
                    f'tm',

                    # f'vpd_{season_name}_baseline',
                    # f'tm_{season_name}_baseline',
                    # f'spei_03_annual_spei_anomaly',
                    'weights',
                    'row','col','country',
                    # 'year',
                    'biogeo',
                    'country_chunk']
        # keep_ratio_8 = 0.5  # 保留比例（比如 0.7=保留70%，即减少30%）
        # seed_8 = 2025  # 控制“去掉哪一部分”的随机种子
        #
        # m8 = filtered_df['biogeo'].astype(str).eq('8')
        # df_out = (pd.concat([filtered_df.loc[~m8],
        #                      filtered_df.loc[m8].sample(frac=keep_ratio_8, random_state=seed_8)],
        #                     axis=0)
        #           .sample(frac=1, random_state=seed_8)  # 可选：最后整体打乱
        #           .reset_index(drop=True))
        self.random_forest_co_kfold(target, features, sub, 'Temporal Series', indices_name, scenario)
    def stack_data(self,input_data,start_year,end_year,indices_name):
        new_data = []
        for index, row in input_data.iterrows():
            for year in range(start_year,end_year):
                new_row = {}
                new_row["gridid"] = f"{row['country']}_{row['row']}_{row['col']}"
                new_row["year"] = year
                new_row['row'] = row['row']
                new_row['col'] = row['col']
                new_row['weights'] = row['weights']
                new_row["country"] = row['country']
                new_row["country_chunk"] = row['country_chunk']
                new_row["biogeo"] = row['biogeo']
                new_row[f"{indices_name}"] = row[f"{indices_name}_{year}"]
                new_row[f"{indices_name}_anomaly"] = row[f"{indices_name}_anomaly_{year}"]
                new_row["tp_spring"] = row[f'tp_spring_{year}']
                new_row["tp_summer"] = row[f'tp_summer_{year}']
                new_row["tp_autumn"] = row[f'tp_autumn_{year}']
                new_row[f'tp_winter'] = row[f'tp_winter_{year-1}']

                new_row["vpd_spring"] = row[f'vpd_spring_{year}']
                new_row["vpd_summer"] = row[f'vpd_summer_{year}']
                new_row["vpd_autumn"] = row[f'vpd_autumn_{year}']
                new_row[f'vpd_winter'] = row[f'vpd_winter_{year-1}']

                new_row["tm_spring"] = row[f'tm_spring_{year}']
                new_row["tm_summer"] = row[f'tm_summer_{year}']
                new_row["tm_autumn"] = row[f'tm_autumn_{year}']
                new_row[f'tm_winter'] = row[f'tm_chilling_{year-1}']


                new_row[f'vpd'] = row[f'vpd_annual_{year}']
                new_row["tm"] = row[f'tm_annual_{year}']
                new_row["spei"] = row[f'spei_03_annual_spei_{year}']

                # new_row["tp_spring_anomaly"] = row[f'tp_spring_anomaly_{year}']
                # new_row["tp_summer_anomaly"] = row[f'tp_summer_anomaly_{year}']
                # new_row["tp_autumn_anomaly"] = row[f'tp_autumn_anomaly_{year}']
                # new_row[f'tp_winter_anomaly'] = row[f'tp_winter_anomaly_{year}']
                # new_row[f'vpd_annual_anomaly'] = row[f'vpd_annual_anomaly_{year}']
                # new_row["tm_annual_anomaly"] = row[f'tm_annual_anomaly_{year}']
                # new_row["spei_03_annual_spei_anomaly"] = row[f'spei_03_annual_spei_anomaly_{year}']
                new_data.append(new_row)
        new_df = pd.DataFrame(new_data)
        return new_df



if __name__ == '__main__':

    legacy = legacy_effects()

    '''# 3. 随机森林建模,对所有数据'''
    data_path = f'afterfirst_revision_summary_lowelevation_broad.csv'
    indices_name = 'evi'
    time_information = {'evi':{'start':2001,'end':2023},'sif':{'start':2001,'end':2016}}
    legacy.rf_modelling_all(data_path,
                            time_information[indices_name]['start'],
                            time_information[indices_name]['end'],
                            indices_name,
                            'lowelevation_broad')

