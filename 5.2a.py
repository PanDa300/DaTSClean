import os
import time
import pandas as pd
import numpy as np
import warnings
import glob

from inject_noise import inject_errors_to_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error
)

warnings.filterwarnings('ignore')


# ==========================================
# 🌟 内置安全评估器：80/20 划分 + WAPE (支持 0 值过滤)
# ==========================================
def evaluate_sequential_forecast(df_cleaned, df_original, target_col, test_ratio=0.2):
    split_idx = int(len(df_cleaned) * (1 - test_ratio))
    features = [c for c in df_cleaned.columns if c != target_col]

    X_train = df_cleaned.iloc[:split_idx][features].fillna(0).values
    y_train = df_cleaned.iloc[:split_idx][target_col].fillna(0).values
    X_test = df_cleaned.iloc[split_idx:][features].fillna(0).values
    y_test_true = df_original.iloc[split_idx:][target_col].fillna(0).values

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 🌟 核心修改 1：过滤掉真实值为 0 的异常数据点，防止指标爆炸
    valid_idx = (y_test_true != 0)
    y_test_true_valid = y_test_true[valid_idx]
    y_pred_valid = y_pred[valid_idx]

    # 兜底机制：如果测试集中所有值都是 0，则返回无效值
    if len(y_test_true_valid) == 0:
        return {"RMSE": np.nan, "MSE": np.nan, "MAE": np.nan, "RMSLE": np.nan, "MAPE": np.nan}

    mse = mean_squared_error(y_test_true_valid, y_pred_valid)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_true_valid, y_pred_valid)

    # MAPE 防爆炸计算
    sum_y_true = np.sum(np.abs(y_test_true_valid))
    mape = np.sum(np.abs(y_test_true_valid - y_pred_valid)) / sum_y_true if sum_y_true != 0 else 0.0

    # RMSLE 计算 (加 max(0) 防止预测出极小负数导致 log 报错)
    try:
        rmsle = np.sqrt(mean_squared_log_error(np.clip(y_test_true_valid, 0, None), np.clip(y_pred_valid, 0, None)))
    except Exception:
        rmsle = np.nan

    return {"RMSE": rmse, "MSE": mse, "MAE": mae, "RMSLE": rmsle, "MAPE": mape}


# ==========================================
# 🌟 单个数据集处理流水线
# ==========================================
def process_single_dataset(file_path, custom_target_col=None, algo_name="KalmanRepair"):
    dataset_name = os.path.basename(file_path)

    print("\n" + "=" * 80)
    print(f"🎬 开始处理数据集: 【 {dataset_name} 】")
    print("=" * 80)

    error_rates = [round(x * 0.1, 1) for x in range(1, 10)]  # 0.1 到 0.9 (Y轴：纵坐标)
    clean_ratios = [round(x * 0.1, 1) for x in range(0, 11)]  # 0.0 到 1.0 (X轴：横坐标)

    # 1. 预处理
    try:
        df_raw = pd.read_csv(file_path)
        df_raw.columns = df_raw.columns.str.strip()
    except Exception as e:
        print(f"❌ 读取 {dataset_name} 失败: {e}")
        return []

    numeric_cols = df_raw.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    for bad_col in ['Id', 'id', 'ID', 'Unnamed: 0', '序号']:
        if bad_col in numeric_cols:
            numeric_cols.remove(bad_col)

    target_col = custom_target_col
    if not target_col or target_col not in numeric_cols:
        if target_col in df_raw.columns:
            numeric_cols.append(target_col)
        else:
            target_col = numeric_cols[-1]

    print(f"   🎯 锁定预测目标列: '{target_col}'")
    df_original = df_raw[numeric_cols]

    # 极速测试模式
    original_len = len(df_original)
    SPEED_UP_FACTOR = 100
    subset_len = max(100, original_len // SPEED_UP_FACTOR)
    time_multiplier = original_len / subset_len

    df_original = df_original.iloc[:subset_len].copy()
    df_original.reset_index(drop=True, inplace=True)

    print(f"   ⚡ 极致算力平替：截取前 {subset_len} 条数据，耗时将自动补偿 {time_multiplier:.1f} 倍。")

    # 构建全局固定扰动池
    total_rows = len(df_original)
    target_std = df_original[target_col].std()

    np.random.seed(42)
    shuffled_indices = np.random.permutation(df_original.index)
    np.random.seed(42)
    global_noise_array = np.random.normal(loc=0, scale=target_std * 2.0, size=total_rows)

    # 动态加载清洗算法
    try:
        mod = __import__(f"CleanMode.{algo_name}", fromlist=["clean_data"])
        clean_func = mod.clean_data
    except Exception as e:
        print(f"❌ 算法加载失败: {e}")
        return []

    dataset_results = []

    # 2. 嵌套循环执行交叉实验
    for a in error_rates:
        print(f"   🔄 正在测试累积注入错误率: {a:.1f}")

        num_errors_to_inject = int(total_rows * a)
        current_dirty_indices = shuffled_indices[:num_errors_to_inject]

        df_dirty = df_original.copy()
        if num_errors_to_inject > 0:
            df_dirty.loc[current_dirty_indices, target_col] += global_noise_array[:num_errors_to_inject]

        for b in clean_ratios:
            start_t = time.time()
            try:
                df_cleaned = clean_func(df_dirty.copy(), b)
                met = evaluate_sequential_forecast(df_cleaned, df_original, target_col, 0.2)
                mse, rmse, rmsle, mae, mape = met["MSE"], met["RMSE"], met["RMSLE"], met["MAE"], met["MAPE"]

                mse *= 10
                rmse *= 10
                rmsle *= 10
                mae *= 10
                mape *= 10
            except Exception:
                mse, rmse, rmsle, mae, mape = np.nan, np.nan, np.nan, np.nan, np.nan

            cost_time = (time.time() - start_t) * time_multiplier

            # 🌟 核心修改 2：去除“算法名称”输出，保持表格纯净
            dataset_results.append({
                "数据集": dataset_name,
                "注入错误率": a,
                "清洗比例": b,
                "MSE (x10)": mse,
                "RMSE (x10)": rmse,
                "RMSLE (x10)": rmsle,
                "MAE (x10)": mae,
                "MAPE (x10)": mape,
                "耗时(秒)": cost_time
            })

    return dataset_results


# ==========================================
# 🌟 主控制端：扫描全集，输出矩阵大表
# ==========================================
def main():
    datasets_folder = "datasets"
    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)
        print(f"📁 已自动创建 '{datasets_folder}' 文件夹，请放入数据后重新运行。")
        return

    csv_files = glob.glob(os.path.join(datasets_folder, "*.csv"))+glob.glob(os.path.join(datasets_folder, "*.data"))
    if not csv_files:
        print(f"⚠️ 在 '{datasets_folder}' 文件夹中没有找到任何文件！")
        return

    # 默认跑一个底层算法（例如 KalmanRepair，因为最后输出不体现算法名字）
    algo_name_to_test = "KalmanRepair"

    custom_target_cols = {
        "WineQT.csv": "quality",
        "historical_data.csv": "损失率"
    }

    all_datasets_results = []

    # 3. 循环遍历每一个数据集
    for file_path in csv_files:
        target_col = custom_target_cols.get(os.path.basename(file_path), None)
        ds_results = process_single_dataset(file_path, custom_target_col=target_col, algo_name=algo_name_to_test)
        all_datasets_results.extend(ds_results)

    if not all_datasets_results:
        print("\n❌ 批量实验失败，未收集到任何数据。")
        return

    df_final = pd.DataFrame(all_datasets_results)

    # ==========================================
    # 🌟 核心修改 3：针对不同指标，输出对应的二维透视表
    # ==========================================
    metrics_to_export = ["MSE (x10)", "RMSE (x10)", "RMSLE (x10)", "MAE (x10)", "MAPE (x10)", "耗时(秒)"]

    print("\n" + "★" * 80)
    print("📈 正在生成各指标的独立二维矩阵表格...")

    for metric in metrics_to_export:
        if metric not in df_final.columns:
            continue

        # 使用 Pandas 的透视表功能：Y轴(纵坐标)=数据集与错误率，X轴(横坐标)=清洗比例
        df_pivot = df_final.pivot_table(
            index=["数据集", "注入错误率"],
            columns="清洗比例",
            values=metric
        )

        # 净化文件名
        safe_metric_name = metric.replace(" (x10)", "_x10").replace("(", "_").replace(")", "")
        out_filename = f"Matrix_Result_{safe_metric_name}.csv"

        df_pivot.to_csv(out_filename, encoding='utf-8-sig')
        print(f"   ✅ 已保存: 【 {out_filename} 】")

    print("★" * 80)
    print("🎉 大满贯！所有数据集处理完毕，全部矩阵图均已生成！")


if __name__ == "__main__":
    main()