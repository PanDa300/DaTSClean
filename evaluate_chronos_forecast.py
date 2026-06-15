import os
import time
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# 🌟 核心替换：引入你新写好的 Chronos 大模型预测器
from PredictMode.ChronosPredictor import evaluate_chronos_forecast

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def run_grid_experiments():
    # ==========================================
    # 1. 实验参数配置
    # ==========================================
    error_rates = [round(x * 0.1, 1) for x in range(1, 10)]  # 0.1 到 0.9
    clean_ratios = [round(x * 0.1, 1) for x in range(0, 11)]  # 0.0 到 1.0

    input_csv = 'historical_data.csv'
    target_col = '损失率'
    algo_name = "KalmanRepair"  # 基础清洗算法

    output_dir = "evaluation_results_matrix"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 已确认结果存放文件夹: {output_dir}\n")

    # ==========================================
    # 2. 预处理与构建【全局固定扰动池】
    # ==========================================
    if not os.path.exists(input_csv):
        print(f"❌ 找不到数据文件 {input_csv}")
        return

    df_raw = pd.read_csv(input_csv)
    df_raw.columns = df_raw.columns.str.strip()
    numeric_cols = df_raw.select_dtypes(include=[np.number, 'bool']).columns.tolist()
    for bad_col in ['Id', 'id', 'ID', 'Unnamed: 0', '序号']:
        if bad_col in numeric_cols: numeric_cols.remove(bad_col)
    if target_col not in numeric_cols and target_col in df_raw.columns:
        numeric_cols.append(target_col)

    df_original = df_raw[numeric_cols]

    # ==========================================
    # 🌟 核心提速修改：1/10 数据切片 (严格保留时序连续性)
    # ==========================================
    subset_len = max(50, len(df_original) // 10)  # 兜底机制，最少保留50条，否则 Chronos 无法切分
    df_original = df_original.iloc[:subset_len].copy()
    df_original.reset_index(drop=True, inplace=True)
    print(f"⚡ 极速测试模式已开启：截取前 1/10 的连续时序数据 (共 {len(df_original)} 条) 进行测试。")

    # 计算累积噪声基础变量
    total_rows = len(df_original)
    target_std = df_original[target_col].std()

    # 锁定随机种子，打乱索引与生成固定噪声
    np.random.seed(42)
    shuffled_indices = np.random.permutation(df_original.index)
    np.random.seed(42)
    global_noise_array = np.random.normal(loc=0, scale=target_std * 2.0, size=total_rows)

    # 动态加载清洗算法
    try:
        mod = __import__(f"CleanMode.{algo_name}", fromlist=["clean_data"])
        clean_func = mod.clean_data
    except Exception as e:
        print(f"❌ 算法 {algo_name} 加载失败: {e}")
        return

    results = {"MSE": [], "RMSE": [], "RMSLE": [], "MAE": [], "MAPE": []}
    total_runs = len(error_rates) * len(clean_ratios)
    current_run = 0

    # ==========================================
    # 3. 嵌套循环执行交叉实验
    # ==========================================
    for a in error_rates:
        print(f"\n" + "★" * 50)
        print(f"🦠 当前底座错误率 a={a} 开始测试 (严格累积注入)...")
        print("★" * 50)

        # 提取对应比例的索引，并在纯净数据上叠加固定噪声
        num_errors_to_inject = int(total_rows * a)
        current_dirty_indices = shuffled_indices[:num_errors_to_inject]

        df_dirty = df_original.copy()
        if num_errors_to_inject > 0:
            df_dirty.loc[current_dirty_indices, target_col] += global_noise_array[:num_errors_to_inject]

        for b in clean_ratios:
            current_run += 1
            print(f"   -> [{current_run}/{total_runs}] 正在应用清洗比例 b={b} ...")

            try:
                # 内存清洗
                df_cleaned = clean_func(df_dirty.copy(), b)

                # 🌟 核心替换：调用 Chronos 大模型进行评估
                met = evaluate_chronos_forecast(df_cleaned, df_original, target_col, 0.2)
                mse, rmse, rmsle, mae, mape = met["MSE"], met["RMSE"], met["RMSLE"], met["MAE"], met["MAPE"]

                # 放大结果 10 倍方便展示
                mse *= 10
                rmse *= 10
                rmsle *= 10
                mae *= 10
                mape *= 10

            except Exception as e:
                print(f"      ⚠️ 本次配置计算失败 ({e})，将记录为缺失值。")
                mse, rmse, rmsle, mae, mape = np.nan, np.nan, np.nan, np.nan, np.nan

            # 记录核心坐标和各项指标
            base_info = {"注入错误率": a, "清洗比例": b}
            results["MSE"].append({**base_info, "MSE": mse})
            results["RMSE"].append({**base_info, "RMSE": rmse})
            results["RMSLE"].append({**base_info, "RMSLE": rmsle})
            results["MAE"].append({**base_info, "MAE": mae})
            results["MAPE"].append({**base_info, "MAPE": mape})

    # ==========================================
    # 4. 生成二维矩阵、热力图与多线折线图
    # ==========================================
    print(f"\n💾 测试完成！正在生成多维分析图表，并导出到 {output_dir} 文件夹中...")

    for metric_name, data_list in results.items():
        if not data_list: continue

        df_flat = pd.DataFrame(data_list)

        # ------------------------------------------
        # 4.1 生成二维矩阵 CSV 与 热力图 (Heatmap)
        # ------------------------------------------
        df_matrix = df_flat.pivot(index="注入错误率", columns="清洗比例", values=metric_name)
        df_matrix.index.name = "注入错误率 (A)"
        df_matrix.columns.name = "清洗比例 (B)"

        csv_path = os.path.join(output_dir, f"Grid_Matrix_{metric_name}_x10.csv")
        df_matrix.to_csv(csv_path, encoding='utf-8-sig')

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            df_matrix,
            annot=True,
            fmt=".2f" if metric_name == "MAPE" else ".4f",
            cmap="YlGnBu",
            cbar_kws={'label': f'{metric_name} 值'}
        )

        plt.title(f"下游任务(Chronos) {metric_name} 随错误率与清洗比例变化热力图 (x10)", fontsize=18, pad=15)
        plt.xlabel("清洗比例 (B)", fontsize=15)
        plt.ylabel("注入错误率 (A)", fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Heatmap_{metric_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # ------------------------------------------
        # 4.2 🌟 绘制学术级参数敏感性多线折线图
        # ------------------------------------------
        plt.figure(figsize=(10, 7))

        df_flat['清洗比例标签'] = df_flat['清洗比例'].apply(lambda x: f"比例: {x:.1f}")

        sns.lineplot(
            data=df_flat,
            x="注入错误率",
            y=metric_name,
            hue="清洗比例标签",
            palette="viridis",
            marker="o",
            linewidth=2,
            markersize=6
        )

        plt.title(f"{metric_name} 随注入错误率变化趋势 (不同清洗比例对比, x10)", fontsize=16, pad=15)
        plt.xlabel("注入错误率", fontsize=15)

        ylabel = f"{metric_name} (百分比)" if metric_name == "MAPE" else f"{metric_name} 值"
        plt.ylabel(ylabel, fontsize=15, labelpad=15)

        plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        plt.tick_params(labelsize=12)

        plt.legend(title="清洗比例配置", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, title_fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Sensitivity_Lines_{metric_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n🎉 大满贯！包含 Chronos 大模型推理的所有图表均已保存在【 {output_dir} 】中！")


if __name__ == "__main__":
    run_grid_experiments()