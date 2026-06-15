import os
import time
import pandas as pd
import numpy as np

# 导入流水线函数
from run import evaluate_cleaning_algorithm

# 💡 新增：导入你要测试的具体清洗算法（这里以基准的 BaseRepair 为例）
# 请确保你的 CleanMode 文件夹下有 BaseRepair.py
from CleanMode.BaseRepair import clean_data as base_repair_func


def run_ratio_time_experiment():
    # ==========================================
    # 1. 实验参数配置
    # ==========================================
    fixed_error_rate_a = 0.3
    clean_ratios = [round(x * 0.1, 1) for x in range(0, 11)]

    input_csv = 'historical_data.csv'
    target_col = '损失率'

    output_dir = "evaluation_results_time"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 已确认结果存放文件夹: {output_dir}\n")

    # ==========================================
    # 2. 初始化数据存储
    # ==========================================
    experiment_results = []
    total_runs = len(clean_ratios)

    print("=" * 60)
    print(f"⏱️ 开始执行清洗比例与时间效率评估 | 固定错误率: {fixed_error_rate_a}")
    print("=" * 60)

    # ==========================================
    # 3. 循环遍历清洗比例并记录时间
    # ==========================================
    for i, b in enumerate(clean_ratios):
        print(f"\n[{i + 1}/{total_runs}] 正在测试清洗比例: b = {b} ...")

        start_time = time.time()  # 🟢 掐表开始

        try:
            # ★ 核心修复：补充传入 clean_func 和 algo_name ★
            final_err, mse, rmse, rmsle, mae, mape = evaluate_cleaning_algorithm(
                input_csv=input_csv,
                error_rate_a=fixed_error_rate_a,
                clean_ratio_b=b,
                clean_func=base_repair_func,  # <--- 补上必填的清洗函数
                algo_name="基础中位数/众数修复",  # <--- 补上名字
                target_col=target_col
            )
        except Exception as e:
            print(f"   ⚠️ 比例 {b} 运行失败 ({e})，记录为缺失值。")
            final_err, mse, rmse, rmsle, mae, mape = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        end_time = time.time()  # 🔴 掐表结束

        elapsed_time = end_time - start_time
        print(f"   ⏳ 本轮耗时: {elapsed_time:.4f} 秒")

        # 组装数据
        row_data = {
            "清洗比例": b,
            "残余错误率": final_err,
            "MSE": mse,
            "RMSE": rmse,
            "RMSLE": rmsle,
            "MAE": mae,
            "MAPE": mape,
            "耗时(秒)": elapsed_time
        }

        experiment_results.append(row_data)

    # ==========================================
    # 4. 导出结果为 CSV
    # ==========================================
    print(f"\n💾 测试完成！正在导出包含时间效率的最终报告...")
    df_results = pd.DataFrame(experiment_results)

    file_path = os.path.join(output_dir, f"Ratio_Time_Efficiency_a{fixed_error_rate_a}.csv")
    df_results.to_csv(file_path, index=False, encoding='utf-8-sig')

    print(f"🎉 报告已成功生成: {file_path}")


if __name__ == "__main__":
    run_ratio_time_experiment()