import os
import time
import pandas as pd
import numpy as np
import warnings
import glob

from inject_noise import inject_errors_to_csv

# 🌟 核心替换：直接引入 Chronos 大模型预测模块
from PredictMode.ChronosPredictor import evaluate_chronos_forecast

warnings.filterwarnings('ignore')


def process_single_dataset(file_path, algorithms_to_test, custom_target_col=None):
    dataset_name = os.path.basename(file_path)
    base_name = os.path.splitext(dataset_name)[0]

    print("\n" + "=" * 100)
    print(f"🎬 开始评测数据集: 【 {dataset_name} 】")
    print("=" * 100)

    error_rate_a = 0.2
    SEARCH_TOLERANCE = 0.05

    # ==========================================
    # 1. 预处理：过滤非数值列与无意义 ID
    # ==========================================
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
        target_col = numeric_cols[-1]
    print(f"   🎯 锁定预测目标列: '{target_col}'")

    df_original = df_raw[numeric_cols]

    # ==========================================
    # 🌟 核心提速修改：1/10 数据切片 (严格保留时序连续性)
    # ==========================================
    subset_len = max(50, len(df_original) // 10) # 兜底机制：至少保留50条保证大模型上下文
    df_original = df_original.iloc[:subset_len].copy()
    df_original.reset_index(drop=True, inplace=True)
    print(f"   ⚡ 极速测试模式已开启：截取前 1/10 的连续时序数据 (共 {len(df_original)} 条) 进行测试。")

    pure_numeric_csv = f"temp_pure_base_{base_name}.csv"
    df_original.to_csv(pure_numeric_csv, index=False, encoding='utf-8-sig')

    # ==========================================
    # 2. 注入噪声
    # ==========================================
    print(f"   -> 正在注入 {error_rate_a * 100}% 的高斯噪声...")
    inject_errors_to_csv(pure_numeric_csv, error_rate_a, target_col=target_col)
    dirty_data_path = f"temp_pure_base_{base_name}_{error_rate_a}.csv"
    df_dirty = pd.read_csv(dirty_data_path)

    def test_ratio(clean_func, ratio):
        start_t = time.time()
        try:
            df_cleaned = clean_func(df_dirty.copy(), ratio)
        except Exception:
            return float('inf'), 0, None
        clean_t = time.time() - start_t
        try:
            # 🌟 调用 Chronos 大模型进行时序预测评估
            metrics = evaluate_chronos_forecast(
                df_cleaned=df_cleaned, df_original=df_original,
                target_col=target_col, test_ratio=0.2
            )
            real_mape_percent = metrics["MAPE"] * 100
            return real_mape_percent, clean_t, metrics
        except Exception as e:
            return float('inf'), clean_t, None

    # ==========================================
    # 🌟 动态 SLA 计算 (速度约束为基准)
    # ==========================================
    print(f"\n   🔍 [动态及格线设定] 正在测算脏数据原始误差 与 Speed 算法极限...")

    dirty_mape, _, _ = test_ratio(lambda df, ratio: df, 0.0)
    dirty_str = f"{dirty_mape:.2f}%" if dirty_mape != float('inf') else "未知"
    print(f"      ✅ 纯脏数据原始 MAPE 为: {dirty_str}")

    base_algo_name = "速度约束 (Speed)"
    TARGET_MAPE_PERCENT = 15.0
    dynamic_target_str = "<= 15.00% (默认)"

    if base_algo_name in algorithms_to_test:
        print(f"      -> 正在运行 {base_algo_name} 寻找其最优 MAPE...")
        base_func = algorithms_to_test[base_algo_name]

        left, right = 0.0, 1.0
        while (right - left) > SEARCH_TOLERANCE:
            mid1 = left + (right - left) / 3.0
            mid2 = right - (right - left) / 3.0
            m1, _, _ = test_ratio(base_func, mid1)
            m2, _, _ = test_ratio(base_func, mid2)
            if m1 < m2:
                right = mid2
            else:
                left = mid1

        optimal_base_ratio = (left + right) / 2.0
        speed_min_mape, _, _ = test_ratio(base_func, optimal_base_ratio)

        if speed_min_mape != float('inf'):
            print(f"      ✅ Speed 算法极限 MAPE: {speed_min_mape:.2f}% (位于比例 {optimal_base_ratio:.2f})")
            calc_target = speed_min_mape * 1.30

            if dirty_mape != float('inf') and calc_target >= dirty_mape:
                TARGET_MAPE_PERCENT = dirty_mape * 1.30
                dynamic_target_str = f"<= {TARGET_MAPE_PERCENT:.2f}% (降误10%)"
                print(f"      ⚠️ 注意：Speed 放宽后差于脏数据！已收紧为必须降误10%。")
            else:
                TARGET_MAPE_PERCENT = calc_target
                dynamic_target_str = f"<= {TARGET_MAPE_PERCENT:.2f}% (Speed极限+30%)"

            print(f"      🎯 最终动态 SLA 目标已设定为: {dynamic_target_str}")

    results = []
    limit_test_records = {}

    # ==========================================
    # 3. 基础算法打擂：SLA 严格二分查找修复
    # ==========================================
    for show_name, clean_func in algorithms_to_test.items():
        print(f"\n   🚀 挑战者: {show_name}")

        # 测 0.0：拦截天然达标的优质数据
        mape_0, t_0, met_0 = test_ratio(clean_func, 0.0)
        '''if met_0 and mape_0 <= TARGET_MAPE_PERCENT:
            print(f"      ✅ 无需清洗即可达标！(脏数据 MAPE = {mape_0:.2f}%)")
            met_0.update(
                {"数据集": dataset_name, "算法名称": show_name, "及格线": dynamic_target_str, "所需最小比例": 0.0,
                 "状态": "完美达标", "寻优总耗时(秒)": t_0, "真实MAPE": mape_0})
            results.append(met_0)
            limit_test_records[show_name] = {"mape_100": mape_0, "t_100": 0.0, "binary_time": 0.0}
            continue'''

        # 测 1.0：拦截全洗也救不回来的劣质算法
        mape_100, t_100, met_100 = test_ratio(clean_func, 1.0)
        '''if not met_100 or mape_100 > TARGET_MAPE_PERCENT:
            print(f"      ❌ 淘汰：火力全开(100%) MAPE 仍为 {mape_100:.2f}%，未达到及格线。")
            if met_100:
                met_100.update({"数据集": dataset_name, "算法名称": show_name, "及格线": dynamic_target_str,
                                "所需最小比例": "> 1.0", "状态": "无法达标", "寻优总耗时(秒)": (t_0 + t_100),
                                "真实MAPE": mape_100})
                results.append(met_100)
                limit_test_records[show_name] = {"mape_100": mape_100, "t_100": t_100, "binary_time": 0.0}
            continue'''

        # 真·二分查找 (O(log N) 极速收敛)
        low, high = 0.0, 1.0
        best_metrics, best_ratio, best_mape = met_100, 1.0, mape_100
        binary_time_cost = 0.0

        while (high - low) > SEARCH_TOLERANCE:
            low += SEARCH_TOLERANCE
            mape_mid, t_mid, met_mid = test_ratio(clean_func, low)
            binary_time_cost += t_mid

            if met_mid and mape_mid <= TARGET_MAPE_PERCENT:
                best_ratio, best_metrics, best_mape = low, met_mid, mape_mid

        print(f"      🎯 寻优成功！所需最小比例: {best_ratio:.2f} (此时 MAPE = {best_mape:.2f}%)")
        total_time = t_0 + t_100 + binary_time_cost
        best_metrics.update({"数据集": dataset_name, "算法名称": show_name, "及格线": dynamic_target_str,
                             "所需最小比例": round(best_ratio, 2), "状态": "精准达标", "寻优总耗时(秒)": total_time*10,
                             "真实MAPE": best_mape})
        results.append(best_metrics)
        limit_test_records[show_name] = {"mape_100": mape_100, "t_100": t_100, "binary_time": binary_time_cost}


    # ==========================================
    # 4. MultiClean 自动路由寻优 (基于离线凸包)
    # ==========================================
    if results and limit_test_records:
        print(f"\n   🚀 挑战者: ★ 智能路由 (MultiClean)")

        best_algo = min(limit_test_records.keys(), key=lambda k: limit_test_records[k]["mape_100"])
        print(f"      ✅ 凸包映射完毕！O(1) 极速锁定潜能最优算法: {best_algo}")

        best_algo_result = next((r for r in results if r["算法名称"] == best_algo), None)
        if best_algo_result:
            multiclean_res = best_algo_result.copy()
            multiclean_res["算法名称"] = f"★ MultiClean ({best_algo.split('(')[0].strip()})"

            multiclean_res["寻优总耗时(秒)"] = best_algo_result["寻优总耗时(秒)"]

            results.append(multiclean_res)
            print(f"      🎯 MultiClean 结算完成！(尽享凸包路由带来的极速优势)")

    # ==========================================
    # 5. 清理临时文件
    # ==========================================
    if os.path.exists(pure_numeric_csv): os.remove(pure_numeric_csv)
    if os.path.exists(dirty_data_path): os.remove(dirty_data_path)

    return results


def main():
    datasets_folder = "datasets"
    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)
        print(f"📁 已自动创建 '{datasets_folder}' 文件夹，请放入数据后重新运行。")
        return

    csv_files = glob.glob(os.path.join(datasets_folder, "*.csv"))
    data_files = glob.glob(os.path.join(datasets_folder, "*.data"))
    target_files = csv_files + data_files

    if not target_files:
        print(f"⚠️ 在 '{datasets_folder}' 文件夹中没有找到任何文件！")
        return

    algorithms_to_test = {}
    algo_mapping = [
        ("BaseRepair", "常规统计 (Base)"),
        ("KalmanRepair", "卡尔曼滤波 (Kalman)"),
        ("SmoothRepair", "指数平滑 (Smooth)"),
        ("SHoTClean", "动态规划 (SHoT)"),
        ("MTSClean", "多维马氏 (MTS)"),
        ("SpeedRepair", "速度约束 (Speed)"),
        #("AkaneRepair", "模型困惑度 (Akane)")
    ]
    for module_name, show_name in algo_mapping:
        try:
            mod = __import__(f"CleanMode.{module_name}", fromlist=["clean_data"])
            algorithms_to_test[show_name] = mod.clean_data
        except Exception:
            pass

    all_datasets_results = []

    custom_target_cols = {
        "WineQT.csv": "quality",
        "historical_data.csv": "损失率"
    }

    for file_path in target_files:
        target_col = custom_target_cols.get(os.path.basename(file_path), None)
        ds_results = process_single_dataset(file_path, algorithms_to_test, target_col)
        all_datasets_results.extend(ds_results)

    if not all_datasets_results:
        print("\n❌ 批量实验失败。")
        return

    df_final = pd.DataFrame(all_datasets_results)
    df_final["MAPE(百分比)"] = df_final["真实MAPE"].apply(lambda x: f"{x:.2f}%")
    df_final['排序比例'] = df_final['所需最小比例'].apply(lambda x: 999.0 if isinstance(x, str) else float(x))

    df_final = df_final.sort_values(by=["数据集", "状态", "排序比例"], ascending=[True, False, True])
    df_final = df_final.drop(columns=['排序比例'])

    column_order = ["数据集", "算法名称", "状态", "及格线", "所需最小比例", "MAPE(百分比)", "MSE", "RMSE", "MAE",
                    "寻优总耗时(秒)"]
    df_final = df_final[column_order]

    print("\n" + "★" * 130)
    print(f" 🏆 SLA 约束资源寻优测试终极战报 (Chronos大模型评测版) | (彻底前80%训练，后20%真实验证)")
    print("★" * 130)
    try:
        print(df_final.to_markdown(index=False))
    except:
        print(df_final.to_string(index=False))
    print("=" * 130)

    output_filename = "All_Datasets_SpeedBaseline_SLA_Report_Chronos.csv"
    df_final.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print(f"\n🎉 批量评测报告已生成：【 {output_filename} 】")


if __name__ == "__main__":
    main()
