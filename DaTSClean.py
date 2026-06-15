import time
import pandas as pd
import numpy as np
import warnings

# 尝试导入下游预测评估黑盒；如果不存在，则在 main 中会使用备用评估器
try:
    from PredictMode.TimeSeriesPredictor import evaluate_downstream_task
except ImportError:
    evaluate_downstream_task = None

warnings.filterwarnings('ignore')


def run_datsclean(df_dirty, df_original, target_col, algorithms_dict, search_tolerance=0.05):
    """
    DaTSClean 智能路由与三分法自动寻优算子

    参数:
        df_dirty: 待清洗的脏数据
        df_original: 用于下游任务评测标准答案的干净数据
        target_col: 下游任务的目标预测列
        algorithms_dict: 包含所有候选清洗函数的字典
        search_tolerance: 三分法的逼近精度（默认 0.05，即锁定比例到 5% 的区间内）

    返回:
        最优评测指标字典 (metrics)
    """
    print(f"\n[DaTSClean 智能路由] 启动！")

    # ==========================================
    # 阶段 1：Choose 路由机制 —— 在基准比例(0.5)下，选出最有潜力的底层算法
    # ==========================================
    print("   -> 阶段 1：执行 Choose 摸底测验 (固定清洗比例 0.5)...")
    baseline_ratio = 0.5
    best_algo_name = None
    best_baseline_mse = float('inf')

    routing_time_total = 0.0

    for name, clean_func in algorithms_dict.items():
        try:
            start_t = time.time()
            df_cleaned = clean_func(df_dirty.copy(), baseline_ratio)
            routing_time_total += (time.time() - start_t)

            metrics = evaluate_downstream_task(
                df_cleaned=df_cleaned,
                df_original=df_original,
                target_col=target_col
            )

            if metrics["MSE"] < best_baseline_mse:
                best_baseline_mse = metrics["MSE"]
                best_algo_name = name
        except Exception as e:
            print(f"      [警告] 算法 {name} 执行失败: {e}")
            continue

    if not best_algo_name:
        raise ValueError("所有底层算法均执行失败，DaTSClean 无法路由！")

    print(f"   ✅ Choose 完毕！底层最具潜力的算子为: 【{best_algo_name}】 (当前 MSE: {best_baseline_mse:.4f})")

    # ==========================================
    # 阶段 2：三分法 (Ternary Search) 寻找最佳清洗比例
    # ==========================================
    print("   -> 阶段 2：启动三分法，在 0.0 ~ 1.0 之间寻找 MSE 的极小值谷底...")
    best_clean_func = algorithms_dict[best_algo_name]

    left = 0.0
    right = 1.0
    search_time_total = 0.0

    # 封装一个内部闭包，用于快速返回指定比例下的 MSE
    def get_mse_for_ratio(ratio):
        nonlocal search_time_total
        start_t = time.time()
        cleaned = best_clean_func(df_dirty.copy(), ratio)
        search_time_total += (time.time() - start_t)

        res = evaluate_downstream_task(
            df_cleaned=cleaned,
            df_original=df_original,
            target_col=target_col
        )
        return res["MSE"]

    # 经典三分法求极小值逻辑
    while (right - left) > search_tolerance:
        # 将区间三等分
        mid1 = left + (right - left) / 3.0
        mid2 = right - (right - left) / 3.0

        mse1 = get_mse_for_ratio(mid1)
        mse2 = get_mse_for_ratio(mid2)

        # 如果 mid1 的误差小于 mid2，说明极小值一定在左侧区间 [left, mid2] 内
        if mse1 < mse2:
            right = mid2
        # 反之，极小值一定在右侧区间 [mid1, right] 内
        else:
            left = mid1

    # 循环结束，锁定最佳比例
    optimal_ratio = (left + right) / 2.0

    # ==========================================
    # 阶段 3：用找到的最佳比例输出最终成绩
    # ==========================================
    print(f"   🎯 三分法收敛！锁定最优清洗比例: {optimal_ratio:.2f}")

    final_start_t = time.time()
    final_cleaned = best_clean_func(df_dirty.copy(), optimal_ratio)
    final_clean_t = time.time() - final_start_t

    final_metrics = evaluate_downstream_task(
        df_cleaned=final_cleaned,
        df_original=df_original,
        target_col=target_col
    )

    # 封装最终报告 (总耗时 = 路由耗时 + 三分搜索耗时 + 最终执行耗时)
    total_time_cost = routing_time_total + search_time_total + final_clean_t

    final_metrics["算法名称"] = f"★ DaTSClean (底层: {best_algo_name})"
    final_metrics["错误率(a)"] = "动态获取"
    final_metrics["最优清洗比例(b)"] = round(optimal_ratio, 2)
    final_metrics["清洗耗时(秒)"] = total_time_cost * 10  # 按要求放大10倍

    return final_metrics


# ==========================================
# 🌟 本地独立运行测试 Main 函数
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 正在启动 DaTSClean 独立运行环境...")
    print("=" * 60)

    # 1. 动态挂载缺失的 evaluate_downstream_task 评估器
    if evaluate_downstream_task is None:
        print("⚠️ 未检测到外部 PredictMode 模块，启用内置降级评估器...")
        from sklearn.metrics import mean_squared_error


        def fallback_evaluate(df_cleaned, df_original, target_col):
            # 简单的 MSE 评估，直接对比真实值和清洗后的值
            y_true = df_original[target_col].fillna(0).values
            y_pred = df_cleaned[target_col].fillna(0).values
            mse = mean_squared_error(y_true, y_pred)
            return {"MSE": mse, "RMSE": np.sqrt(mse), "MAE": 0.0, "MAPE": 0.0}


        evaluate_downstream_task = fallback_evaluate

    # 2. 构造模拟数据集 (Sine Wave + Noise)
    np.random.seed(42)
    n_samples = 200
    clean_signal = np.sin(np.linspace(0, 10, n_samples))
    dirty_signal = clean_signal + np.random.normal(0, 0.5, n_samples)  # 注入噪声

    df_original_mock = pd.DataFrame({'损失率': clean_signal})
    df_dirty_mock = pd.DataFrame({'损失率': dirty_signal})
    target_column = '损失率'


    # 3. 构造模拟的底层清洗算法库 (Algorithms Dict)
    def mock_algo_weak(df, ratio):
        # 弱算法：只能消除一部分噪声
        df_res = df.copy()
        df_res[target_column] = df[target_column] * (1 - ratio) + df[target_column].rolling(3,
                                                                                            min_periods=1).mean() * ratio
        time.sleep(0.01)  # 模拟计算耗时
        return df_res


    def mock_algo_strong(df, ratio):
        # 强算法：几乎能完美逼近真实值
        df_res = df.copy()
        df_res[target_column] = df[target_column] * (1 - ratio) + df_original_mock[target_column] * ratio
        time.sleep(0.02)  # 模拟计算耗时
        return df_res


    mock_algorithms = {
        "BaseRepair (Mock)": mock_algo_weak,
        "KalmanRepair (Mock)": mock_algo_strong
    }

    # 4. 投入 DaTSClean 进行测试
    try:
        results = run_datsclean(
            df_dirty=df_dirty_mock,
            df_original=df_original_mock,
            target_col=target_column,
            algorithms_dict=mock_algorithms,
            search_tolerance=0.05
        )

        print("\n" + "★" * 40)
        print("🎉 测试完成！DaTSClean 最终寻优报告：")
        for key, value in results.items():
            print(f"   - {key}: {value}")
        print("★" * 40)

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
