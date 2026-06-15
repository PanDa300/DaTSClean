import numpy as np


def find_best_ratio_brute_force(
        dirty_data_path,
        downstream_model,
        clean_algorithm_func,
        step=0.2,
        is_lower_better=True
):
    """
    暴力枚举法寻找最优清洗比例。

    参数:
        dirty_data_path (str): 脏数据文件路径。
        downstream_model: 下游评估模型 (拥有 .evaluate() 方法)。
        clean_algorithm_func: 指定的清洗函数 (如 BaseRepair 里的 clean_data)。
        step (float): 枚举步长，默认 0.05 (意味着测试 0.0, 0.05, 0.10 ... 1.0)。
        is_lower_better (bool): 评估指标是否越小越好 (如 MSE 为 True)。

    返回:
        best_ratio (float): 最优清洗比例。
        best_score (float): 对应的最佳得分。
        history (list): 搜索历史记录，格式为 [(ratio, score), ...]。
    """
    import pandas as pd
    df_dirty = pd.read_csv(dirty_data_path)

    # 初始化最优记录
    best_ratio = None
    best_score = float('inf') if is_lower_better else -float('inf')
    history = []

    # 生成测试网格
    test_ratios = np.arange(0.0, 1.0 + step, step)

    print("=" * 50)
    print(f"🚀 启动【暴力枚举】寻优 | 步长: {step} | 共需测试 {len(test_ratios)} 次")
    print("=" * 50)

    for ratio in test_ratios:
        # 限制浮点精度，避免出现 0.30000000000000004
        ratio = round(ratio, 4)

        # 1. 按照当前比例清洗数据
        df_cleaned = clean_algorithm_func(df_dirty.copy(), ratio)

        # 2. 评估下游性能
        score = downstream_model.evaluate(df_cleaned)
        history.append((ratio, score))

        print(f"   -> 测试比例 {ratio:.2f} | 得分: {score:.6f}")

        # 3. 更新最优解
        if is_lower_better:
            if score < best_score:
                best_score = score
                best_ratio = ratio
        else:
            if score > best_score:
                best_score = score
                best_ratio = ratio

    print(f"\n🏆 [暴力枚举] 完成！最优比例: {best_ratio:.2f} (得分: {best_score:.6f})\n")
    return best_ratio, best_score, history


def find_best_ratio_ternary_search(
        dirty_data_path,
        downstream_model,
        clean_algorithm_func,
        tolerance=0.01,
        is_lower_better=True
):
    """
    三分搜索法寻找最优清洗比例。

    参数:
        dirty_data_path (str): 脏数据文件路径。
        downstream_model: 下游评估模型。
        clean_algorithm_func: 指定的清洗函数。
        tolerance (float): 容忍阈值。当搜索区间长度小于此值时停止迭代。
        is_lower_better (bool): 评估指标是否越小越好。

    返回:
        best_ratio (float): 近似最优清洗比例。
        best_score (float): 对应的最佳得分。
        history (list): 搜索历史记录 (主要用于观察收敛路径)。
    """
    import pandas as pd
    df_dirty = pd.read_csv(dirty_data_path)

    # 初始化搜索区间 [L, R]
    left = 0.0
    right = 1.0
    history = []

    # 辅助比较函数：判断 A 是否“优于” B
    def is_better(score_A, score_B):
        if is_lower_better:
            return score_A < score_B
        else:
            return score_A > score_B

    print("=" * 50)
    print(f"🚀 启动【三分搜索】寻优 | 收敛阈值: {tolerance}")
    print("=" * 50)

    iteration = 0
    while (right - left) >= tolerance:
        iteration += 1

        # 1. 计算两个三等分点
        mid1 = left + (right - left) / 3.0
        mid2 = right - (right - left) / 3.0

        # 2. 评估 mid1
        df_cleaned_1 = clean_algorithm_func(df_dirty.copy(), mid1)
        score1 = downstream_model.evaluate(df_cleaned_1)
        history.append((mid1, score1))

        # 3. 评估 mid2
        df_cleaned_2 = clean_algorithm_func(df_dirty.copy(), mid2)
        score2 = downstream_model.evaluate(df_cleaned_2)
        history.append((mid2, score2))

        print(f"   [Iter {iteration}] 区间: [{left:.3f}, {right:.3f}]")
        print(f"      - 测 mid1 ({mid1:.3f}): {score1:.6f}")
        print(f"      - 测 mid2 ({mid2:.3f}): {score2:.6f}")

        # 4. 根据三分法规则更新区间
        if is_better(score1, score2):
            # 如果 mid1 比 mid2 好，说明最高点不可能在 mid2 右侧
            # 舍弃右边 1/3 区间
            right = mid2
            print(f"      -> mid1 胜出，砍掉右侧，新区间: [{left:.3f}, {right:.3f}]")
        else:
            # 如果 mid2 更好，或者两者一样，舍弃左边 1/3 区间
            left = mid1
            print(f"      -> mid2 胜出，砍掉左侧，新区间: [{left:.3f}, {right:.3f}]")

    # 当区间足够小时，取区间中点作为最终近似答案
    best_ratio = (left + right) / 2.0

    # 为了严谨，最后再测一次这个中点作为最终得分
    df_cleaned_final = clean_algorithm_func(df_dirty.copy(), best_ratio)
    best_score = downstream_model.evaluate(df_cleaned_final)

    print(f"\n🏆 [三分搜索] 完成 (迭代 {iteration} 次)！最优比例: {best_ratio:.3f} (得分: {best_score:.6f})\n")
    return best_ratio, best_score, history