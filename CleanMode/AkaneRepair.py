import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")


# ==========================================
# 🧬 Akane 核心算法框架 (多维增强版)
# ==========================================
class AkaneTimeSeriesCleaner:
    def __init__(self, window_size=3, budget=10.0, cost_per_repair=1.0):
        """
        Akane 核心框架
        :param window_size: 模式挖掘的窗口大小
        :param budget: 允许修改的总代价上限（允许修复的最大点数）
        :param cost_per_repair: 修复一个点消耗的代价
        """
        self.window_size = window_size
        self.budget = budget
        self.cost_per_repair = cost_per_repair
        self.history_variance = 1.0  # 用于计算概率密度的方差基准

    def _fit_local_pattern(self, window):
        """用窗口前 n-1 个点预测第 n 个点的期望值"""
        if len(window) < self.window_size:
            return window[-1] if len(window) > 0 else 0
        return np.mean(window[:-1])

    def calculate_perplexity(self, sequence):
        """计算单列序列的困惑度 (Perplexity)"""
        n = len(sequence)
        if n <= self.window_size:
            return 1.0

        log_likelihood_sum = 0
        count = 0

        for i in range(self.window_size, n):
            window = sequence[i - self.window_size: i + 1]
            expected_val = self._fit_local_pattern(window)
            target = window[-1]

            # 将预测误差转化为高斯概率密度
            prob = norm.pdf(target, loc=expected_val, scale=self.history_variance)
            prob = max(prob, 1e-6)  # 防止数值下溢

            log_likelihood_sum += np.log(prob)
            count += 1

        avg_negative_log_likelihood = -log_likelihood_sum / count
        return np.exp(avg_negative_log_likelihood)

    def _get_candidate_repairs(self, sequence, index):
        """寻找推荐的修复候选值"""
        start = max(0, index - self.window_size)
        end = min(len(sequence), index + self.window_size + 1)
        neighbor_mean = np.mean(sequence[start:end])

        if index >= self.window_size:
            pred_val = self._fit_local_pattern(sequence[index - self.window_size: index + 1])
        else:
            pred_val = neighbor_mean
        return [neighbor_mean, pred_val]

    def fit_import_and_clean(self, dirty_sequence):
        clean_sequence = np.array(dirty_sequence, dtype=float)
        current_budget = self.budget
        self.history_variance = max(np.std(dirty_sequence) * 0.5, 0.1)

        # 💡 【核心剪枝】：如果数据量太大，调大步长，不要挨个点挨个点去算
        N = len(clean_sequence)
        search_step = max(1, N // 100)  # 强行限制每轮最多扫描 100 个候选点

        while current_budget >= self.cost_per_repair:
            best_gain = 0
            best_idx = -1
            best_val = None

            current_pp = self.calculate_perplexity(clean_sequence)

            # 💡 修改这里：加入 search_step
            for i in range(0, N, search_step):
                original_val = clean_sequence[i]
                candidates = self._get_candidate_repairs(clean_sequence, i)

                for cand in candidates:
                    if cand == original_val or np.isnan(cand):
                        continue

                    clean_sequence[i] = cand
                    new_pp = self.calculate_perplexity(clean_sequence)

                    gain = current_pp - new_pp
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                        best_val = cand

                    clean_sequence[i] = original_val

            if best_gain < 1e-4 or best_idx == -1:
                break

            clean_sequence[best_idx] = best_val
            current_budget -= self.cost_per_repair

            # 💡 【核心防卡死】：防止 budget 太大导致死循环，最多允许贪心修 20 次
            if (self.budget - current_budget) >= 20:
                break

        return clean_sequence


# ==========================================
# 🔌 ★ 统一模块化标准接口定义 ★
# ==========================================
def clean_data(df_dirty: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """
    符合 ChooseClean 资产库规范的统一接口函数

    参数:
        df_dirty (pd.DataFrame): 传入的脏数据，包含特征列和标签列
        ratio (float): 清洗比例控制 (0.0 ~ 1.0)，在 Akane 中等价转换为点级修复 Budget 预算开销上限
    """
    df_result = df_dirty.copy()
    if ratio <= 0:
        return df_result

    # 1. 自动提取所有需要清洗的数值型特征列
    feature_cols = [c for c in df_result.columns if pd.api.types.is_numeric_dtype(df_result[c])]

    # 2. 计算本序列在该清洗比例下分配的总 Budget 预算限制 (允许修复的最大点数)
    sequence_length = len(df_result)
    calculated_budget = max(int(sequence_length * ratio), 1)

    # 3. 循环遍历各列，对每一列特征分别应用 Akane 困惑度约束清洗
    for col in feature_cols:
        # 将当前列转化为一维数组并初步填补空值防止计算中断
        raw_series = df_result[col].fillna(df_result[col].mean()).values

        # 实例化 Akane 清洗核心类
        cleaner = AkaneTimeSeriesCleaner(
            window_size=3,
            budget=calculated_budget,
            cost_per_repair=1.0
        )

        # 运行优化寻优修复
        cleaned_series = cleaner.fit_import_and_clean(raw_series)

        # 写回清洗结果
        df_result[col] = cleaned_series

    return df_result