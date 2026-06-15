import os
import time
import numpy as np
import pandas as pd
import warnings
from typing import List, Tuple, Set, Union

warnings.filterwarnings("ignore")


# ==========================================
# 🛠️ 兼容性桥接类 (Bridge Entities)
# ==========================================
class TimePoint:
    """模拟原工程中的单点数据实体，解耦外部工具库依赖"""

    def __init__(self, timestamp, value, truth=None, noise=None):
        self.timestamp = timestamp
        self.value = value
        self.truth = truth
        self.noise = noise
        self.label = True


class TimeSeries:
    """模拟原工程中的时序序列实体"""

    def __init__(self):
        self.__series = []

    def add_point(self, p: TimePoint):
        self.__series.append(p)

    def get_timeseries(self) -> List[TimePoint]:
        return self.__series


# ==========================================
# 🧬 SHoTClean_B 核心动态规划清洗算法逻辑
# ==========================================
class SHoTClean_B:
    def __init__(
            self,
            timeseries: TimeSeries,
            s_max: Union[float, List[float], np.ndarray],
            s_min: Union[float, List[float], np.ndarray],
            alpha: float = 0.01,
            is_soft: bool = True
    ):
        self.timeseries = timeseries
        self.size = len(timeseries.get_timeseries())

        self.SMAX = np.array(s_max, dtype=float) if not np.isscalar(s_max) else float(s_max)
        self.SMIN = np.array(s_min, dtype=float) if not np.isscalar(s_min) else float(s_min)

        self.is_soft = is_soft
        self.alpha = alpha

        series = self.timeseries.get_timeseries()
        self.time_array = np.array([p.timestamp for p in series], dtype=float)

        raw_values = np.array([p.value for p in series], dtype=float)
        if raw_values.ndim == 1:
            self.value_array = raw_values.reshape(-1, 1)
        else:
            self.value_array = raw_values
        self.N, self.D = self.value_array.shape

        self._init_prior_distribution()

    def _init_prior_distribution(self):
        self.prior_mu = np.mean(self.value_array, axis=0)
        self.prior_std = np.std(self.value_array, axis=0)
        self.prior_std[self.prior_std < 1e-6] = 1e-6

    def _temporal_decay(self, gap: int) -> float:
        return np.exp(-0.1 * gap) if self.is_soft else 1.0

    def _compute_scores(self) -> np.ndarray:
        if self.is_soft:
            deviation = (self.value_array - self.prior_mu) / self.prior_std
            deviation_norm = np.linalg.norm(deviation, axis=1)
            scores = np.exp(-self.alpha * deviation_norm)
            return scores
        else:
            return np.ones(self.N, dtype=float)

    def clean(self) -> TimeSeries:
        outlier_indices, scores = self.outlier_detection()
        clean_series = self.outlier_repair(outlier_indices, scores)
        return clean_series

    def outlier_detection(self) -> Tuple[List[int], np.ndarray]:
        scores = self._compute_scores()
        dp = np.zeros(self.N, dtype=float)
        path = np.zeros(self.N, dtype=np.int32)
        max_score = -np.inf
        end_idx = 0

        predecessors = self._precompute_predecessors()

        for i in range(self.N):
            dp[i] = scores[i]
            path[i] = i
            for j in predecessors[i]:
                candidate = dp[j] + scores[i] * self._temporal_decay(i - j)
                if candidate > dp[i]:
                    dp[i] = candidate
                    path[i] = j
            if dp[i] > max_score:
                max_score = dp[i]
                end_idx = i

        normal_indices = self._backtrack_path(path, end_idx)
        outlier_indices = [idx for idx in range(self.N) if idx not in normal_indices]
        return outlier_indices, scores

    def _precompute_predecessors(self) -> List[List[int]]:
        predecessors: List[List[int]] = [[] for _ in range(self.N)]
        for i in range(1, self.N):
            valid_count = 0
            for j in range(i - 1, -1, -1):
                if self._is_valid_predecessor(j, i):
                    predecessors[i].append(j)
                    valid_count += 1
                    if valid_count >= 5:
                        break
        return predecessors

    def _is_valid_predecessor(self, j: int, i: int) -> bool:
        delta_t = self.time_array[i] - self.time_array[j]
        if delta_t <= 0:
            return False

        delta_v = self.value_array[i] - self.value_array[j]
        speed_vec = delta_v / delta_t

        if np.isscalar(self.SMIN) and np.isscalar(self.SMAX):
            speed_norm = np.linalg.norm(speed_vec)
            return (self.SMIN <= speed_norm <= self.SMAX)
        else:
            return np.all(speed_vec >= self.SMIN) and np.all(speed_vec <= self.SMAX)

    def _backtrack_path(self, path: np.ndarray, end_idx: int) -> Set[int]:
        normal_indices: Set[int] = set()
        cur = end_idx
        while True:
            normal_indices.add(cur)
            if path[cur] == cur:
                break
            cur = path[cur]
        return normal_indices

    def outlier_repair(self, outlier_indices: List[int], scores: np.ndarray) -> TimeSeries:
        label = np.ones(self.N, dtype=bool)
        label[outlier_indices] = False
        repaired_values = self.value_array.copy()

        for i in outlier_indices:
            prev_idx = self._find_nearest_normal(i, label, direction=-1)
            next_idx = self._find_nearest_normal(i, label, direction=1)

            if prev_idx is not None and next_idx is not None:
                t_prev = self.time_array[prev_idx]
                t_next = self.time_array[next_idx]
                v_prev = self.value_array[prev_idx]
                v_next = self.value_array[next_idx]
                ratio = (self.time_array[i] - t_prev) / (t_next - t_prev)
                repaired_values[i, :] = v_prev + ratio * (v_next - v_prev)
            elif prev_idx is not None:
                repaired_values[i, :] = self.value_array[prev_idx]
            elif next_idx is not None:
                repaired_values[i, :] = self.value_array[next_idx]
            else:
                repaired_values[i, :] = self.prior_mu

        clean_series = TimeSeries()
        for idx in range(self.N):
            new_tp = TimePoint(self.time_array[idx], repaired_values[idx, :])
            new_tp.label = bool(label[idx])
            clean_series.add_point(new_tp)
        return clean_series

    def _find_nearest_normal(self, idx: int, label: np.ndarray, direction: int) -> Union[int, None]:
        step = 1 if direction > 0 else -1
        cur = idx + step
        while 0 <= cur < self.N:
            if label[cur]:
                return cur
            cur += step
        return None


# ==========================================
# 🔌 ★ 核心改造：模块化标准接口定义 ★
# ==========================================
def clean_data(df_dirty: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """
    符合 ChooseClean 资产库规范的统一接口函数

    参数:
        df_dirty (pd.DataFrame): 传入的脏数据，包含索引及数值列
        ratio (float): 清洗比例控制。由于 SHoTClean 基于全局动态规划寻优，
                       本接口中通过 ratio 动态对修复强度和速度阈值进行软缩放，
                       或者按比例采样部分异常行予以最终还原。
    """
    # 1. 备份数据，防内存污染
    df_result = df_dirty.copy()
    if ratio <= 0:
        return df_result

    # 2. 自动补充或推断时间戳列
    # 如果数据中不包含明显的 'timestamp' 或 'time' 列，直接使用行索引作为时间参考轴
    time_col = None
    for col in ['timestamp', 'time', '时间']:
        if col in df_result.columns:
            time_col = col
            break

    if time_col:
        timestamps = df_result[time_col].values
        # 移非数值型的统计列参与时序计算
        feature_cols = [c for c in df_result.columns if c != time_col and pd.api.types.is_numeric_dtype(df_result[c])]
    else:
        timestamps = np.arange(len(df_result), dtype=float)
        feature_cols = [c for c in df_result.columns if pd.api.types.is_numeric_dtype(df_result[c])]

    # 3. 数据类型桥接：将 pd.DataFrame 转换为算法所需的内置 TimeSeries 结构
    ts_input = TimeSeries()
    matrix_values = df_result[feature_cols].fillna(df_result[feature_cols].mean()).values

    for idx in range(len(df_result)):
        tp = TimePoint(timestamp=timestamps[idx], value=matrix_values[idx, :])
        ts_input.add_point(tp)

    # 4. 智能适配物理速度约束范围 (SMAX / SMIN)
    # 根据你原代码中对不同数据集范围的启发式先验，这里计算数据的全局极差来进行自适应速度松弛
    data_range = np.ptp(matrix_values, axis=0)
    data_range[data_range == 0] = 1.0
    s_max = data_range * 0.5 / (ratio + 1e-5)  # 清洗比例 ratio 越小，放宽约束范围，反之收紧
    s_min = -s_max

    # 5. 实例化算法并运行动态规划清洗
    cleaner = SHoTClean_B(ts_input, s_max=s_max, s_min=s_min, is_soft=True)
    ts_output = cleaner.clean()

    # 6. 将 TimeSeries 修复数据还原回 Pandas DataFrame 结构
    repaired_points = ts_output.get_timeseries()
    repaired_matrix = np.array([p.value for p in repaired_points])

    # 7. 根据资源清洗比例 ratio，对行进行控制性还原 (消融/约束机制)
    # 如果算法识别出大量异常，但 ratio 资源受限，仅回填前 ratio 比例最严重的点，其余保持原脏数据状态
    outlier_mask = np.array([not p.label for p in repaired_points])
    outlier_indices = np.where(outlier_mask)[0]

    # 计算允许修复的上限行数
    max_clean_rows = int(len(df_result) * ratio)
    if len(outlier_indices) > max_clean_rows:
        # 超过预算，随机或者按顺序选择部分行不予以清洗恢复
        np.random.seed(42)
        discard_indices = np.random.choice(outlier_indices, size=(len(outlier_indices) - max_clean_rows), replace=False)
        for idx in discard_indices:
            repaired_matrix[idx, :] = matrix_values[idx, :]  # 还原回脏数据现状

    # 覆盖原数值矩阵
    df_result[feature_cols] = repaired_matrix

    return df_result