import os
import importlib.util
import pandas as pd
import numpy as np


def load_cleaning_algorithms(folder_path="CleanMode"):
    """
    动态扫描并加载指定文件夹下的所有清洗算法。
    假设文件夹下的每个 .py 文件代表一个算法，且内部包含一个统一名称的函数，例如 `clean_data`。
    """
    algorithms = {}

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"❌ 找不到文件夹: {folder_path}，请确保路径正确！")

    for filename in os.listdir(folder_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]  # 去掉 .py 后缀
            file_path = os.path.join(folder_path, filename)

            # 动态导入模块
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # 假设你的每个清洗脚本里，执行清洗的统一函数名叫 clean_data
            if hasattr(module, 'clean_data'):
                algorithms[module_name] = module.clean_data
            else:
                print(f"⚠️ 警告: 模块 {module_name} 中没有找到 'clean_data' 函数，已跳过。")

    print(f"📂 成功加载了 {len(algorithms)} 个清洗算法: {list(algorithms.keys())}")
    return algorithms


def select_best_algorithm(dirty_data_path, clean_ratio_b, downstream_model, is_lower_better=True):
    """
    凸包单点查询：在固定清洗比例下，遍历所有算法并返回最优的清洗算法名称。

    参数:
        dirty_data_path: 已经注入了特定错误率(a)的脏数据文件路径
        clean_ratio_b: 固定的清洗比例
        downstream_model: 下游的预测模型对象 (需要有一个评估方法)
        is_lower_better: 评估指标是否越小越好 (如 MSE/RMSE 选 True，F1/Accuracy 选 False)

    返回:
        best_algo_name (str): 最优清洗算法的名称
    """
    # 1. 动态加载所有清洗算法
    algorithms = load_cleaning_algorithms("CleanMode")

    if not algorithms:
        raise ValueError("没有可用的清洗算法，退出选择。")

    # 2. 读取脏数据
    df_dirty = pd.read_csv(dirty_data_path)

    # 3. 初始化擂台参数
    best_algo_name = None
    # 如果指标越小越好（如误差），初始最好成绩设为无穷大；反之设为负无穷大
    best_score = float('inf') if is_lower_better else -float('inf')

    print("\n" + "=" * 50)
    print(f"🔍 开始算法寻优 | 目标清洗比例: {clean_ratio_b}")
    print("=" * 50)

    # 4. 遍历每个算法进行效果测试
    for algo_name, clean_func in algorithms.items():
        try:
            print(f"🔄 正在评估算法: [{algo_name}] ...")

            # 步骤 A：调用当前算法清洗数据 (这里假设你的清洗函数接收原始数据和比例)
            # 注意：实际中你的 clean_data 可能需要接收 DataFrame 或者文件路径，这里以 DataFrame 为例
            df_cleaned = clean_func(df_dirty.copy(), clean_ratio_b)

            # 步骤 B：使用下游任务评估清洗后的数据质量
            # 假设 downstream_model 有一个 evaluate 方法，输入清洗后数据，输出得分 (比如 MSE)
            score = downstream_model.evaluate(df_cleaned)
            print(f"   -> 下游任务评估得分: {score:.4f}")

            # 步骤 C：更新凸包边界 (单点找极值)
            if is_lower_better:
                if score < best_score:
                    best_score = score
                    best_algo_name = algo_name
            else:
                if score > best_score:
                    best_score = score
                    best_algo_name = algo_name

        except Exception as e:
            print(f"   ❌ 算法 [{algo_name}] 运行失败: {e}")

    print("\n" + "★" * 50)
    print(f"🏆 寻优完成！最佳清洗算法是: 【{best_algo_name}】 (得分: {best_score:.4f})")
    print("★" * 50 + "\n")

    return best_algo_name


# ==========================================
# 💡 接口使用示例
# ==========================================
if __name__ == "__main__":
    # --- 模拟下游模型对象 ---
    class MockDownstreamModel:
        def evaluate(self, data):
            # 这是一个假模型，为了测试代码能跑通，它会随机返回一个 MSE 误差
            return np.random.uniform(0.01, 0.10)


    my_model = MockDownstreamModel()

    # --- 调用核心函数 ---
    # 假设你有一份注入了 0.3 错误率的数据 "data_0.3.csv"
    # 我们想找出在清洗比例为 0.5 的情况下，哪个算法能让下游模型的误差（MSE）最小
    best_algorithm = select_best_algorithm(
        dirty_data_path="WineQT_0.2.csv",
        clean_ratio_b=0.5,
        downstream_model=my_model,
        is_lower_better=True  # 因为我们看的是误差，所以越小越好
    )

    # 拿到名字后，你可以根据需要进行后续的部署或记录
    # print(f"最终我要使用的算法名是：{best_algorithm}")