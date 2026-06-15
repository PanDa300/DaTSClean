import numpy as np
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn.utils.validation import check_X_y
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 🛠️ 终极兼容性补丁区 (修复 gplearn 与新版 sklearn 的冲突)
# ==========================================

# 补丁 1: 修复 _validate_data 缺失错误
if not hasattr(SymbolicRegressor, '_validate_data'):
    SymbolicRegressor._validate_data = lambda self, X, y, **kwargs: check_X_y(X, y, **kwargs)

# 补丁 2: 修复 n_features_in_ 缺失错误
# 保存原始的 fit 方法
original_fit = SymbolicRegressor.fit

# 定义一个新的 fit 方法，在训练前强行加上 n_features_in_ 属性
def patched_fit(self, X, y, sample_weight=None):
    # 强制加上特征数量属性 (二维数组的列数)
    self.n_features_in_ = X.shape[1] if len(X.shape) > 1 else 1
    # 调用原本的 fit 逻辑
    return original_fit(self, X, y, sample_weight)

# 将原来的 fit 替换为我们修改过的版本
SymbolicRegressor.fit = patched_fit

# ==========================================
# 1. 生成模拟数据 (X与Y)
# ==========================================
np.random.seed(53)
X_data = np.linspace(1, 10, 100).reshape(-1, 1)
y_data = 2 * np.sqrt(X_data.ravel()) + 1.5 * np.log(X_data.ravel()) + np.random.normal(0, 0.2, 100)

# ==========================================
# 2. 定义并运行遗传算法模型
# ==========================================
function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs']

est_gp = SymbolicRegressor(
    population_size=2000,
    generations=20,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    parsimony_coefficient=0.02,
    random_state=42,
    function_set=function_set
)

print("🚀 开始遗传算法进化过程...")
est_gp.fit(X_data, y_data)

# 打印结果
print("\n" + "="*50)
print("🏆 进化出的最佳公式:")
print(est_gp._program)
print("="*50 + "\n")

# 用进化出来的模型进行预测
y_pred = est_gp.predict(X_data)

# 计算 R^2 评估拟合优度
ss_res = np.sum((y_data - y_pred) ** 2)
ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f"📈 最终模型的决定系数 (R²): {r2:.4f}")

y_data_new=y_data
y_pred_new=y_pred

for i in range(len(y_data_new)):
    y_data_new[i]=11-y_data_new[i]
for i in range(len(y_pred_new)):
    y_pred_new[i]=11-y_pred_new[i]
# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(X_data, y_data_new, color='gray', label='Original Data (原始数据)', alpha=0.5)
plt.plot(X_data, y_pred_new, color='red', linewidth=2.5, label='GP Evolved Model (遗传进化模型)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()