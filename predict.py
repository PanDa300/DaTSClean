import numpy as np
import matplotlib.pyplot as plt  # 新增：引入 matplotlib 绘图库
from gplearn.genetic import SymbolicRegressor
from sklearn.utils.validation import check_X_y
import warnings

# 忽略警告以保持控制台整洁
warnings.filterwarnings('ignore')

# ==========================================
# 🛠️ 兼容性补丁区 (保持原样，确保在各种环境中都能运行)
# ==========================================
if not hasattr(SymbolicRegressor, '_validate_data'):
    SymbolicRegressor._validate_data = lambda self, X, y, **kwargs: check_X_y(X, y, **kwargs)

original_fit = SymbolicRegressor.fit


def patched_fit(self, X, y, sample_weight=None):
    self.n_features_in_ = X.shape[1] if len(X.shape) > 1 else 1
    return original_fit(self, X, y, sample_weight)


SymbolicRegressor.fit = patched_fit


# ==========================================
# 🧠 核心功能 1: 训练模型
# ==========================================
def train_symbolic_model(x_data, y_data, generations=20, population_size=2000, verbose=1):
    """
    输入 X 和 Y 数据，训练并返回一个符号回归模型。
    """
    # 1. 强制整理数据格式，确保兼容性
    x_array = np.array(x_data)
    y_array = np.array(y_data).ravel()  # 确保 Y 是一维的

    # gplearn 要求 X 是二维数组
    if x_array.ndim == 1:
        X_2d = x_array.reshape(-1, 1)
    else:
        X_2d = x_array

    # 2. 定义基础数学符号积木
    function_set = ['add', 'sub', 'mul', 'div', 'log', 'sqrt', 'abs']

    # 3. 初始化符号回归器
    model = SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        stopping_criteria=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.9,
        verbose=verbose,
        parsimony_coefficient=0.02,  # 惩罚复杂公式
        random_state=42,
        function_set=function_set
    )

    if verbose:
        print("🚀 开始使用遗传算法寻找通用公式...")

    # 4. 训练模型
    model.fit(X_2d, y_array)

    if verbose:
        print("\n🏆 训练完成！最佳公式为:")
        print(model._program)
        print("-" * 30)

    # 5. 返回训练好的模型对象
    return model


# ==========================================
# 🎯 核心功能 2: 使用模型进行预测
# ==========================================
def predict_y(model, x):
    """
    根据训练好的 SymbolicRegressor 模型预测对应的 y 值。
    """
    x_array = np.array(x)
    is_scalar = x_array.ndim == 0

    if is_scalar:
        X_2d = np.array([[x]])
    else:
        X_2d = x_array.reshape(-1, 1)

    y_pred = model.predict(X_2d)

    if is_scalar:
        return y_pred[0]
    else:
        return y_pred


# ==========================================
# 📈 核心功能 3: 绘制预测曲线 (新增部分)
# ==========================================
# ==========================================
# 📈 核心功能 3: 绘制预测曲线 (新增部分 - 已修改背景色)
# ==========================================
def plot_prediction_curve(model, x_train, y_train):
    """
    绘制原始数据点与模型预测曲线的对比图
    """
    # 1. 修改这里：添加 facecolor='#fdfcf5' 修改图片外围背景色

    # 尝试设置支持中文的字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except:
        pass
    plt.figure(figsize=(9, 7))

    # 2. 修改这里：获取当前坐标轴对象，并修改内部绘图区域的背景色
    ax = plt.gca()
    ax.set_facecolor('#fdfcf5')

    # (可选) 加上浅色的背景网格线，让图表看起来更整洁专业
    ax.grid(True, linestyle='--', color='#d3d3d3', zorder=0)

    # 3. 绘制原始的训练数据点（散点图）
    plt.scatter(x_train, y_train, color='blue', label='Actual Result', zorder=5)

    # 4. 生成密集的 X 值，用来画一条平滑的预测曲线
    x_min, x_max = min(x_train), max(x_train)
    # 加了一点点保护：确保如果公式里有 sqrt，X 预测范围不要掉到 0 以下导致报错
    x_start = max(0, x_min - 1)
    x_dense = np.linspace(x_start, x_max + 3, 200)

    # 获取密集 X 对应的预测 Y 值
    y_dense = predict_y(model, x_dense)

    # 5. 绘制模型预测的曲线（折线图）
    plt.plot(x_dense, y_dense, color='red', linewidth=4.5, label='Predicted Curve', zorder=4)

    # 6. 图表装饰
    plt.xlabel('清洗比例', fontsize=25)
    plt.ylabel('错误率', fontsize=25)
    plt.legend(fontsize=25)
    plt.tick_params(labelsize=20)

    # 7. 显示和保存图表
    print("📊 正在展示预测曲线图...")
    # 提示：如果您在代码里使用 savefig 保存图片，也需要显式传入 facecolor
    # plt.savefig('my_prediction.png', facecolor='#fdfcf5', dpi=150, bbox_inches='tight')
    plt.show()

# ==========================================
# 💡 测试与使用示例
# ==========================================
# ==========================================
# 💡 测试与使用示例
# ==========================================
if __name__ == "__main__":
    # 1. 准备训练数据
    print("生成模拟数据（已加入随机误差）...")
    X_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # 设置一个随机种子，确保每次运行代码产生的误差是一样的（方便调试）。
    # 如果你想每次运行都看到不同的误差，可以把下面这行注释掉：
    np.random.seed(42)

    # 假设真实的隐藏规律是 y = 2.5 * sqrt(x)
    # 我们加上高斯噪声: loc=0.0 (误差均值为0), scale=0.4 (标准差为0.4，控制误差的大小)
    # 你可以尝试修改 scale 的值，比如改成 1.0，误差就会大很多！
    Y_train = [2.5 * np.sqrt(x) + np.random.normal(loc=0.0, scale=0.4) for x in X_train]

    # 2. 调用核心函数：输入数据，拿到模型对象
    my_model = train_symbolic_model(X_train, Y_train, generations=20, verbose=1)

    # 3. 拿到模型后，使用 predict_y 进行预测
    test_x = 12.5
    predicted_y = predict_y(my_model, test_x)

    print(f"\n🔮 单点预测结果: 当 x = {test_x} 时，模型计算出的 y = {predicted_y:.4f}")

    # 4. 画出图表！
    plot_prediction_curve(my_model, X_train, Y_train)