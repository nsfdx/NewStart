import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

# ============================
# 解决中文字体显示问题
# ============================
def setup_chinese_font():
    """设置matplotlib支持中文显示"""
    try:
        # 方法1: 尝试设置常见的中文字体
        chinese_fonts = [
            'PingFang SC',  # macOS系统字体
            'Hiragino Sans GB',  # macOS中文字体
            'STSong',  # 华文宋体
            'SimHei',  # 黑体
            'Microsoft YaHei',  # 微软雅黑
            'Arial Unicode MS'  # Arial Unicode
        ]
        
        available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
        
        for font in chinese_fonts:
            if font in available_fonts:
                rcParams['font.sans-serif'] = [font]
                rcParams['axes.unicode_minus'] = False
                print(f"✅ 使用字体: {font}")
                return True
        
        # 方法2: 如果没有中文字体，使用英文
        print("⚠️  未找到中文字体，使用英文标签")
        rcParams['font.sans-serif'] = ['DejaVu Sans']
        return False
        
    except Exception as e:
        print(f"字体设置出错: {e}")
        return False

# 设置字体
use_chinese = setup_chinese_font()

print("=" * 60)
print("NumPy Array Operations: vstack and hstack" if not use_chinese else "数据合并操作详解：np.vstack 和 np.hstack")
print("=" * 60)

# ============================
# 1. 重现你代码中的数据生成过程
# ============================
print("\n1. Data Generation Process" if not use_chinese else "\n1. 数据生成过程演示")
print("-" * 40)

# 设置随机种子以获得可重现的结果
np.random.seed(42)
n_samples = 1000

# 生成两个类别的数据
class_0 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], n_samples//2)
class_1 = np.random.multivariate_normal([6, 6], [[1, -0.5], [-0.5, 1]], n_samples//2)

print(f"class_0 shape: {class_0.shape}")  # (500, 2)
print(f"class_1 shape: {class_1.shape}")  # (500, 2)
print(f"n_samples//2 = {n_samples//2}")   # 500

print(f"\nclass_0 first 5 samples:" if not use_chinese else f"\nclass_0 前5个样本:")
print(class_0[:5])
print(f"\nclass_1 first 5 samples:" if not use_chinese else f"\nclass_1 前5个样本:")
print(class_1[:5])

# ============================
# 2. np.vstack 详解 - 垂直堆叠特征数据
# ============================
print("\n2. np.vstack - Vertical Stacking" if not use_chinese else "\n2. np.vstack - 垂直堆叠特征数据")
print("-" * 40)

info_text = """
np.vstack stacks arrays vertically (row-wise):
class_0: (500, 2) array
class_1: (500, 2) array
↓ vertical stack
Result: (1000, 2) array
""" if not use_chinese else """
np.vstack 将两个数组在垂直方向（行方向）堆叠:
class_0: (500, 2) 的数组
class_1: (500, 2) 的数组
↓ 垂直堆叠
结果: (1000, 2) 的数组
"""

print(info_text)

X = np.vstack([class_0, class_1])
print(f"\nX shape: {X.shape}")  # (1000, 2)

structure_text = f"""
X structure:
First 500 rows (index 0-499): from class_0
Last 500 rows (index 500-999): from class_1
""" if not use_chinese else f"""
X 的结构:
前500行 (索引0-499): 来自 class_0
后500行 (索引500-999): 来自 class_1
"""

print(structure_text)

print(f"\nVerification - X first 5 rows (from class_0):" if not use_chinese else f"\n验证 - X 的前5行 (来自class_0):")
print(X[:5])
print(f"\nVerification - X rows 500-504 (from class_1):" if not use_chinese else f"\n验证 - X 的第500-504行 (来自class_1):")
print(X[500:505])

# ============================
# 3. np.hstack 详解 - 水平堆叠标签数据
# ============================
print("\n3. np.hstack - Horizontal Stacking" if not use_chinese else "\n3. np.hstack - 水平堆叠标签数据")
print("-" * 40)

print("Create labels for each class:" if not use_chinese else "为每个类别创建标签:")
labels_class_0 = np.zeros(n_samples//2)  # 500个0
labels_class_1 = np.ones(n_samples//2)   # 500个1

print(f"labels_class_0 shape: {labels_class_0.shape}")  # (500,)
print(f"labels_class_1 shape: {labels_class_1.shape}")  # (500,)

print(f"\nlabels_class_0 first 10: {labels_class_0[:10]}" if not use_chinese else f"\nlabels_class_0 前10个: {labels_class_0[:10]}")
print(f"labels_class_1 first 10: {labels_class_1[:10]}" if not use_chinese else f"labels_class_1 前10个: {labels_class_1[:10]}")

hstack_info = """
np.hstack concatenates 1D arrays horizontally:
labels_class_0: (500,) all zeros
labels_class_1: (500,) all ones
→ horizontal concatenation
Result: (1000,) first 500 are 0s, last 500 are 1s
""" if not use_chinese else """
np.hstack 将两个一维数组在水平方向（列方向）连接:
labels_class_0: (500,) 全是0
labels_class_1: (500,) 全是1
→ 水平连接
结果: (1000,) 前500个是0，后500个是1
"""

print(hstack_info)

y = np.hstack([labels_class_0, labels_class_1])
print(f"\ny shape: {y.shape}")  # (1000,)
print(f"y first 10 labels: {y[:10]}" if not use_chinese else f"y 的前10个标签: {y[:10]}")     # 全是0
print(f"y labels 500-509: {y[500:510]}" if not use_chinese else f"y 的第500-509个标签: {y[500:510]}")  # 全是1
print(f"y last 10 labels: {y[-10:]}" if not use_chinese else f"y 的最后10个标签: {y[-10:]}")   # 全是1

# ============================
# 4. 可视化数据结构 (修复字体问题)
# ============================
print("\n4. Data Visualization" if not use_chinese else "\n4. 数据结构可视化")
print("-" * 40)

def visualize_data_combination():
    """可视化数据合并过程"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 根据是否支持中文选择标题
    if use_chinese:
        titles = [
            'Class 0 数据\n(红色, 标签=0)',
            'Class 1 数据\n(蓝色, 标签=1)', 
            '合并后的数据\n(X = vstack([class_0, class_1]))',
            '标签分布\n(y = hstack([zeros, ones]))'
        ]
        xlabel = '特征1'
        ylabel = '特征2'
        legend_labels = ['前500个(class_0)', '后500个(class_1)']
    else:
        titles = [
            'Class 0 Data\n(Red, Label=0)',
            'Class 1 Data\n(Blue, Label=1)',
            'Combined Data\n(X = vstack([class_0, class_1]))',
            'Label Distribution\n(y = hstack([zeros, ones]))'
        ]
        xlabel = 'Feature 1'
        ylabel = 'Feature 2'
        legend_labels = ['First 500 (class_0)', 'Last 500 (class_1)']
    
    # 子图1: class_0 原始数据
    axes[0, 0].scatter(class_0[:, 0], class_0[:, 1], c='red', alpha=0.6, s=20)
    axes[0, 0].set_title(titles[0], fontsize=12)
    axes[0, 0].set_xlabel(xlabel)
    axes[0, 0].set_ylabel(ylabel)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: class_1 原始数据
    axes[0, 1].scatter(class_1[:, 0], class_1[:, 1], c='blue', alpha=0.6, s=20)
    axes[0, 1].set_title(titles[1], fontsize=12)
    axes[0, 1].set_xlabel(xlabel)
    axes[0, 1].set_ylabel(ylabel)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: 合并后的数据
    colors = ['red' if label == 0 else 'blue' for label in y]
    axes[1, 0].scatter(X[:, 0], X[:, 1], c=colors, alpha=0.6, s=20)
    axes[1, 0].set_title(titles[2], fontsize=12)
    axes[1, 0].set_xlabel(xlabel)
    axes[1, 0].set_ylabel(ylabel)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图4: 标签分布
    axes[1, 1].hist([y[:500], y[500:]], bins=2, alpha=0.7, 
                   color=['red', 'blue'], label=legend_labels)
    axes[1, 1].set_title(titles[3], fontsize=12)
    axes[1, 1].set_xlabel('Label Value' if not use_chinese else '标签值')
    axes[1, 1].set_ylabel('Count' if not use_chinese else '数量')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    filename = '/Users/xiang/Documents/GitHub/NewStart/NewStart/data_combination_fixed.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    success_msg = f"Data visualization saved as data_combination_fixed.png" if not use_chinese else f"数据合并可视化已保存为 data_combination_fixed.png"
    print(success_msg)

try:
    visualize_data_combination()
except Exception as e:
    print(f"Visualization error: {e}" if not use_chinese else f"可视化过程中出现错误: {e}")

# ============================
# 5. 验证数据对应关系
# ============================
print("\n5. Data Index Correspondence" if not use_chinese else "\n5. 数据顺序和索引对应关系")
print("-" * 40)

correspondence_table = """
Final dataset index correspondence:
┌─────────────┬──────────────┬─────────────┐
│    Index    │ Data Source  │    Label    │
├─────────────┼──────────────┼─────────────┤
│   0-499     │   class_0    │      0      │
│  500-999    │   class_1    │      1      │
└─────────────┴──────────────┴─────────────┘
""" if not use_chinese else """
最终数据集的索引对应关系:
┌─────────────┬──────────────┬─────────────┐
│    索引     │   数据来源    │    标签     │
├─────────────┼──────────────┼─────────────┤
│   0-499     │   class_0    │      0      │
│  500-999    │   class_1    │      1      │
└─────────────┴──────────────┴─────────────┘
"""

print(correspondence_table)

print(f"\nVerification of index correspondence:" if not use_chinese else f"\n验证索引对应关系:")
print(f"X[0] from class_0[0]: {np.array_equal(X[0], class_0[0])}" if not use_chinese else f"X[0] 来自 class_0[0]: {np.array_equal(X[0], class_0[0])}")
print(f"X[499] from class_0[499]: {np.array_equal(X[499], class_0[499])}" if not use_chinese else f"X[499] 来自 class_0[499]: {np.array_equal(X[499], class_0[499])}")
print(f"X[500] from class_1[0]: {np.array_equal(X[500], class_1[0])}" if not use_chinese else f"X[500] 来自 class_1[0]: {np.array_equal(X[500], class_1[0])}")
print(f"X[999] from class_1[499]: {np.array_equal(X[999], class_1[499])}" if not use_chinese else f"X[999] 来自 class_1[499]: {np.array_equal(X[999], class_1[499])}")

print(f"\ny[0] = {y[0]} (corresponds to class_0)" if not use_chinese else f"\ny[0] = {y[0]} (对应 class_0)")
print(f"y[499] = {y[499]} (corresponds to class_0)" if not use_chinese else f"y[499] = {y[499]} (对应 class_0)")
print(f"y[500] = {y[500]} (corresponds to class_1)" if not use_chinese else f"y[500] = {y[500]} (对应 class_1)")
print(f"y[999] = {y[999]} (corresponds to class_1)" if not use_chinese else f"y[999] = {y[999]} (对应 class_1)")

# ============================
# 6. 总结
# ============================
print("\n" + "="*60)
print("Summary" if not use_chinese else "总结")
print("="*60)

summary_text = """
📊 Array Combination Operations Summary:

🔸 np.vstack([class_0, class_1]):
  • Function: Vertically stack two arrays
  • Input: class_0 (500×2), class_1 (500×2)
  • Output: X (1000×2)
  • Result: First 500 rows are class_0, last 500 rows are class_1

🔸 np.hstack([np.zeros(500), np.ones(500)]):
  • Function: Horizontally concatenate two 1D arrays
  • Input: zeros(500), ones(500)
  • Output: y (1000,)
  • Result: First 500 are 0s, last 500 are 1s

🎯 Final Dataset:
  • X: 1000 samples, each with 2 features
  • y: 1000 labels, first 500 are 0s, last 500 are 1s
  • Perfect binary classification dataset!

💡 This data organization ensures:
  ✅ Features and labels correspond one-to-one
  ✅ Class balance (500 samples per class)
  ✅ Suitable for supervised learning algorithms
""" if not use_chinese else """
📊 数据合并操作总结:

🔸 np.vstack([class_0, class_1]):
  • 功能: 垂直堆叠两个数组
  • 输入: class_0 (500×2), class_1 (500×2)
  • 输出: X (1000×2)
  • 结果: 前500行是class_0，后500行是class_1

🔸 np.hstack([np.zeros(500), np.ones(500)]):
  • 功能: 水平连接两个一维数组
  • 输入: zeros(500), ones(500)
  • 输出: y (1000,)
  • 结果: 前500个是0，后500个是1

🎯 最终数据集:
  • X: 1000个样本，每个样本2个特征
  • y: 1000个标签，前500个是0，后500个是1
  • 完美的二元分类数据集！

💡 这种数据组织方式确保了:
  ✅ 特征和标签一一对应
  ✅ 类别平衡（每类500个样本）
  ✅ 适合监督学习算法训练
"""

print(summary_text)

if __name__ == "__main__":
    final_msg = "\n🎉 Array combination operations explanation completed!" if not use_chinese else "\n🎉 数据合并操作解释完成！"
    print(final_msg)