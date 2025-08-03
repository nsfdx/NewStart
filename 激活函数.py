import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("激活函数详解：ReLU 及其他常用激活函数")
print("=" * 70)

# ============================
# 1. 什么是激活函数？
# ============================
print("\n1. 激活函数的作用")
print("-" * 50)

activation_intro = """
🧠 激活函数的核心作用：

1. **引入非线性**: 
   - 没有激活函数，多层神经网络等价于单层线性变换
   - 激活函数使网络能够学习复杂的非线性关系

2. **控制信号传递**:
   - 决定神经元是否被"激活"
   - 控制信息在网络中的流动

3. **数值稳定性**:
   - 将输出限制在合理范围内
   - 避免梯度爆炸或消失

在你的PyTorch代码中：
x = F.relu(self.fc1(x))  # 第一层后应用ReLU
x = F.relu(self.fc2(x))  # 第二层后应用ReLU
x = self.fc3(x)          # 输出层不用激活函数
"""
print(activation_intro)

# ============================
# 2. ReLU 激活函数详解
# ============================
print("\n2. ReLU (Rectified Linear Unit) 详解")
print("-" * 50)

def relu_function(x):
    """ReLU函数的实现"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU函数的导数"""
    return (x > 0).astype(float)

# ReLU的数学定义
relu_math = """
📐 ReLU 数学定义：

f(x) = max(0, x) = {
    x,  if x > 0
    0,  if x ≤ 0
}

导数：
f'(x) = {
    1,  if x > 0
    0,  if x ≤ 0
}

🔍 ReLU 的特点：

✅ 优点：
• 计算简单，速度快
• 缓解梯度消失问题
• 稀疏激活（部分神经元输出为0）
• 无饱和区域（x>0时）

❌ 缺点：
• Dead ReLU 问题（负值被完全抑制）
• 输出不以0为中心
• 在x<0时梯度为0，可能导致神经元"死亡"
"""
print(relu_math)

# ============================
# 3. 常用激活函数对比
# ============================
print("\n3. 常用激活函数对比")
print("-" * 50)

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # 防止溢出

def tanh(x):
    """Tanh激活函数"""
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU激活函数"""
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    """ELU激活函数"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def swish(x):
    """Swish激活函数"""
    return x * sigmoid(x)

def gelu(x):
    """GELU激活函数（近似）"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

# 可视化所有激活函数
def plot_activation_functions():
    """绘制所有激活函数的图像"""
    x = np.linspace(-5, 5, 1000)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 定义激活函数和标题
    functions = [
        (relu_function, "ReLU: f(x) = max(0, x)"),
        (sigmoid, "Sigmoid: f(x) = 1/(1+e^-x)"),
        (tanh, "Tanh: f(x) = tanh(x)"),
        (leaky_relu, "Leaky ReLU: f(x) = max(0.01x, x)"),
        (elu, "ELU: f(x) = x if x>0 else α(e^x-1)"),
        (swish, "Swish: f(x) = x·sigmoid(x)")
    ]
    
    for i, (func, title) in enumerate(functions):
        y = func(x)
        axes[i].plot(x, y, 'b-', linewidth=2)
        axes[i].set_title(title, fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[i].axvline(x=0, color='k', linestyle='-', alpha=0.3)
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('f(x)')
        
        # 特殊标记
        if i == 0:  # ReLU
            axes[i].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='拐点')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('/Users/xiang/Documents/GitHub/NewStart/NewStart/activation_functions.png', 
                dpi=300, bbox_inches='tight')
    print("激活函数对比图已保存为 activation_functions.png")

try:
    plot_activation_functions()
    plt.show()
except Exception as e:
    print(f"绘图出错: {e}")

# ============================
# 4. PyTorch中的激活函数实现
# ============================
print("\n4. PyTorch中的激活函数实现")
print("-" * 50)

def demonstrate_pytorch_activations():
    """演示PyTorch中的激活函数"""
    
    # 创建测试数据
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    print(f"输入数据: {x}")
    print()
    
    # 方法1: 使用F.函数
    print("方法1: 使用 torch.nn.functional")
    print(f"F.relu(x):        {F.relu(x)}")
    print(f"F.sigmoid(x):     {F.sigmoid(x)}")
    print(f"F.tanh(x):        {F.tanh(x)}")
    print(f"F.leaky_relu(x):  {F.leaky_relu(x, negative_slope=0.01)}")
    print(f"F.elu(x):         {F.elu(x)}")
    print(f"F.gelu(x):        {F.gelu(x)}")
    print()
    
    # 方法2: 使用nn.Module
    print("方法2: 使用 torch.nn.Module")
    activations = {
        'ReLU': nn.ReLU(),
        'Sigmoid': nn.Sigmoid(),
        'Tanh': nn.Tanh(),
        'LeakyReLU': nn.LeakyReLU(negative_slope=0.01),
        'ELU': nn.ELU(),
        'GELU': nn.GELU()
    }
    
    for name, activation in activations.items():
        result = activation(x)
        print(f"{name:10}: {result}")

demonstrate_pytorch_activations()

# ============================
# 5. 不同激活函数在实际网络中的表现
# ============================
print("\n5. 不同激活函数的性能对比")
print("-" * 50)

def test_activation_performance():
    """测试不同激活函数在简单网络中的性能"""
    
    # 使用你的数据生成函数
    def generate_data(n_samples=1000):
        np.random.seed(42)
        class_0 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], n_samples//2)
        class_1 = np.random.multivariate_normal([6, 6], [[1, -0.5], [-0.5, 1]], n_samples//2)
        X = np.vstack([class_0, class_1])
        y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
        return X, y
    
    # 定义不同激活函数的网络
    class TestClassifier(nn.Module):
        def __init__(self, activation='relu'):
            super(TestClassifier, self).__init__()
            self.fc1 = nn.Linear(2, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 2)
            self.dropout = nn.Dropout(0.2)
            
            # 选择激活函数
            if activation == 'relu':
                self.activation = nn.ReLU()
            elif activation == 'sigmoid':
                self.activation = nn.Sigmoid()
            elif activation == 'tanh':
                self.activation = nn.Tanh()
            elif activation == 'leaky_relu':
                self.activation = nn.LeakyReLU()
            elif activation == 'elu':
                self.activation = nn.ELU()
            elif activation == 'gelu':
                self.activation = nn.GELU()
        
        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.dropout(x)
            x = self.activation(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    # 准备数据
    X_train, y_train = generate_data(800)
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    # 测试不同激活函数
    activations = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'gelu']
    results = {}
    
    print("简单训练测试 (10个epoch):")
    print("激活函数       最终损失    收敛速度")
    print("-" * 40)
    
    for act in activations:
        model = TestClassifier(activation=act)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 简单训练
        model.train()
        losses = []
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # 计算收敛速度 (损失下降率)
        convergence_rate = (losses[0] - losses[-1]) / losses[0]
        results[act] = {'final_loss': losses[-1], 'convergence': convergence_rate}
        
        print(f"{act:12}   {losses[-1]:.4f}      {convergence_rate:.3f}")
    
    return results

performance_results = test_activation_performance()

# ============================
# 6. 激活函数选择指南
# ============================
print("\n6. 激活函数选择指南")
print("-" * 50)

selection_guide = """
🎯 激活函数选择指南：

📊 **ReLU (最常用)**
• 适用: 大多数深度网络
• 优势: 计算快，缓解梯度消失
• 场景: 卷积神经网络、全连接网络

📈 **Leaky ReLU**
• 适用: ReLU出现死神经元时
• 优势: 解决Dead ReLU问题
• 场景: 深层网络，GAN

📉 **ELU**
• 适用: 需要负值输出时
• 优势: 输出均值接近0
• 场景: 自编码器，深层网络

🔄 **Tanh**
• 适用: RNN，小型网络
• 优势: 输出范围(-1,1)，0中心
• 场景: 循环神经网络

📊 **Sigmoid**
• 适用: 二分类输出层
• 优势: 输出概率解释
• 场景: 输出层，门控机制

🚀 **GELU (现代)**
• 适用: Transformer，现代架构
• 优势: 平滑，性能优秀
• 场景: BERT、GPT等

🎲 **Swish**
• 适用: 移动端模型
• 优势: 自门控，平滑
• 场景: 轻量级网络
"""
print(selection_guide)

# ============================
# 7. 在你的代码中修改激活函数
# ============================
print("\n7. 在你的代码中修改激活函数")
print("-" * 50)

code_modification = """
在你的 SimpleClassifier 中替换激活函数：

原始代码:
```python
def forward(self, x):
    x = F.relu(self.fc1(x))      # ← 这里使用ReLU
    x = self.dropout(x)
    x = F.relu(self.fc2(x))      # ← 这里使用ReLU
    x = self.dropout(x)
    x = self.fc3(x)
    return x
```

修改为其他激活函数:
```python
# 使用Leaky ReLU
x = F.leaky_relu(self.fc1(x), negative_slope=0.01)

# 使用ELU
x = F.elu(self.fc1(x))

# 使用GELU
x = F.gelu(self.fc1(x))

# 使用Tanh
x = F.tanh(self.fc1(x))
```

或者在__init__中定义:
```python
def __init__(self, input_size, hidden_size, output_size):
    super(SimpleClassifier, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, output_size)
    self.dropout = nn.Dropout(0.2)
    self.activation = nn.GELU()  # ← 定义激活函数

def forward(self, x):
    x = self.activation(self.fc1(x))  # ← 使用定义的激活函数
    x = self.dropout(x)
    x = self.activation(self.fc2(x))
    x = self.dropout(x)
    x = self.fc3(x)
    return x
```
"""
print(code_modification)

# ============================
# 8. 梯度流动对比
# ============================
print("\n8. 梯度流动对比")
print("-" * 50)

def visualize_gradients():
    """可视化不同激活函数的梯度"""
    x = np.linspace(-3, 3, 1000)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    functions_and_derivatives = [
        (relu_function, relu_derivative, "ReLU"),
        (sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x)), "Sigmoid"),
        (tanh, lambda x: 1 - tanh(x)**2, "Tanh"),
        (leaky_relu, lambda x: np.where(x > 0, 1, 0.01), "Leaky ReLU"),
        (elu, lambda x: np.where(x > 0, 1, elu(x) + 1), "ELU"),
        (swish, lambda x: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)), "Swish")
    ]
    
    for i, (func, derivative, name) in enumerate(functions_and_derivatives):
        y = func(x)
        dy = derivative(x)
        
        # 函数图像
        ax = axes[i]
        ax.plot(x, y, 'b-', linewidth=2, label='函数')
        ax.plot(x, dy, 'r--', linewidth=2, label='导数')
        ax.set_title(f'{name}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlim(-3, 3)
        
        # 标记梯度消失区域
        if name in ['Sigmoid', 'Tanh']:
            ax.axvspan(-3, -2, alpha=0.2, color='red', label='梯度小')
            ax.axvspan(2, 3, alpha=0.2, color='red')
    
    plt.tight_layout()
    plt.savefig('/Users/xiang/Documents/GitHub/NewStart/NewStart/activation_gradients.png', 
                dpi=300, bbox_inches='tight')
    print("激活函数梯度对比图已保存为 activation_gradients.png")

try:
    visualize_gradients()
    plt.show()
except Exception as e:
    print(f"梯度可视化出错: {e}")

# ============================
# 9. 总结
# ============================
print("\n" + "="*70)
print("总结")
print("="*70)

summary = """
🎓 激活函数总结：

🔥 **ReLU的重要性**：
• 在你的代码中使用F.relu()是明智选择
• 计算效率高，训练速度快
• 有效缓解梯度消失问题
• 适合大多数深度学习任务

📈 **选择建议**：
1. **默认选择**: ReLU (99%情况下都没问题)
2. **遇到死神经元**: 试试Leaky ReLU或ELU
3. **现代架构**: 考虑GELU (Transformer等)
4. **输出层**: 
   - 二分类: Sigmoid
   - 多分类: 不用激活函数 (配合CrossEntropyLoss)
   - 回归: 不用激活函数或根据输出范围选择

🚀 **优化建议**：
• 从ReLU开始，有问题再换
• 不同层可以使用不同激活函数
• 注意激活函数对学习率的影响
• 结合BatchNorm使用效果更好

💡 **在你的PyTorch代码中**：
当前使用ReLU是最佳实践，如果想提升性能，可以尝试：
1. 将ReLU改为GELU
2. 添加BatchNorm层
3. 尝试不同的学习率
"""
print(summary)

if __name__ == "__main__":
    print("\n🎉 激活函数详解完成！")