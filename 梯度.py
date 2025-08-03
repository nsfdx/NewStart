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
print("梯度在神经网络中的作用详解")
print("=" * 70)

# ============================
# 1. 梯度的本质含义
# ============================
print("\n1. 梯度的本质含义")
print("-" * 50)

gradient_concept = """
🎯 梯度的本质：

📐 **数学定义**：
• 梯度是函数在某点处的变化率（导数）
• 对于多变量函数，梯度是偏导数组成的向量
• 梯度指向函数值增长最快的方向

🧠 **在神经网络中的含义**：
• 梯度告诉我们损失函数对每个参数的敏感度
• 正梯度：增加参数会增加损失
• 负梯度：增加参数会减少损失
• 梯度大小：表示敏感程度

🎛️ **直观理解**：
想象你在山上迷雾中寻找最低点：
• 梯度 = 当前位置的坡度方向
• 负梯度方向 = 下山最快的方向
• 梯度大小 = 坡度陡峭程度
"""
print(gradient_concept)

# ============================
# 2. 梯度在训练循环中的具体作用
# ============================
print("\n2. 梯度在训练循环中的具体作用")
print("-" * 50)

def demonstrate_gradient_flow():
    """演示梯度在训练中的流动"""
    
    # 创建一个简单的线性模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1, bias=False)
            # 初始化权重为一个已知值
            self.linear.weight.data = torch.tensor([[2.0]])
        
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    
    # 准备简单数据：y = 3x (我们想让模型学到权重为3)
    x = torch.tensor([[1.0], [2.0], [3.0]])
    y_true = torch.tensor([[3.0], [6.0], [9.0]])
    
    criterion = nn.MSELoss()
    
    print("📊 梯度计算演示：")
    print(f"目标关系: y = 3x")
    print(f"模型初始权重: {model.linear.weight.item():.3f}")
    print(f"训练数据: x={x.flatten()}, y_true={y_true.flatten()}")
    print()
    
    # 前向传播
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    
    print(f"初始预测: {y_pred.flatten()}")
    print(f"初始损失: {loss.item():.3f}")
    
    # 反向传播计算梯度
    loss.backward()
    
    gradient = model.linear.weight.grad.item()
    print(f"计算得到的梯度: {gradient:.3f}")
    print()
    
    # 解释梯度的含义
    print("🔍 梯度含义解析：")
    print(f"• 梯度 = {gradient:.3f} < 0")
    print("• 负梯度意味着：减少权重会增加损失，增加权重会减少损失")
    print("• 所以我们应该向梯度的反方向更新权重")
    print(f"• 当前权重 {model.linear.weight.item():.3f} 应该增加到接近目标值 3.0")
    
    return model, gradient

model, gradient = demonstrate_gradient_flow()

# ============================
# 3. 梯度下降的工作原理
# ============================
print("\n3. 梯度下降的工作原理")
print("-" * 50)

def visualize_gradient_descent():
    """可视化梯度下降过程"""
    
    # 定义一个简单的二次函数作为损失函数
    def loss_function(w):
        return (w - 3)**2  # 最小值在 w = 3
    
    def loss_gradient(w):
        return 2 * (w - 3)  # 梯度
    
    # 梯度下降参数
    learning_rate = 0.1
    initial_w = 0.0
    num_steps = 20
    
    # 记录训练过程
    w_history = [initial_w]
    loss_history = [loss_function(initial_w)]
    gradient_history = [loss_gradient(initial_w)]
    
    w = initial_w
    print("📈 梯度下降过程：")
    print(f"{'步骤':>4} {'权重':>8} {'损失':>8} {'梯度':>8} {'更新':>12}")
    print("-" * 50)
    
    for step in range(num_steps):
        grad = loss_gradient(w)
        update = -learning_rate * grad  # 负梯度方向
        w_new = w + update
        loss_val = loss_function(w)
        
        print(f"{step:4d} {w:8.3f} {loss_val:8.3f} {grad:8.3f} {update:+8.3f}")
        
        w = w_new
        w_history.append(w)
        loss_history.append(loss_function(w))
        gradient_history.append(loss_gradient(w))
        
        # 如果梯度很小，说明接近最优解
        if abs(grad) < 0.01:
            print(f"✅ 收敛！最终权重: {w:.3f}, 目标权重: 3.0")
            break
    
    # 可视化梯度下降过程
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失函数曲线
        w_range = np.linspace(-1, 5, 100)
        loss_range = [(w - 3)**2 for w in w_range]
        
        ax1.plot(w_range, loss_range, 'b-', label='损失函数 L(w)=(w-3)²', linewidth=2)
        ax1.plot(w_history, loss_history, 'ro-', label='梯度下降路径', markersize=5)
        ax1.set_xlabel('权重 w')
        ax1.set_ylabel('损失 L(w)')
        ax1.set_title('梯度下降优化过程')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 损失随训练步骤的变化
        ax2.plot(loss_history, 'g-o', linewidth=2, markersize=4)
        ax2.set_xlabel('训练步骤')
        ax2.set_ylabel('损失值')
        ax2.set_title('损失随训练步骤的下降')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/xiang/Documents/GitHub/NewStart/NewStart/gradient_descent.png', 
                    dpi=300, bbox_inches='tight')
        print(f"\n梯度下降可视化已保存")
        
    except Exception as e:
        print(f"可视化出错: {e}")
    
    return w_history, loss_history

w_hist, loss_hist = visualize_gradient_descent()

# ============================
# 4. 在你的PyTorch代码中的梯度流动
# ============================
print("\n4. 在你的PyTorch代码中的梯度流动")
print("-" * 50)

def analyze_pytorch_gradient_flow():
    """分析你的PyTorch代码中的梯度流动"""
    
    # 重现你的模型结构
    class SimpleClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, output_size)
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x
    
    model = SimpleClassifier(2, 4, 2)  # 简化版本便于演示
    criterion = nn.CrossEntropyLoss()
    
    # 创建示例数据
    x = torch.randn(3, 2)  # 3个样本，2个特征
    y = torch.tensor([0, 1, 0])  # 3个标签
    
    print("🔄 PyTorch训练步骤中的梯度：")
    print()
    
    # 前向传播
    print("1️⃣ 前向传播：")
    print(f"   输入 x: {x.shape}")
    output = model(x)
    print(f"   输出 output: {output.shape}")
    loss = criterion(output, y)
    print(f"   损失 loss: {loss.item():.4f}")
    print()
    
    # 查看参数初始状态
    print("2️⃣ 反向传播前的参数状态：")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"   {name}: grad = {param.grad.norm().item():.4f}")
        else:
            print(f"   {name}: grad = None")
    print()
    
    # 反向传播
    print("3️⃣ 反向传播：")
    loss.backward()
    print("   计算梯度完成！")
    print()
    
    # 查看计算出的梯度
    print("4️⃣ 计算出的梯度：")
    total_grad_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm**2
            print(f"   {name:12}: 梯度范数 = {grad_norm:.6f}")
    
    total_grad_norm = np.sqrt(total_grad_norm)
    print(f"   总梯度范数: {total_grad_norm:.6f}")
    print()
    
    # 模拟参数更新
    print("5️⃣ 参数更新 (学习率 = 0.01)：")
    lr = 0.01
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is not None:
                old_param = param.clone()
                param -= lr * param.grad
                update_size = (old_param - param).norm().item()
                print(f"   {name:12}: 更新大小 = {update_size:.6f}")

analyze_pytorch_gradient_flow()

# ============================
# 5. 梯度的问题和解决方案
# ============================
print("\n5. 梯度的常见问题和解决方案")
print("-" * 50)

gradient_problems = """
🚨 梯度相关的常见问题：

1️⃣ **梯度消失 (Gradient Vanishing)**：
   • 问题: 梯度在反向传播过程中逐渐变小，趋近于0
   • 原因: 激活函数导数小（如Sigmoid）、网络过深
   • 症状: 前几层参数几乎不更新，训练缓慢
   • 解决: 使用ReLU、残差连接、BatchNorm

2️⃣ **梯度爆炸 (Gradient Exploding)**：
   • 问题: 梯度变得非常大，导致参数更新过大
   • 原因: 权重初始化不当、学习率过大
   • 症状: 损失突然增大、NaN值出现
   • 解决: 梯度裁剪、降低学习率、权重初始化

3️⃣ **死亡ReLU (Dead ReLU)**：
   • 问题: ReLU神经元输出永远为0，梯度为0
   • 原因: 负值输入导致ReLU输出0，无法恢复
   • 症状: 部分神经元不更新
   • 解决: 使用Leaky ReLU、降低学习率

4️⃣ **梯度噪声**：
   • 问题: 小批量训练导致梯度估计不准确
   • 原因: 批量大小太小
   • 症状: 训练不稳定，震荡
   • 解决: 增大批量、使用动量优化器
"""
print(gradient_problems)

# ============================
# 6. 梯度监控和调试
# ============================
print("\n6. 梯度监控和调试技巧")
print("-" * 50)

def gradient_monitoring_demo():
    """演示梯度监控技巧"""
    
    # 创建一个可能有梯度问题的深层网络
    class DeepModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(10, 10) for _ in range(10)  # 10层
            ])
            self.output = nn.Linear(10, 1)
        
        def forward(self, x):
            for layer in self.layers:
                x = torch.sigmoid(layer(x))  # 使用Sigmoid（容易梯度消失）
            return self.output(x)
    
    model = DeepModel()
    criterion = nn.MSELoss()
    
    # 创建数据
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)
    
    # 前向传播和反向传播
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    print("🔍 梯度监控技巧：")
    print()
    
    # 1. 检查梯度范数
    print("1️⃣ 各层梯度范数：")
    layer_grads = []
    for i, layer in enumerate(model.layers):
        if layer.weight.grad is not None:
            grad_norm = layer.weight.grad.norm().item()
            layer_grads.append(grad_norm)
            print(f"   Layer {i:2d}: {grad_norm:.6f}")
    
    # 2. 梯度比率分析
    print("\n2️⃣ 梯度消失分析：")
    if len(layer_grads) > 1:
        ratios = [layer_grads[i+1]/layer_grads[i] if layer_grads[i] > 0 else 0 
                 for i in range(len(layer_grads)-1)]
        avg_ratio = np.mean(ratios)
        print(f"   平均梯度比率: {avg_ratio:.6f}")
        if avg_ratio < 0.1:
            print("   ⚠️  梯度消失风险！")
        elif avg_ratio > 10:
            print("   ⚠️  梯度爆炸风险！")
        else:
            print("   ✅ 梯度传播正常")
    
    # 3. 零梯度检测
    print("\n3️⃣ 零梯度检测：")
    zero_grad_count = 0
    total_params = 0
    for param in model.parameters():
        if param.grad is not None:
            zero_count = (param.grad.abs() < 1e-8).sum().item()
            zero_grad_count += zero_count
            total_params += param.numel()
    
    zero_ratio = zero_grad_count / total_params
    print(f"   零梯度参数比例: {zero_ratio:.2%}")
    if zero_ratio > 0.5:
        print("   ⚠️  过多参数梯度接近零！")

gradient_monitoring_demo()

# ============================
# 7. 优化器与梯度的关系
# ============================
print("\n7. 不同优化器如何使用梯度")
print("-" * 50)

optimizer_explanation = """
🎛️ 优化器如何使用梯度：

📊 **SGD (随机梯度下降)**：
   • 更新公式: θ = θ - lr × ∇θ
   • 直接使用梯度进行更新
   • 简单但可能震荡

🚀 **Adam (你代码中使用的)**：
   • 更新公式: θ = θ - lr × m̂ / (√v̂ + ε)
   • m̂: 梯度的一阶矩估计（动量）
   • v̂: 梯度的二阶矩估计（自适应学习率）
   • 结合了动量和自适应学习率

💨 **Momentum**：
   • 更新公式: θ = θ - (α×v + lr×∇θ)
   • 利用历史梯度信息加速收敛
   • 减少震荡

🎯 **RMSprop**：
   • 自适应调整学习率
   • 对频繁更新的参数降低学习率

在你的PyTorch代码中：
optimizer = optim.Adam(model.parameters(), lr=0.001)
• Adam会自动处理梯度的动量和自适应调整
• 你只需要调用optimizer.step()即可
"""
print(optimizer_explanation)

# ============================
# 8. 总结
# ============================
print("\n" + "="*70)
print("梯度作用总结")
print("="*70)

summary = """
🎓 梯度在神经网络中的关键作用：

🧭 **1. 指向优化方向**：
   • 梯度告诉我们如何调整参数来减少损失
   • 负梯度方向是损失函数下降最快的方向

⚡ **2. 控制学习速度**：
   • 梯度大小决定参数更新的幅度
   • 大梯度 → 大更新，小梯度 → 小更新

🔄 **3. 传递学习信号**：
   • 通过反向传播，梯度从输出层传递到输入层
   • 每一层的参数都能得到相应的更新信号

📊 **4. 在你的代码中**：
   ```python
   # 计算梯度
   loss.backward()           # 梯度告诉我们如何改进
   
   # 使用梯度更新参数  
   optimizer.step()          # Adam利用梯度智能更新参数
   
   # 清零梯度准备下一轮
   optimizer.zero_grad()     # 避免梯度累积
   ```

🎯 **关键理解**：
• 梯度 = 损失函数对参数的敏感度
• 优化器 = 梯度的智能使用方式
• 激活函数 = 梯度流动的控制器
• 学习率 = 梯度更新的缩放因子

💡 **最佳实践**：
✅ 监控梯度范数，避免梯度消失/爆炸
✅ 选择合适的激活函数（如ReLU）
✅ 使用合适的优化器（如Adam）
✅ 设置合理的学习率
"""
print(summary)

if __name__ == "__main__":
    print("\n🎉 梯度作用解析完成！")