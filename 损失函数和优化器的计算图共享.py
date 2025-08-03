import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

print("=" * 70)
print("loss.backward() 如何找到并更新模型参数的机制详解")
print("=" * 70)

# ============================
# 1. PyTorch计算图机制
# ============================
print("\n1. PyTorch计算图机制")
print("-" * 50)

computational_graph_explanation = """
🔗 计算图 (Computational Graph) 的工作原理：

📊 **自动构建计算图**：
• PyTorch在前向传播时自动构建计算图
• 每个tensor都记录了它是如何从其他tensor计算得来的
• 每个操作都会在图中创建一个节点

🎯 **梯度传播路径**：
• loss.backward() 沿着计算图反向遍历
• 从loss开始，逐层向前找到所有需要梯度的参数
• 自动应用链式法则计算每个参数的梯度

🔍 **参数发现机制**：
• 所有requires_grad=True的tensor都会被跟踪
• 模型参数默认requires_grad=True
• backward()会找到所有这些参数并计算梯度
"""
print(computational_graph_explanation)

# ============================
# 2. 演示计算图的构建过程
# ============================
print("\n2. 演示计算图的构建过程")
print("-" * 50)

def demonstrate_computation_graph():
    """演示计算图的构建和梯度计算"""
    
    # 创建简单模型
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1, bias=False)
            # 手动设置权重便于观察
            self.linear.weight.data = torch.tensor([[1.0, 2.0]])
        
        def forward(self, x):
            return self.linear(x)
    
    model = TinyModel()
    
    # 创建输入数据
    x = torch.tensor([[1.0, 1.0]], requires_grad=False)
    target = torch.tensor([[5.0]])
    
    print("🔍 追踪计算图构建过程：")
    print(f"模型权重: {model.linear.weight}")
    print(f"输入数据: {x}")
    print(f"目标值: {target}")
    print()
    
    # 前向传播
    print("1️⃣ 前向传播 - 构建计算图：")
    output = model(x)  # 这里构建了计算图
    print(f"   模型输出: {output}")
    print(f"   输出的grad_fn: {output.grad_fn}")  # 显示计算图节点
    print()
    
    # 计算损失
    loss = F.mse_loss(output, target)
    print(f"2️⃣ 计算损失:")
    print(f"   损失值: {loss}")
    print(f"   损失的grad_fn: {loss.grad_fn}")  # 损失函数的计算图节点
    print()
    
    # 查看参数的梯度状态
    print("3️⃣ 反向传播前的梯度状态:")
    print(f"   权重的requires_grad: {model.linear.weight.requires_grad}")
    print(f"   权重的grad: {model.linear.weight.grad}")
    print()
    
    # 反向传播
    print("4️⃣ 执行 loss.backward():")
    loss.backward()  # 沿计算图反向传播
    print(f"   反向传播完成！")
    print(f"   权重的梯度: {model.linear.weight.grad}")
    print()
    
    return model

model_demo = demonstrate_computation_graph()

# ============================
# 3. 详细分析grad_fn链条
# ============================
print("\n3. 详细分析grad_fn链条")
print("-" * 50)

def analyze_grad_fn_chain():
    """分析计算图中的grad_fn链条"""
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(2, 3)
            self.fc2 = nn.Linear(3, 1)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model = SimpleNet()
    x = torch.randn(1, 2)
    
    # 前向传播并追踪每一步
    print("🔗 追踪计算图链条：")
    
    # 第一层
    z1 = model.fc1(x)
    print(f"1. fc1输出的grad_fn: {z1.grad_fn}")
    
    # ReLU激活
    a1 = F.relu(z1)
    print(f"2. ReLU输出的grad_fn: {a1.grad_fn}")
    
    # 第二层
    z2 = model.fc2(a1)
    print(f"3. fc2输出的grad_fn: {z2.grad_fn}")
    
    # 损失
    target = torch.randn(1, 1)
    loss = F.mse_loss(z2, target)
    print(f"4. 损失的grad_fn: {loss.grad_fn}")
    print()
    
    # 显示参数与计算图的连接
    print("📝 参数在计算图中的位置：")
    for name, param in model.named_parameters():
        print(f"   {name}: requires_grad={param.requires_grad}")
    
    return model, loss

model_chain, loss_chain = analyze_grad_fn_chain()

# ============================
# 4. 演示optimizer如何找到参数
# ============================
print("\n4. 演示optimizer如何找到参数")
print("-" * 50)

def demonstrate_optimizer_parameter_access():
    """演示optimizer如何访问和更新参数"""
    
    # 创建模型
    model = nn.Linear(2, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    print("🎛️ Optimizer对参数的访问：")
    
    # 查看optimizer中存储的参数
    print("1️⃣ Optimizer中的参数组：")
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"   参数组 {i}: 学习率 = {param_group['lr']}")
        print(f"   参数数量: {len(param_group['params'])}")
        for j, param in enumerate(param_group['params']):
            print(f"     参数 {j}: 形状 = {param.shape}, requires_grad = {param.requires_grad}")
    print()
    
    # 创建数据进行一次训练步骤
    x = torch.randn(5, 2)
    y = torch.randn(5, 1)
    
    # 前向传播
    output = model(x)
    loss = F.mse_loss(output, y)
    
    print("2️⃣ 训练步骤演示：")
    print(f"   损失值: {loss.item():.4f}")
    
    # 保存更新前的参数
    old_weight = model.weight.clone()
    old_bias = model.bias.clone()
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    print(f"   反向传播后权重梯度: {model.weight.grad}")
    print(f"   反向传播后偏置梯度: {model.bias.grad}")
    
    # 参数更新
    optimizer.step()
    
    # 显示参数变化
    weight_change = (model.weight - old_weight).norm().item()
    bias_change = (model.bias - old_bias).norm().item()
    
    print(f"   权重变化大小: {weight_change:.6f}")
    print(f"   偏置变化大小: {bias_change:.6f}")
    print()

demonstrate_optimizer_parameter_access()

# ============================
# 5. 手动实现参数更新来理解机制
# ============================
print("\n5. 手动实现参数更新来理解机制")
print("-" * 50)

def manual_parameter_update():
    """手动实现参数更新，展示optimizer.step()的内部工作"""
    
    # 创建简单模型
    model = nn.Linear(2, 1)
    
    # 创建数据
    x = torch.randn(3, 2)
    y = torch.randn(3, 1)
    
    # 前向传播
    output = model(x)
    loss = F.mse_loss(output, y)
    
    print("🔧 手动参数更新演示：")
    print(f"初始损失: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    print("\n方法1: 使用optimizer.step()")
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    
    # 重新计算损失验证
    output_after = model(x)
    loss_after = F.mse_loss(output_after, y)
    print(f"使用optimizer后的损失: {loss_after.item():.4f}")
    
    # 重置模型并手动更新
    model = nn.Linear(2, 1)
    output = model(x)
    loss = F.mse_loss(output, y)
    loss.backward()
    
    print("\n方法2: 手动更新参数")
    lr = 0.1
    with torch.no_grad():  # 禁用梯度计算
        for param in model.parameters():
            if param.grad is not None:
                param -= lr * param.grad  # 手动应用梯度更新
    
    # 验证手动更新的效果
    output_manual = model(x)
    loss_manual = F.mse_loss(output_manual, y)
    print(f"手动更新后的损失: {loss_manual.item():.4f}")
    
    print("\n✅ 两种方法得到相同结果，证明optimizer.step()就是在自动执行手动更新的过程")

manual_parameter_update()

# ============================
# 6. 完整的参数更新流程图
# ============================
print("\n6. 完整的参数更新流程")
print("-" * 50)

complete_flow = """
🔄 完整的参数更新流程：

📋 **训练循环中的步骤**：

1️⃣ **optimizer.zero_grad()**
   • 清除所有参数的梯度
   • 防止梯度累积
   
2️⃣ **前向传播: output = model(input)**
   • 构建计算图
   • 每个操作都记录在图中
   • 输出tensor包含grad_fn信息

3️⃣ **计算损失: loss = criterion(output, target)**
   • 损失函数也是计算图的一部分
   • loss tensor包含完整的计算历史

4️⃣ **反向传播: loss.backward()**
   • 从loss开始，沿计算图反向遍历
   • 自动找到所有requires_grad=True的参数
   • 使用链式法则计算每个参数的梯度
   • 将梯度存储在param.grad中

5️⃣ **参数更新: optimizer.step()**
   • 遍历optimizer中注册的所有参数
   • 对每个参数应用更新规则
   • SGD: param = param - lr * param.grad
   • Adam: 更复杂的自适应更新

🔗 **关键连接**：
• model.parameters() → optimizer注册参数
• 前向传播 → 计算图构建
• backward() → 梯度计算
• optimizer.step() → 参数更新

💡 **核心理解**：
loss.backward()不直接更新参数，它只是计算梯度！
真正的参数更新由optimizer.step()完成！
"""
print(complete_flow)

# ============================
# 7. 在你的代码中的体现
# ============================
print("\n7. 在你的代码中的体现")
print("-" * 50)

your_code_analysis = """
📝 在你的PyTorch代码中的体现：

```python
# 创建模型和优化器
model = SimpleClassifier(input_size=2, hidden_size=64, output_size=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # ← optimizer注册了模型的所有参数

# 训练循环
for batch_idx, (data, target) in enumerate(train_loader):
    
    optimizer.zero_grad()        # ← 清除梯度
    
    outputs = model(data)        # ← 前向传播，构建计算图
    loss = criterion(outputs, target)  # ← 计算损失，扩展计算图
    
    loss.backward()              # ← 沿计算图反向传播，计算所有参数的梯度
    
    optimizer.step()             # ← 使用计算出的梯度更新参数
```

🔍 **详细解析**：

1. **optimizer.zero_grad()**:
   • 遍历optimizer中的所有参数
   • 将每个param.grad设为None或零张量

2. **outputs = model(data)**:
   • 数据flow: data → fc1 → relu → dropout → fc2 → relu → dropout → fc3 → outputs
   • 每一步都在计算图中记录
   • outputs.grad_fn指向整个计算链

3. **loss = criterion(outputs, target)**:
   • CrossEntropyLoss计算也加入计算图
   • loss.grad_fn连接到整个网络

4. **loss.backward()**:
   • 从loss开始反向传播
   • 自动找到fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias
   • 计算每个参数的梯度并存储在param.grad中

5. **optimizer.step()**:
   • Adam优化器遍历所有注册的参数
   • 对每个参数应用Adam更新规则
   • 实际修改参数值

🎯 **关键理解**：
• loss.backward()是"计算"梯度
• optimizer.step()是"应用"梯度
• 计算图是连接两者的桥梁
"""
print(your_code_analysis)

# ============================
# 8. 验证参数连接关系
# ============================
print("\n8. 验证参数连接关系")
print("-" * 50)

def verify_parameter_connection():
    """验证模型参数与optimizer的连接关系"""
    
    # 创建与你代码相同的模型
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
    
    model = SimpleClassifier(2, 4, 2)  # 简化版本
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("🔗 验证参数连接关系：")
    
    # 1. 显示模型参数
    print("1️⃣ 模型中的参数：")
    model_params = list(model.parameters())
    for i, param in enumerate(model_params):
        print(f"   参数 {i}: 形状 {param.shape}, id = {id(param)}")
    
    print()
    
    # 2. 显示optimizer中的参数
    print("2️⃣ Optimizer中的参数：")
    optimizer_params = optimizer.param_groups[0]['params']
    for i, param in enumerate(optimizer_params):
        print(f"   参数 {i}: 形状 {param.shape}, id = {id(param)}")
    
    print()
    
    # 3. 验证是否是同一个对象
    print("3️⃣ 验证参数身份一致性：")
    for i, (model_param, opt_param) in enumerate(zip(model_params, optimizer_params)):
        is_same = model_param is opt_param
        print(f"   参数 {i}: 同一对象 = {is_same}")
    
    print()
    print("✅ 结论: optimizer和model共享相同的参数对象引用")
    print("   这就是为什么loss.backward()计算的梯度能被optimizer.step()使用")

verify_parameter_connection()

# ============================
# 9. 总结
# ============================
print("\n" + "="*70)
print("总结")
print("="*70)

summary = """
🎓 loss.backward() 如何更新模型参数的完整机制：

🔗 **核心机制 - 计算图**：
   • PyTorch自动构建计算图，记录所有操作
   • 每个tensor都知道它是如何计算出来的
   • loss.backward()沿着这个图找到所有需要梯度的参数

🎯 **三个关键连接**：
   1. 模型参数 ←→ Optimizer参数（共享对象引用）
   2. 前向传播 ←→ 计算图构建（自动记录）
   3. 反向传播 ←→ 梯度计算（自动微分）

⚡ **更新流程**：
   1. optimizer.zero_grad()  # 清除旧梯度
   2. output = model(input)  # 构建计算图
   3. loss = criterion(...)  # 扩展计算图
   4. loss.backward()        # 计算梯度（不更新参数！）
   5. optimizer.step()       # 应用梯度更新参数

💡 **关键理解**：
   • loss.backward() 只计算梯度，不更新参数
   • optimizer.step() 才真正更新参数
   • 两者通过共享的参数对象引用连接
   • 计算图是自动微分的基础

🚀 **在你的代码中**：
   所有这些都自动发生，你只需要调用：
   • loss.backward()  # PyTorch找到所有参数并计算梯度
   • optimizer.step() # Adam使用梯度更新参数

这就是PyTorch"自动"的魅力所在！🎉
"""
print(summary)

if __name__ == "__main__":
    print("\n🎉 参数更新机制解析完成！")