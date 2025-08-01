# PyTorch 快速入门总结

这个示例展示了PyTorch的核心概念，通过一个完整的二分类神经网络项目。

## 🎯 主要学习点

### 1. 张量操作
```python
import torch

# 创建张量
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
random_tensor = torch.randn(3, 4)  # 随机张量

# 张量运算
result = x + y  # 加法
result = torch.mm(a, b)  # 矩阵乘法
```

### 2. 数据处理
```python
from torch.utils.data import DataLoader, TensorDataset

# 创建数据集和数据加载器
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 3. 定义神经网络
```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 4. 训练循环
```python
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(num_epochs):
    for data, target in dataloader:
        optimizer.zero_grad()    # 梯度清零
        output = model(data)     # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()          # 反向传播
        optimizer.step()         # 更新参数
```

### 5. 模型评估
```python
model.eval()  # 切换到评估模式
with torch.no_grad():  # 禁用梯度计算
    outputs = model(test_data)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == labels).float().mean()
```

## 📊 运行结果

✅ **训练成功**: 50个epoch后达到99.5%准确率
✅ **模型保存**: 成功保存和加载模型权重
✅ **可视化**: 生成了训练历史和决策边界图
✅ **预测**: 单样本预测功能正常

## 🚀 关键概念

1. **张量(Tensor)**: PyTorch的核心数据结构
2. **自动微分**: 自动计算梯度，支持反向传播
3. **nn.Module**: 所有神经网络模型的基类
4. **DataLoader**: 高效的批量数据加载
5. **设备管理**: CPU/GPU之间的数据转移

## 📁 生成的文件

- `pytorch_quickstart.py`: 完整的示例代码
- `simple_classifier.pth`: 训练好的模型权重
- `training_history.png`: 训练损失和准确率曲线
- `decision_boundary.png`: 神经网络的决策边界可视化

这个例子涵盖了从数据准备到模型部署的完整机器学习流程，是学习PyTorch的绝佳起点！
