"""
PyTorch 快速入门示例
通过一个完整的例子学习 PyTorch 的核心概念：
1. 张量 (Tensors)
2. 数据集和数据加载器
3. 神经网络模型
4. 损失函数和优化器
5. 训练和评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================
# 1. 张量基础操作
# ============================
print("\n" + "="*50)
print("1. PyTorch 张量基础")
print("="*50)

# 创建张量
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
print(f"一维张量: {x}")

# 创建矩阵
matrix = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(f"二维张量: \n{matrix}")

# 随机张量
random_tensor = torch.randn(3, 4)  # 3x4的随机张量
print(f"随机张量: \n{random_tensor}")

# 张量运算
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(f"张量加法: {a + b}")
print(f"张量乘法: {a * b}")
print(f"矩阵乘法: {torch.mm(matrix, matrix.T)}")

# ============================
# 2. 生成示例数据集
# ============================
print("\n" + "="*50)
print("2. 创建数据集 - 二元分类问题")
print("="*50)

def generate_data(n_samples=1000):
    """生成一个简单的二元分类数据集"""
    np.random.seed(42)
    
    # 生成两个高斯分布的数据（正态分布）
    class_0 = np.random.multivariate_normal([2, 2], [[1, 0.5], [0.5, 1]], n_samples//2)
    class_1 = np.random.multivariate_normal([6, 6], [[1, -0.5], [-0.5, 1]], n_samples//2)

    
    X = np.vstack([class_0, class_1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    return X, y

# 生成数据
X_train, y_train = generate_data(800)
X_test, y_test = generate_data(200)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

print(f"训练数据形状: {X_train_tensor.shape}")
print(f"训练标签形状: {y_train_tensor.shape}")
print(f"特征范围: [{X_train_tensor.min():.2f}, {X_train_tensor.max():.2f}]")

# ============================
# 3. 创建数据加载器
# ============================
print("\n" + "="*50)
print("3. 数据加载器")
print("="*50)

# 创建数据集
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练批次数: {len(train_loader)}")
print(f"测试批次数: {len(test_loader)}")

# ============================
# 4. 定义神经网络模型
# ============================
print("\n" + "="*50)
print("4. 神经网络模型")
print("="*50)

class SimpleClassifier(nn.Module):
    """简单的二分类神经网络"""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # 前向传播
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 创建模型
model = SimpleClassifier(input_size=2, hidden_size=64, output_size=2)
model = model.to(device)  # 移动到设备

print(f"模型结构:")
print(model)

# 计算参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数数: {total_params}")
print(f"可训练参数数: {trainable_params}")

# ============================
# 5. 定义损失函数和优化器
# ============================
print("\n" + "="*50)
print("5. 损失函数和优化器")
print("="*50)

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

print(f"损失函数: {criterion}")
print(f"优化器: {optimizer}")

# ============================
# 6. 训练模型
# ============================
print("\n" + "="*50)
print("6. 训练模型")
print("="*50)

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """训练模型"""
    model.train()  # 设置为训练模式
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, target)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # 计算平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return train_losses, train_accuracies

# 训练模型
num_epochs = 50
train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs)

# ============================
# 7. 评估模型
# ============================
print("\n" + "="*50)
print("7. 模型评估")
print("="*50)

def evaluate_model(model, test_loader, criterion):
    """评估模型"""
    model.eval()  # 设置为评估模式
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100 * correct / total
    
    return test_loss, accuracy

test_loss, test_accuracy = evaluate_model(model, test_loader, criterion)
print(f'测试损失: {test_loss:.4f}')
print(f'测试准确率: {test_accuracy:.2f}%')

# ============================
# 8. 可视化结果
# ============================
print("\n" + "="*50)
print("8. 可视化训练过程")
print("="*50)

def plot_training_history(train_losses, train_accuracies):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(train_losses)
    ax1.set_title('训练损失')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accuracies)
    ax2.set_title('训练准确率')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/xiang/Documents/GitHub/NewStart/NewStart/training_history.png', dpi=300, bbox_inches='tight')
    print("训练历史图已保存为 training_history.png")

def plot_decision_boundary(model, X, y):
    """绘制决策边界"""
    model.eval()
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    grid_points = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(device)
    
    with torch.no_grad():
        outputs = model(grid_points)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()
    
    predictions = predictions.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, predictions, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.title('神经网络决策边界')
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.savefig('/Users/xiang/Documents/GitHub/NewStart/NewStart/decision_boundary.png', dpi=300, bbox_inches='tight')
    print("决策边界图已保存为 decision_boundary.png")

# 绘制图表
try:
    plot_training_history(train_losses, train_accuracies)
    plot_decision_boundary(model, X_test, y_test)
except ImportError:
    print("matplotlib 未安装，跳过可视化部分")

# ============================
# 9. 模型保存和加载
# ============================
print("\n" + "="*50)
print("9. 模型保存和加载")
print("="*50)

# 保存模型
model_path = '/Users/xiang/Documents/GitHub/NewStart/NewStart/simple_classifier.pth'
torch.save(model.state_dict(), model_path)
print(f"模型已保存到: {model_path}")

# 加载模型
new_model = SimpleClassifier(input_size=2, hidden_size=64, output_size=2)
new_model.load_state_dict(torch.load(model_path))
new_model.eval()
print("模型加载成功")

# 验证加载的模型
test_loss_new, test_accuracy_new = evaluate_model(new_model, test_loader, criterion)
print(f'加载后的模型测试准确率: {test_accuracy_new:.2f}%')

# ============================
# 10. 单个样本预测
# ============================
print("\n" + "="*50)
print("10. 单个样本预测")
print("="*50)

def predict_single_sample(model, x):
    """预测单个样本"""
    model.eval()
    with torch.no_grad():
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)  # 添加batch维度
        output = model(x_tensor)
        probabilities = F.softmax(output, dim=1)
        _, predicted_class = torch.max(output, 1)
        
        return predicted_class.item(), probabilities.squeeze().cpu().numpy()

# 预测几个样本
test_samples = [[2, 2], [6, 6], [4, 4]]
for sample in test_samples:
    pred_class, probabilities = predict_single_sample(model, sample)
    print(f"样本 {sample}: 预测类别 = {pred_class}, 概率 = {probabilities}")

print("\n" + "="*50)
print("PyTorch 核心概念总结")
print("="*50)
print("""
🔥 PyTorch 核心概念:

1. **张量 (Tensors)**: PyTorch的基本数据结构，类似NumPy数组但支持GPU加速
2. **自动微分**: torch.autograd 自动计算梯度
3. **nn.Module**: 所有神经网络的基类
4. **DataLoader**: 高效的数据批处理和加载
5. **优化器**: 自动更新模型参数
6. **设备管理**: CPU/GPU之间的数据移动

🚀 训练流程:
1. 定义模型 → 2. 准备数据 → 3. 前向传播 → 4. 计算损失 → 5. 反向传播 → 6. 更新参数

💡 关键方法:
- model.train(): 训练模式
- model.eval(): 评估模式  
- optimizer.zero_grad(): 梯度清零
- loss.backward(): 反向传播
- optimizer.step(): 参数更新
- torch.no_grad(): 禁用梯度计算
""")
