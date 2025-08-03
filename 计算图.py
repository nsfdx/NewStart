import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print("=" * 70)
print("loss.backward() 如何找到计算图的深层机制解析")
print("=" * 70)

# ============================
# 1. 计算图存储在tensor本身
# ============================
print("\n1. 计算图存储在tensor本身")
print("-" * 50)

tensor_graph_explanation = """
🔗 核心机制：计算图存储在tensor的属性中

📊 **每个tensor都携带计算图信息**：
• grad_fn: 指向创建该tensor的函数节点
• requires_grad: 是否需要计算梯度
• is_leaf: 是否是叶子节点（用户创建的tensor）

🎯 **loss.backward() 的工作原理**：
• loss tensor本身就"知道"整个计算图
• 通过grad_fn属性，可以访问整个计算链
• 从loss开始，递归遍历所有父节点

💡 **关键理解**：
计算图不是独立存在的数据结构，而是分布式存储在各个tensor中！
"""
print(tensor_graph_explanation)

# ============================
# 2. 演示tensor如何携带计算图信息
# ============================
print("\n2. 演示tensor如何携带计算图信息")
print("-" * 50)

def demonstrate_tensor_graph_storage():
    """演示tensor如何存储计算图信息"""
    
    # 创建需要梯度的叶子节点
    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)
    
    print("🌱 叶子节点（用户创建的tensor）:")
    print(f"x = {x}, grad_fn = {x.grad_fn}, is_leaf = {x.is_leaf}")
    print(f"y = {y}, grad_fn = {y.grad_fn}, is_leaf = {y.is_leaf}")
    print()
    
    # 进行计算，观察计算图的构建
    print("🔄 计算过程中计算图的构建:")
    
    # 第一步计算
    z1 = x * y
    print(f"z1 = x * y = {z1}")
    print(f"z1.grad_fn = {z1.grad_fn}")
    print(f"z1.is_leaf = {z1.is_leaf}")
    print()
    
    # 第二步计算
    z2 = z1 + 10
    print(f"z2 = z1 + 10 = {z2}")
    print(f"z2.grad_fn = {z2.grad_fn}")
    print(f"z2.is_leaf = {z2.is_leaf}")
    print()
    
    # 第三步计算
    loss = z2 ** 2
    print(f"loss = z2 ** 2 = {loss}")
    print(f"loss.grad_fn = {loss.grad_fn}")
    print(f"loss.is_leaf = {loss.is_leaf}")
    print()
    
    return loss, x, y

loss_demo, x_demo, y_demo = demonstrate_tensor_graph_storage()

# ============================
# 3. 深入探索grad_fn链条
# ============================
print("\n3. 深入探索grad_fn链条")
print("-" * 50)

def explore_grad_fn_chain(tensor):
    """递归探索grad_fn链条"""
    
    print("🔍 从loss开始递归探索计算图:")
    
    def recursive_explore(node, depth=0):
        indent = "  " * depth
        if hasattr(node, 'next_functions'):
            print(f"{indent}节点: {node}")
            print(f"{indent}类型: {type(node).__name__}")
            
            # 探索下一层节点
            for i, (next_fn, _) in enumerate(node.next_functions):
                if next_fn is not None:
                    print(f"{indent}  └─ 下级节点 {i}:")
                    recursive_explore(next_fn, depth + 2)
                else:
                    print(f"{indent}  └─ 叶子节点 {i} (None)")
        else:
            print(f"{indent}叶子节点: {node}")
    
    if tensor.grad_fn is not None:
        recursive_explore(tensor.grad_fn)
    else:
        print("这是一个叶子节点，没有grad_fn")

explore_grad_fn_chain(loss_demo)

# ============================
# 4. 模拟backward()的工作过程
# ============================
print("\n4. 模拟backward()的工作过程")
print("-" * 50)

def simulate_backward_process():
    """模拟backward()如何遍历计算图"""
    
    # 创建简单的计算图
    a = torch.tensor([2.0], requires_grad=True)
    b = torch.tensor([3.0], requires_grad=True)
    
    c = a * b      # MulBackward
    d = c + 1      # AddBackward  
    loss = d ** 2  # PowBackward
    
    print("📊 构建的计算图:")
    print(f"loss = (a * b + 1) ** 2")
    print(f"a = {a.item()}, b = {b.item()}")
    print(f"loss = {loss.item()}")
    print()
    
    print("🔄 backward()的遍历过程（模拟）:")
    
    # 模拟backward过程
    print("1. 从loss开始 (grad = 1.0)")
    loss_grad = 1.0
    
    print("2. PowBackward: d_loss/d_d = 2 * d")
    d_value = (a * b + 1).item()
    d_grad = 2 * d_value * loss_grad
    print(f"   d的梯度 = {d_grad}")
    
    print("3. AddBackward: d_d/d_c = 1")
    c_grad = 1.0 * d_grad
    print(f"   c的梯度 = {c_grad}")
    
    print("4. MulBackward: d_c/d_a = b, d_c/d_b = a")
    a_grad = b.item() * c_grad
    b_grad = a.item() * c_grad
    print(f"   a的梯度 = {a_grad}")
    print(f"   b的梯度 = {b_grad}")
    
    # 验证我们的手动计算
    print("\n✅ 验证计算结果:")
    loss.backward()
    print(f"PyTorch计算的a梯度: {a.grad.item()}")
    print(f"PyTorch计算的b梯度: {b.grad.item()}")
    print(f"手动计算是否正确: a={abs(a.grad.item() - a_grad) < 1e-6}, b={abs(b.grad.item() - b_grad) < 1e-6}")

simulate_backward_process()

# ============================
# 5. 在你的神经网络中的计算图
# ============================
print("\n5. 在你的神经网络中的计算图")
print("-" * 50)

def analyze_neural_network_graph():
    """分析神经网络中的计算图结构"""
    
    # 重现你的模型
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
    
    model = SimpleClassifier(2, 4, 2)
    criterion = nn.CrossEntropyLoss()
    
    # 创建输入数据
    x = torch.randn(3, 2)
    target = torch.tensor([0, 1, 0])
    
    print("🧠 神经网络计算图分析:")
    
    # 前向传播的每一步
    print("前向传播过程:")
    print(f"输入 x: {x.shape}, grad_fn = {x.grad_fn}")
    
    h1 = model.fc1(x)
    print(f"fc1输出: {h1.shape}, grad_fn = {type(h1.grad_fn).__name__}")
    
    a1 = F.relu(h1)
    print(f"relu1输出: {a1.shape}, grad_fn = {type(a1.grad_fn).__name__}")
    
    d1 = model.dropout(a1)
    print(f"dropout1输出: {d1.shape}, grad_fn = {type(d1.grad_fn).__name__}")
    
    h2 = model.fc2(d1)
    print(f"fc2输出: {h2.shape}, grad_fn = {type(h2.grad_fn).__name__}")
    
    a2 = F.relu(h2)
    print(f"relu2输出: {a2.shape}, grad_fn = {type(a2.grad_fn).__name__}")
    
    d2 = model.dropout(a2)
    print(f"dropout2输出: {d2.shape}, grad_fn = {type(d2.grad_fn).__name__}")
    
    output = model.fc3(d2)
    print(f"最终输出: {output.shape}, grad_fn = {type(output.grad_fn).__name__}")
    
    loss = criterion(output, target)
    print(f"损失: {loss.shape}, grad_fn = {type(loss.grad_fn).__name__}")
    print()
    
    # 显示计算图的深度
    def count_graph_depth(tensor):
        if tensor.grad_fn is None:
            return 0
        
        max_depth = 0
        if hasattr(tensor.grad_fn, 'next_functions'):
            for next_fn, _ in tensor.grad_fn.next_functions:
                if next_fn is not None:
                    # 这里我们简化，不递归计算
                    max_depth = max(max_depth, 1)
        return max_depth + 1
    
    depth = count_graph_depth(loss)
    print(f"📏 计算图深度: {depth} (简化计算)")
    
    return loss

loss_nn = analyze_neural_network_graph()

# ============================
# 6. 计算图的内存管理
# ============================
print("\n6. 计算图的内存管理")
print("-" * 50)

def demonstrate_graph_memory():
    """演示计算图的内存管理"""
    
    x = torch.tensor([1.0], requires_grad=True)
    
    print("🗄️ 计算图的内存管理:")
    
    # 创建计算图
    y = x ** 2
    z = y * 3
    loss = z + 10
    
    print(f"loss创建后，loss.grad_fn存在: {loss.grad_fn is not None}")
    print(f"计算图链条: loss -> {type(loss.grad_fn).__name__} -> {type(y.grad_fn).__name__} -> ...")
    
    # backward后计算图的状态
    loss.backward()
    print(f"backward后，loss.grad_fn存在: {loss.grad_fn is not None}")
    
    # 再次backward会发生什么？
    try:
        loss.backward()
    except RuntimeError as e:
        print(f"再次backward失败: {e}")
    
    # retain_graph=True的情况
    y2 = x ** 2
    z2 = y2 * 3  
    loss2 = z2 + 10
    
    print(f"\n使用retain_graph=True:")
    loss2.backward(retain_graph=True)
    print(f"第一次backward后，loss2.grad_fn存在: {loss2.grad_fn is not None}")
    
    loss2.backward(retain_graph=True)
    print(f"第二次backward成功，但梯度会累积!")
    print(f"x的累积梯度: {x.grad}")

demonstrate_graph_memory()

# ============================
# 7. 动态计算图 vs 静态计算图
# ============================
print("\n7. 动态计算图 vs 静态计算图")
print("-" * 50)

dynamic_vs_static = """
🔄 PyTorch的动态计算图特点:

🟢 **动态计算图 (PyTorch)**:
• 每次前向传播都重新构建计算图
• 支持控制流（if、for、while）
• 图结构可以在运行时改变
• 便于调试，可以使用Python调试器

🔵 **静态计算图 (TensorFlow 1.x)**:
• 先定义图结构，再执行计算
• 图结构固定，不能在运行时改变
• 优化更充分，执行效率更高
• 难以调试

💡 **PyTorch动态图的实现**:
每次调用.backward()时：
1. 从loss tensor开始
2. 通过grad_fn递归遍历整个图
3. 应用链式法则计算梯度
4. 默认释放图内存（除非retain_graph=True）

这就是为什么loss.backward()不需要显式的图引用！
"""
print(dynamic_vs_static)

# ============================
# 8. 自定义函数的计算图
# ============================
print("\n8. 自定义函数的计算图")
print("-" * 50)

class CustomSquare(torch.autograd.Function):
    """自定义平方函数，展示如何集成到计算图"""
    
    @staticmethod
    def forward(ctx, input):
        """前向传播"""
        ctx.save_for_backward(input)  # 保存用于反向传播的tensor
        return input ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        input, = ctx.saved_tensors
        return grad_output * 2 * input

def demonstrate_custom_function():
    """演示自定义函数如何集成到计算图"""
    
    print("🛠️ 自定义函数的计算图集成:")
    
    x = torch.tensor([3.0], requires_grad=True)
    
    # 使用自定义函数
    y = CustomSquare.apply(x)
    loss = y.sum()
    
    print(f"x = {x}")
    print(f"y = CustomSquare(x) = {y}")
    print(f"y.grad_fn = {y.grad_fn}")
    print(f"loss.grad_fn = {loss.grad_fn}")
    
    # 反向传播
    loss.backward()
    print(f"x的梯度 = {x.grad} (应该是 2*3 = 6)")

demonstrate_custom_function()

# ============================
# 9. 总结
# ============================
print("\n" + "="*70)
print("总结")
print("="*70)

summary = """
🎓 loss.backward() 如何找到计算图的完整机制：

🔗 **核心原理**：
   • 计算图不是独立的数据结构
   • 而是分布式存储在每个tensor的grad_fn属性中
   • loss tensor通过grad_fn链可以访问整个计算图

📊 **tensor的关键属性**：
   • grad_fn: 指向创建该tensor的函数节点
   • requires_grad: 是否需要计算梯度
   • is_leaf: 是否是叶子节点（用户创建）

🔄 **backward()的工作流程**：
   1. 从loss.grad_fn开始
   2. 递归访问next_functions
   3. 对每个节点应用链式法则
   4. 将梯度累积到叶子节点的.grad属性

💡 **在你的代码中**：
   ```python
   outputs = model(data)           # 构建计算图，存储在outputs.grad_fn中
   loss = criterion(outputs, target)  # 扩展计算图，存储在loss.grad_fn中
   loss.backward()                 # 从loss.grad_fn开始遍历整个图
   ```

🎯 **关键理解**：
   • loss.backward()不需要外部图引用
   • 因为loss本身就"携带"了整个计算图的信息
   • 通过grad_fn链，可以找到所有需要梯度的参数
   • 这就是PyTorch动态计算图的精妙设计！

🚀 **动态图的优势**：
   • 每次前向传播重新构建，支持动态控制流
   • 便于调试，可以检查中间结果
   • 自然支持变长序列等动态输入

这种设计让PyTorch既强大又灵活！🎉
"""
print(summary)

if __name__ == "__main__":
    print("\n🎉 计算图机制解析完成！")