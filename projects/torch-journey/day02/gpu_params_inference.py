import torch
import torch.nn.init as init
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from safetensors.torch import load_file
from day01.mlp_training import MLP
import time
import os

print("\n" + "=" * 70)
print("【第一部分】加载 Day 1 模型")
print("=" * 70)


model_path = 'day01/mlp_model.safetensors'
if os.path.exists(model_path):
    print(f" 找到模型文件：{model_path}")
else:
    print(f" 未找到模型文件：{model_path}")
    exit(1)


model = MLP()
state_dict = load_file(model_path)
model.load_state_dict(state_dict)
print(f"    模型参数加载成功")

# =============================================================================
# 第二部分：参数管理
# =============================================================================
print("\n" + "=" * 70)
print("【第二部分】参数管理")
print("=" * 70)

# 2.1 打印所有参数名和形状
print("\n所有参数名和形状：")
print("-" * 70)
for name, param in model.named_parameters():
    print(f"  {name:30s} | shape: {str(tuple(param.shape)):15s} | numel: {param.numel():6,}")

# 2.2 访问特定层的参数
print("\n访问第一层线性层的权重和偏置：")
first_layer_weight = model.layers[0].weight
first_layer_bias = model.layers[0].bias
print(f"  权重形状：{str(tuple(first_layer_weight.shape))}")
print(f"  偏置形状：{str(tuple(first_layer_bias.shape))}")
print(f"  权重均值：{first_layer_weight.mean().item():.6f}")
print(f"  权重标准差：{first_layer_weight.std().item():.6f}")

# 2.3 修改某层参数（例如，重新初始化）
print("\n   使用 Xavier 均匀分布重新初始化第一层权重：")
print(f"  初始化前 - 均值：{first_layer_weight.mean().item():.6f}, 标准差：{first_layer_weight.std().item():.6f}")
init.xavier_uniform_(first_layer_weight)
print(f"  初始化后 - 均值：{first_layer_weight.mean().item():.6f}, 标准差：{first_layer_weight.std().item():.6f}")
print(f"  Xavier 初始化完成")

# 2.4 冻结某层参数
print("\n   冻结第一层参数（requires_grad=False）：")
for param in model.layers[0].parameters():
    param.requires_grad = False
print(f"  第一层权重 requires_grad: {model.layers[0].weight.requires_grad}")
print(f"  第一层偏置 requires_grad: {model.layers[0].bias.requires_grad}")

# =============================================================================
# 第三部分：设备管理
# =============================================================================
print("\n" + "=" * 70)
print("【第三部分】设备管理")
print("=" * 70)

# 自动选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  自动选择设备：{device}")

# =============================================================================
# 第四部分：数据加载
# =============================================================================
print("\n" + "=" * 70)
print("【第四部分】数据加载（MNIST 测试集）")
print("=" * 70)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

batch_size = 256
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True
)

print(f"  测试集样本数：{len(test_dataset):,}")
print(f"  批次大小：{batch_size}")
print(f"  测试批次数：{len(test_loader)}")

# =============================================================================
# 第五部分：推理模式与批量推理
# =============================================================================
print("\n" + "=" * 70)
print("【第五部分】推理模式与批量推理")
print("=" * 70)

# 切换到评估模式
model.eval()

# 批量推理函数
def batch_inference(model, data_loader, device):
    """批量推理并返回所有预测结果"""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.inference_mode():  # 不计算梯度
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return all_preds, all_labels, all_probs

# 5.1 执行推理
print("\n   执行批量推理...")
model = model.to(device)
predictions, true_labels, probabilities = batch_inference(model, test_loader, device)
print(f"    推理完成，共 {len(predictions)} 个样本")

# 5.2 计算准确率
correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
accuracy = 100 * correct / len(predictions)
print(f"  测试准确率：{accuracy:.2f}%")

# =============================================================================
# 第六部分：CPU vs GPU 推理速度对比
# =============================================================================
print("\n" + "=" * 70)
print("【第七部分】CPU vs GPU 推理速度对比")
print("=" * 70)

# 准备测试数据
test_images, test_labels = next(iter(test_loader))
test_images = test_images[:64]  # 取 64 个样本
test_labels = test_labels[:64]
num_runs = 10

# 6.1 CPU 推理
print("\n   CPU 推理速度测试：")
model_cpu = MLP()
model_cpu.load_state_dict(load_file(model_path))
model_cpu.eval()
model_cpu = model_cpu.to('cpu')

cpu_times = []
with torch.inference_mode():
    for _ in range(num_runs):
        start = time.time()
        _ = model_cpu(test_images)
        end = time.time()
        cpu_times.append(end - start)

cpu_avg = sum(cpu_times) / len(cpu_times)
cpu_std = (sum((t - cpu_avg) ** 2 for t in cpu_times) / len(cpu_times)) ** 0.5
print(f"  平均推理时间：{cpu_avg * 1000:.2f} ± {cpu_std * 1000:.2f} ms")
print(f"  吞吐量：{64 / cpu_avg:.1f} samples/s")

# 6.2 GPU 推理（如果可用）
if torch.cuda.is_available():
    print("\n   GPU 推理速度测试：")
    model_gpu = MLP()
    model_gpu.load_state_dict(load_file(model_path))
    model_gpu.eval()
    model_gpu = model_gpu.to('cuda')
    test_images_gpu = test_images.to('cuda')
    
    # 预热
    with torch.inference_mode():
        _ = model_gpu(test_images_gpu)
    
    gpu_times = []
    with torch.inference_mode():
        for _ in range(num_runs):
            start = time.time()
            _ = model_gpu(test_images_gpu)
            end = time.time()
            gpu_times.append(end - start)
    
    gpu_avg = sum(gpu_times) / len(gpu_times)
    gpu_std = (sum((t - gpu_avg) ** 2 for t in gpu_times) / len(gpu_times)) ** 0.5
    print(f"  平均推理时间：{gpu_avg * 1000:.2f} ± {gpu_std * 1000:.2f} ms")
    print(f"  吞吐量：{64 / gpu_avg:.1f} samples/s")
    
    # 加速比
    speedup = cpu_avg / gpu_avg
    print(f"\n  GPU 加速比：{speedup:.2f}x")
else:
    print("\n   GPU 不可用，跳过 GPU 测试")