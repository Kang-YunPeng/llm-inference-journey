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
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
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