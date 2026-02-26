import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from safetensors.torch import save_file, load_file


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.inference_mode():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 返回 (最大值, 最大值索引)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


if __name__ == '__main__':
    # =============================================================================
    # 第一部分：张量与自动微分验证
    # =============================================================================
    print("=" * 60)
    print("【第一部分】张量与自动微分验证")
    print("=" * 60)

    # 不推荐 x = torch.Tensor([1, 2, 3])  默认float32（可能非预期）
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x ** 2 + 2 * x + 1
    y_sum = y.sum()
    y_sum.backward()

    print(f"    梯度 dx = {x.grad}")
    print(f"    理论梯度 (2x+2): {2*x + 2}")
    print(f"    梯度计算验证通过: {torch.allclose(x.grad, 2*x + 2)}\n")

    x.grad.zero_()
    print(f"    梯度清零后, x.grad: {x.grad}\n")

    # =============================================================================
    # 第二部分：构建多层感知机
    # =============================================================================
    print("=" * 60)
    print("【第二部分】构建多层感知机 (MLP)")
    print("=" * 60)

    model = MLP()
    print(f"模型结构:\n{model}\n")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}\n")

    # =============================================================================
    # 第三部分：数据加载
    # =============================================================================
    print("=" * 60)
    print("【第三部分】数据加载 (MNIST)")
    print("=" * 60)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    use_gpu = torch.cuda.is_available()

    batch_size = 256
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_gpu,
        persistent_workers=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_gpu,
        persistent_workers=True
    )

    print(f"    训练集样本数: {len(train_dataset):,}")
    print(f"    测试集样本数: {len(test_dataset):,}")
    print(f"    批次大小 (batch_size): {batch_size}")
    print(f"    训练批次数: {len(train_loader)}\n")

    # =============================================================================
    # 第四部分：训练配置
    # =============================================================================
    print("=" * 60)
    print("【第四部分】训练配置")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    使用设备: {device}")
    model = model.to(device)
    base_model = model
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
        print("    已启用 torch.compile 加速")
    else:
        print("    警告: torch.compile 未启用，训练速度可能较慢")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    print(f"    损失函数: CrossEntropyLoss (label_smoothing=0.1)")
    print(f"    优化器: AdamW (lr=0.001)")

    # =============================================================================
    # 第五部分：训练循环
    # =============================================================================
    num_epochs = 50
    print("=" * 60)
    print(f"【第五部分】训练循环 ({num_epochs} 轮)")
    print("=" * 60)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    print(f"    学习率调度: CosineAnnealingLR (T_max={num_epochs})")

    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        test_acc = evaluate(model, test_loader, device)
        test_accuracies.append(test_acc)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | Train Loss: {train_losses[-1]:.4f} | Test Accuracy: {test_acc:.2f}% | LR: {current_lr:.6f}")

    # =============================================================================
    # 第六部分：结果验证与模型保存
    # =============================================================================
    print("=" * 60)
    print("【第六部分】结果验证与模型保存")
    print("=" * 60)

    final_accuracy = test_accuracies[-1]
    print(f"    最终测试准确率: {final_accuracy:.2f}%")

    model_path = './day01/mlp_model.safetensors'
    save_file(base_model.state_dict(), model_path)
    print(f"    模型已保存至: {model_path}")

    loaded_model = MLP()
    loaded_model.load_state_dict(load_file(model_path, device=str(device)))
    loaded_model.to(device)
    loaded_model.eval()

    verify_acc = evaluate(loaded_model, test_loader, device)
    print(f"    加载后验证准确率: {verify_acc:.2f}%")
    print(f"    模型保存/加载验证通过\n")
