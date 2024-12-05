import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
# 数据预处理
import csv
import matplotlib.pyplot as plt


# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载 CIFAR-100 数据集
batch_size = 64
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

# 使用 DataLoader 时指定 num_workers 等参数
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)


class FourierFeature(nn.Module):
    def __init__(self, d, num_heads, k=1):
        super(FourierFeature, self).__init__()
        self.k = k
        self.a = nn.Parameter(torch.randn(d, d, k))  # 学习参数 a
        self.b = nn.Parameter(torch.randn(d, d, k))  # 学习参数 b
        self.num_heads = num_heads
        

    def forward(self, x):
        device = x.device  # 获取输入的设备
        self.a = self.a.to(device)  # 确保 a 和 b 在同一设备
        self.b = self.b.to(device)

        
        # 计算傅里叶变换中的 cos 和 sin 部分
        cos_part = torch.cos(x.unsqueeze(-1) * torch.arange(1, self.k+1).to(device))
        sin_part = torch.sin(x.unsqueeze(-1) * torch.arange(1, self.k+1).to(device))
        
        # print('before mult shapes:', cos_part.shape, self.a.shape)

        cos_part = torch.unsqueeze(cos_part, dim = 1)* self.a
        sin_part = torch.unsqueeze(sin_part, dim = 1)* self.b

        
        result = cos_part.sum(dim=-1) + sin_part.sum(dim=-1)  # 输出为 [B, N, C]     [B, 3, N*C]

        return result



class KAN_FourierAttention(nn.Module):
    def __init__(self, dim, num_heads,):
        super(KAN_FourierAttention, self).__init__()
        self.num_heads = num_heads
        self.fourier = FourierFeature(dim, num_heads=4)

    def forward(self, x):
        # print("x.shape:",x.shape)
        device = x.device
        
        

        # 获取傅里叶特征
        fourier_feature = self.fourier(x.to(device))
        

  
        return fourier_feature



# 模型实现
class CustomCIFAR10Model(nn.Module):
    def __init__(self, n, k, num_classes=100, hidden_dim=3072):
        """
        参数:
        - n: 输入变量数量（公式中的 n）
        - k: 嵌套激活函数的数量（公式中的 k）
        - num_classes: CIFAR-10 分类数目（10类）
        - hidden_dim: 中间隐藏单元数量
        """
        super(CustomCIFAR10Model, self).__init__()
        
        # 卷积层提取特征
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 输入通道 3，输出通道 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 下采样
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # 第一部分：线性加权和
        self.w = nn.Parameter(torch.randn(n))  # w_i
        self.n = nn.Parameter(torch.randn(n))  # n_i
        
        # 第二部分：嵌套激活函数 φ_j 和线性层
        self.phi_j = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        ) for _ in range(k)])  # 定义 φ_j
        
        # 嵌套线性变换
        self.linear = nn.Linear(3072, hidden_dim)  # 展平后的特征映射
        self.kan = KAN_FourierAttention(hidden_dim, 2)#feats = 64 * 8 * 8, hidden_dim = hidden_dim)
        # self.kan_second = KA_attention(out_dim = 1, feats = 64 * 8 * 8, hidden_dim = hidden_dim)
        
        # 输出分类层
        self.fc = nn.Linear(hidden_dim, num_classes)  # 分类层
    
    def forward(self, x):
        """
        前向传播实现:
        - x: 输入图像张量, 形状为 [batch_size, 3, 32, 32]
        """
        # 卷积特征提取
        # x = self.conv(x)  # 输出形状 [batch_size, 64, 8, 8]
        x = x.view(x.size(0), -1)  # 展平特征映射为 [batch_size, 64 * 8 * 8]
        
        # 第一部分: 加权和
        part1 = 0
        for i in range(1, len(self.n)):
            part1 += self.w[i] * self.n[i] + self.w[i-1] * self.n[i-1]
        
        linear_out = self.kan(x)
        # print('after linear shape: ', linear_out.shape) #[batch, n, n]
        
        part2 = torch.sum(linear_out, dim = 1)
 
 
        Z = part1 + part2#self.kan_second(part1 + part2)
        
        # 分类输出
        output = self.fc(Z)
        return output

# 模型实例化
n = 5  # 输入变量数量
k = 2  # 嵌套激活函数的数量
model = CustomCIFAR10Model(n=n, k=k)

# 定义损失函数和优化器


criterion = nn.CrossEntropyLoss()  # 分类任务损失
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)



# 用于保存训练和测试数据
train_results = []
test_results = []

# 测试模型函数
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        # 添加进度条
        with tqdm(test_loader, desc="Testing") as pbar:
            for images, labels in pbar:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                loss = criterion(outputs, labels)  # 计算测试损失
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)  # 获取预测类别
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 更新进度条显示
                accuracy = 100 * correct / total
                pbar.set_postfix(accuracy=accuracy)

    # 计算平均损失和最终正确率
    avg_loss = total_loss / len(test_loader)
    final_accuracy = 100 * correct / total
    print(f"Test Accuracy: {final_accuracy:.2f}%, Test Loss: {avg_loss:.4f}")

    # 保存测试结果
    test_results.append({"epoch": len(train_results), "loss": avg_loss, "accuracy": final_accuracy})

    return avg_loss, final_accuracy


# 训练模型函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        # 添加进度条
        with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]") as pbar:
            for images, labels in pbar:
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)  # 使用 GPU（如果可用）

                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

                # 计算正确率
                _, predicted = torch.max(outputs, 1)  # 获取每个样本的预测类别
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # 更新进度条显示
                accuracy = 100 * correct / total
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy)

        avg_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # 保存训练结果
        train_results.append({"epoch": epoch+1, "loss": avg_loss, "accuracy": train_accuracy})

        # 测试并保存结果
        print(f"\nTesting after epoch {epoch+1}:")
        test_loss, test_accuracy = test_model(model, test_loader)

        # 保存结果到文件
        save_results_to_csv("training_results.csv", train_results)
        save_results_to_csv("testing_results.csv", test_results)


# 保存结果到 CSV 文件
def save_results_to_csv(filename, results):
    keys = results[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


# 绘制损失和准确率曲线
def plot_results(train_results, test_results):
    train_epochs = [r["epoch"] for r in train_results]
    train_losses = [r["loss"] for r in train_results]
    train_accuracies = [r["accuracy"] for r in train_results]

    test_epochs = [r["epoch"] for r in test_results]
    test_losses = [r["loss"] for r in test_results]
    test_accuracies = [r["accuracy"] for r in test_results]

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_epochs, train_losses, label="Train Loss", color="blue")
    plt.plot(test_epochs, test_losses, label="Test Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_epochs, train_accuracies, label="Train Accuracy", color="blue")
    plt.plot(test_epochs, test_accuracies, label="Test Accuracy", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.show()


# 开始训练
model = model.cuda()  # 将模型移动到 GPU
train_model(model, train_loader, criterion, optimizer, num_epochs=100)

# 测试完成后绘制结果
plot_results(train_results, test_results)