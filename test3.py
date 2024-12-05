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
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

# 加载 CIFAR-10 数据集
batch_size = 64
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# 模型定义
class EnhancedKolmogorovArnoldNetwork(nn.Module):
    def __init__(self, n, k):
        super(EnhancedKolmogorovArnoldNetwork, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc1 = nn.Linear(256 * 4 * 4, n)
        self.w = nn.Parameter(torch.randn(n))
        self.n = nn.Parameter(torch.randn(n))
        self.phi_j = nn.Parameter(torch.randn(k))
        
        self.mlp = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, k)
        )
        self.phi_ij = nn.Parameter(torch.randn(256 * 4 * 4, k))
        self.linear = nn.Linear(k, 100)
    
    def forward(self, X):
        X = self.conv_layers(X)
        X = X.view(X.size(0), -1)
        part1 = sum(self.w[i] * self.n[i] + self.w[i-1] * self.n[i-1] for i in range(1, len(self.w)))
        # linear_combination = torch.matmul(X, self.phi_ij)
        linear_combination = self.mlp(X)
        part2 = torch.sum(self.phi_j * torch.relu(self.linear(linear_combination)), dim=1)
        Z = part1 + part2
        return Z



# 模型实例化

model = EnhancedKolmogorovArnoldNetwork(n=256, k=100)

# 定义损失函数和优化器
# criterion = nn.CrossEntropyLoss()  # 分类任务损失
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 不使用 label_smoothing

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)



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