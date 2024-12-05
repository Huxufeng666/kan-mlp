import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 数据预处理
import csv
import matplotlib.pyplot as plt


# 数据预处理：转换为张量，并进行标准化处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化：均值和方差
])

# 加载 CIFAR-10 数据集
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 类别标签
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class KolmogorovArnoldNetwork(nn.Module):
    def __init__(self, n, k):
        super(KolmogorovArnoldNetwork, self).__init__()
        
        # 图像输入的维度为 32x32x3 (CIFAR-10)
        self.fc1 = nn.Linear(32 * 32 * 3, n)  # 输入层，展开图像为一维向量
        
        # 线性变换权重和输入
        self.w = nn.Parameter(torch.randn(n))  # 长度为 n 的权重
        self.n = nn.Parameter(torch.randn(n))  # 长度为 n 的输入
        
        # phi_j 和 phi_ij 是与非线性激活函数相关的参数
        self.phi_j = nn.Parameter(torch.randn(k))  # 长度为 k 的参数
        self.phi_ij = nn.Parameter(torch.randn(n, k))  # (n, k) 维度的参数
        
        # 定义一个线性层，负责对加权和进行线性变换
        self.linear = nn.Linear(n, 100)  # 输出 10 类的分类
        
    def forward(self, X):
        # 扁平化图像并传递给全连接层
        X = X.view(-1, 32 * 32 * 3)
        X = self.fc1(X)
        
        # 第一部分的加权和求和: ∑_(i=1)^(n-1) (w_i * n_i + w_(i-1) * n_(i-1))
        part1 = sum(self.w[i] * self.n[i] + self.w[i-1] * self.n[i-1] for i in range(1, len(self.w)))
        
        # 第二部分的加权和求和: ∑_(j=0)^k φ_j * Linear( ∑_(i=0)^n φ_ij * X_i )
        part2 = 0
        for j in range(len(self.phi_j)):
            linear_combination = sum(self.phi_ij[i, j] * X[i] for i in range(len(X)))  # ∑_(i=0)^n φ_ij * X_i
            part2 += self.phi_j[j] * torch.relu(self.linear(linear_combination))  # 应用线性层和 ReLU 激活

        # 返回最终的 Z
        Z = part1 + part2
        Z = self.linear(X) 
        return Z

# 测试模型函数
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # 添加进度条
        with tqdm(test_loader, desc="Testing") as pbar:
            for images, labels in pbar:
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)  # 获取预测类别
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # 更新进度条显示
                accuracy = 100 * correct / total
                pbar.set_postfix(accuracy=accuracy)

    # 打印最终正确率
    final_accuracy = 100 * correct / total
    print(f"Test Accuracy: {final_accuracy:.2f}%")
    return final_accuracy


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
                images, labels = images.cuda(), labels.cuda()  # 使用 GPU（如果可用）
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()

                # 计算正确率
                _, predicted = torch.max(outputs, 1)  # 获取每个样本的预测类别
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # 更新进度条显示
                accuracy = 100 * correct / total
                pbar.set_postfix(loss=loss.item(), accuracy=accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f},Train Accuracy: {accuracy:.2f}%")
        
        print(f"\nTesting after epoch {epoch+1}:")
        test_model(model, testloader)  # 调用测试函数

# 参数设置
n = 100  # 假设网络层的神经元个数
k = 10   # 假设非线性部分的参数个数

# 实例化模型
model = KolmogorovArnoldNetwork(n, k).cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 对于分类问题使用交叉熵损失
optimizer = optim.AdamW(model.parameters(), lr=0.0001)



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
        test_loss, test_accuracy = test_model(model, testloader)

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
train_model(model, trainloader, criterion, optimizer, num_epochs=100)

# 测试完成后绘制结果
plot_results(train_results, test_results)