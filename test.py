import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
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

# 模型实现
class CustomCIFAR10Model(nn.Module):
    def __init__(self, n, k, num_classes=100, hidden_dim=64):
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
            nn.ReLU()
        ) for _ in range(k)])  # 定义 φ_j
        
        # 嵌套线性变换
        self.linear = nn.Linear(64 * 8 * 8, hidden_dim)  # 展平后的特征映射
        
        # 输出分类层
        self.fc = nn.Linear(hidden_dim, num_classes)  # 分类层
    
    def forward(self, x):
        """
        前向传播实现:
        - x: 输入图像张量, 形状为 [batch_size, 3, 32, 32]
        """
        # 卷积特征提取
        x = self.conv(x)  # 输出形状 [batch_size, 64, 8, 8]
        x = x.view(x.size(0), -1)  # 展平特征映射为 [batch_size, 64 * 8 * 8]
        
        # 第一部分: 加权和
        part1 = 0
        for i in range(1, len(self.n)):
            # part1 += self.w[i] * self.n[i] + self.w[i-1] * self.n[i-1]
            part1 = self.w[i] * self.n[i] + self.w[i-1] * self.n[i-1]
        
        # 第二部分: 嵌套激活和线性变换
        linear_out = self.linear(x)  # ∑_(𝑖=0)^𝑛 的线性变换
        part2 = 0
        for j, phi in enumerate(self.phi_j):
            # part2 += phi(linear_out)  # 通过每个 φ_j 的嵌套非线性
            part2 = phi(linear_out)  # 通过每个 φ_j 的嵌套非线性
        
        # 合并两部分结果
        # Z = part1 + part2.sum(dim=1)
        Z = part1 + part2
        
        # 分类输出
        output = self.fc(Z)
        return output

# 模型实例化
n = 5  # 输入变量数量
k = 3  # 嵌套激活函数的数量
model = CustomCIFAR10Model(n=n, k=k)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 分类任务损失
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


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
        test_model(model, test_loader)  # 调用测试函数



# 开始训练
model = model.cuda()  # 将模型移动到 GPU
train_model(model, train_loader, criterion, optimizer, num_epochs=100)
test_model(model, test_loader)
