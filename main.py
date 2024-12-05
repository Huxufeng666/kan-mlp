import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import  CustomCIFAR10Model,KolmogorovArnoldNetwork
from loss_accuracy import save_results_to_csv,plot_results

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


# model = CustomCIFAR10Model(n=5, k=2)
model = KolmogorovArnoldNetwork(n=100, k=10)
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
        plot_results(train_results, test_results,save_path="cifar100_training_results.png")


# 开始训练
model = model.cuda()  # 将模型移动到 GPU
train_model(model, train_loader, criterion, optimizer, num_epochs=100)

# 测试完成后绘制结果
