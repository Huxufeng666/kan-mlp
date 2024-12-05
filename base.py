import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 下载 CIFAR-10 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


# 模型和相关模块的定义
class AdaptiveMultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(AdaptiveMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        print("out",out.shape)
        return out, attn

class ReAttention(nn.Module):
    def __init__(self, dim, num_heads=8, re_scale=0.5):
        super(ReAttention, self).__init__()
        self.re_attn_layer = AdaptiveMultiHeadAttention(dim, num_heads)
        self.re_scale = re_scale

    def forward(self, x):
        x, attn = self.re_attn_layer(x)
        reweighted_x = x * (attn * self.re_scale).sum(dim=1, keepdim=True)
        return reweighted_x

class AttentionGuidedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(AttentionGuidedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.attn_layer = AdaptiveMultiHeadAttention(dim=out_channels, num_heads=4)

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.view(B, H * W, C)
        x, _ = self.attn_layer(x)
        x = x.view(B, C, H, W)
        return x

class DynamicFeatureSelector(nn.Module):
    def __init__(self, in_features, num_select=128):
        super(DynamicFeatureSelector, self).__init__()
        self.select_layer = nn.Linear(in_features, num_select)

    def forward(self, x):
    
        weights = torch.sigmoid(self.select_layer(x.T))
        # x = x * weights

        # print("x",x.shape)
        return  weights

class KAN_Model(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(KAN_Model, self).__init__()
        self.attn_layer1 = AdaptiveMultiHeadAttention(input_dim, num_heads=8)
        self.reattention_layer = ReAttention(input_dim, num_heads=8)
        self.attn_guided_conv = AttentionGuidedConv(in_channels=3, out_channels=64)
        self.dynamic_selector = DynamicFeatureSelector(64)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        
        print("x.shape",x.shape)
        x = self.attn_guided_conv(x)
        B, C, H, W = x.shape
        x = x.view(B, C * H * W)
        x = self.dynamic_selector(x)
        x, attn = self.attn_layer1(x)
        x = self.reattention_layer(x)
        x = self.fc(x)
        return x

# 初始化模型
model = KAN_Model(input_dim=32*32*3, num_classes=10).cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# 测试函数
def test(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return correct / total

# 训练和测试循环
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, trainloader, optimizer, criterion)
    test_accuracy = test(model, testloader, criterion)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')