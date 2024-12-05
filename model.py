
import torch
import torch.nn as nn
from fourierkan import KAN_FourierAttention




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
        output = self.fc( Z )
        return output
    
    
    

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

