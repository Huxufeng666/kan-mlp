
import torch
import torch.nn as nn

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
