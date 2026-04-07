import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层1：1输入通道 -> 6输出通道，5×5卷积核，无填充
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # 池化层：2×2窗口，步长2（最大池化）
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 卷积层2：6输入通道 -> 16输出通道，5×5卷积核
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 卷积层3：16输入通道 -> 120输出通道，5×5卷积核（等效全连接）
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        # 全连接层
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        # 输入x: (batch_size, 1, 28, 28) -> 补全至32×32
        x = F.pad(x, (2, 2, 2, 2))  # 28×28 -> 32×32
        x = self.pool(F.tanh(self.conv1(x)))  # C1->S2: 32×32 -> 28×28 -> 14×14
        x = self.pool(F.tanh(self.conv2(x)))  # C3->S4: 14×14 -> 10×10 -> 5×5
        x = F.tanh(self.conv3(x))  # C5: 5×5 -> 1×1×120
        x = x.view(-1, 120)  # 展平
        x = F.tanh(self.fc1(x))  # F6: 120 -> 84
        x = self.fc2(x)  # 输出层: 84 -> 10
        return x

# 各层尺寸与参数表（放入report.md）
"""
| 层名   | 类型       | 输入尺寸       | 输出尺寸       | 参数量计算逻辑                  |
|--------|------------|----------------|----------------|---------------------------------|
| 输入层 | -          | (B,1,28,28)    | (B,1,28,28)    | -                               |
| C1     | 卷积       | (B,1,32,32)    | (B,6,28,28)    | 6×(5×5×1)+6 = 156               |
| S2     | 池化       | (B,6,28,28)    | (B,6,14,14)    | 无参数（平均池化）              |
| C3     | 卷积       | (B,6,14,14)    | (B,16,10,10)   | 16×(5×5×6)+16 = 2416            |
| S4     | 池化       | (B,16,10,10)   | (B,16,5,5)     | 无参数                        |
| C5     | 卷积       | (B,16,5,5)     | (B,120,1,1)    | 120×(5×5×16)+120 = 48120        |
| F6     | 全连接     | (B,120)        | (B,84)         | 120×84+84 = 10164               |
| 输出层 | 全连接     | (B,84)         | (B,10)         | 84×10+10 = 850                   |
总参数量：61606
"""