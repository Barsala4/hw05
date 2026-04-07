import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import sys

# 关键：确保当前目录在Python路径中，解决跨目录导入问题
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从同目录的lenet5.py导入LeNet5类
from lenet5 import LeNet5

# 设置中文字体（解决画图中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 数据集路径（相对于code目录，自动创建上级data文件夹）
data_root = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(data_root, exist_ok=True)

# 加载数据
train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义极简CNN模型（与任务一保持一致）
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练函数 (带记录功能)
def train_and_record(model, train_loader, test_loader, criterion, optimizer, epochs=5):
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 测试
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        accuracy = 100. * correct / len(test_loader.dataset)
        test_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Test Loss={avg_test_loss:.4f}, Acc={accuracy:.2f}%')
    
    return train_losses, test_losses, test_accuracies

# 主程序
if __name__ == "__main__":
    # 1. 训练极简CNN
    print("\n--- 开始训练极简 CNN ---")
    cnn_model = SimpleCNN().to(device)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
    cnn_criterion = nn.CrossEntropyLoss()
    cnn_train_loss, cnn_test_loss, cnn_acc = train_and_record(
        cnn_model, train_loader, test_loader, cnn_criterion, cnn_optimizer, epochs=5
    )

    # 2. 训练 LeNet-5
    print("\n--- 开始训练 LeNet-5 ---")
    lenet_model = LeNet5().to(device)
    lenet_optimizer = optim.SGD(lenet_model.parameters(), lr=0.01, momentum=0.9)
    lenet_criterion = nn.CrossEntropyLoss()
    lenet_train_loss, lenet_test_loss, lenet_acc = train_and_record(
        lenet_model, train_loader, test_loader, lenet_criterion, lenet_optimizer, epochs=10
    )

    # 3. 绘制并保存曲线图
    print("\n--- 生成可视化图表 ---")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    ax1.plot(cnn_train_loss, label='极简CNN 训练损失', marker='o', linestyle='-')
    ax1.plot(lenet_train_loss, label='LeNet-5 训练损失', marker='s', linestyle='--')
    ax1.set_title('训练损失曲线 (Training Loss)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(cnn_acc, label='极简CNN 准确率 (%)', marker='o', linestyle='-')
    ax2.plot(lenet_acc, label='LeNet-5 准确率 (%)', marker='s', linestyle='--')
    ax2.set_title('测试准确率曲线 (Test Accuracy)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    # 保存到上级assets文件夹
    save_path = os.path.join(os.path.dirna)