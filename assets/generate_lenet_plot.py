import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 设置绘图风格
plt.style.use('default')
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 80)
ax.axis('off')

# 定义颜色
COLOR_INPUT = '#A8DADC'
COLOR_CONV = '#457B9D'
COLOR_POOL = '#F1FAEE'
COLOR_FC = '#E63946'
COLOR_TEXT = '#000000'

def add_layer(ax, x, y, width, height, color, label):
    """添加图层矩形"""
    rect = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', 
            fontsize=10, fontweight='bold', color=COLOR_TEXT)

def add_arrow(ax, x1, y1, x2, y2):
    """添加箭头连接"""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# 绘制网络结构（按 LeNet-5 顺序）
layers = [
    (10, 60, 15, 10, COLOR_INPUT, '输入\n(B,1,28,28)'),
    (30, 60, 15, 10, COLOR_CONV, 'C1 卷积\n6@28x28\n(k=5)'),
    (50, 60, 15, 10, COLOR_POOL, 'S2 池化\n6@14x14\n(2x2)'),
    (70, 60, 15, 10, COLOR_CONV, 'C3 卷积\n16@10x10\n(k=5)'),
    (90, 60, 15, 10, COLOR_POOL, 'S4 池化\n16@5x5\n(2x2)'),
    (30, 40, 15, 10, COLOR_CONV, 'C5 卷积\n120@1x1\n(k=5)'),
    (55, 40, 15, 10, COLOR_FC, 'F6 全连接\n84'),
    (80, 40, 15, 10, COLOR_FC, '输出\n10 (0-9)')
]

# 绘制图层
for x, y, w, h, c, l in layers:
    add_layer(ax, x, y, w, h, c, l)

# 绘制箭头连接（横向主路径）
for i in range(len(layers)-1):
    if i == 2:  # C3 到 S4 跳过中间，直接连到下一组
        add_arrow(ax, layers[i][0]+layers[i][2]/2, layers[i][1]+5, 
                  layers[i+1][0]+layers[i+1][2]/2, layers[i+1][1]+5)
        continue
    add_arrow(ax, layers[i][0]+layers[i][2]/2, layers[i][1]+5, 
              layers[i+1][0]+layers[i+1][2]/2, layers[i+1][1]+5)

# 绘制垂直连接（C5->F6->输出）
add_arrow(ax, layers[3][0]+layers[3][2]/2, layers[3][1], 
          layers[4][0]+layers[4][2]/2, layers[4][1]+10)
add_arrow(ax, layers[4][0]+layers[4][2]/2, layers[4][1], 
          layers[5][0]+layers[5][2]/2, layers[5][1]+10)
add_arrow(ax, layers[5][0]+layers[5][2]/2, layers[5][1]+5, 
          layers[6][0]+layers[6][2]/2, layers[6][1]+5)
add_arrow(ax, layers[6][0]+layers[6][2]/2, layers[6][1]+5, 
          layers[7][0]+layers[7][2]/2, layers[7][1]+5)

# 添加标题
plt.title('LeNet-5 网络结构可视化', fontsize=16, fontweight='bold', pad=20)

# 保存图片（关键：保存到 assets 文件夹）
plt.tight_layout()
plt.savefig('assets/lenet5_structure.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ LeNet-5 结构图片已生成：assets/lenet5_structure.png")