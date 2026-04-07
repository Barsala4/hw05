现象	原因分析	修改点
导入 torch 报错：No module named 'torch'	未安装 PyTorch 或环境未激活	执行 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118（匹配 CUDA 版本）
数据集下载失败：HTTP 错误	网络问题或官方源访问受限	手动下载 MNIST 数据集（http://yann.lecun.com/exdb/mnist/），放入 ./data/MNIST/raw 目录
运行时 CUDA out of memory	batch_size 过大或 GPU 显存不足	将 batch_size 从 64 调至 32，或改用 CPU 训练（device='cpu'）
中文路径报错：UnicodeDecodeError	代码中路径包含中文	将数据集保存路径改为纯英文（如 ./data 而非 ./数据集）
准确率长期低于 90%	模型过浅/学习率不当	增加卷积核数量（如 16→32），将学习率从 0.001 调至 0.0005