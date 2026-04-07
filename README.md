# hw05
pip install -r requirements.txt
# 若需GPU版本，额外安装对应CUDA的PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# 运行极简CNN
python code/minimal_cnn.py
# 运行LeNet-5
python code/train_lenet.py
# 一键运行（训练两个模型并生成对比结果）
python code
