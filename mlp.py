# Himari Department of Sumisora University
# Hoshiori Lab.

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# PART1. 加载数据
print("正在加载数据...")
housing = fetch_california_housing()
X, y = housing.data, housing.target

print(f"数据集大小: {X.shape}")
print(f"特征数量: {X.shape[1]}")
print(f"样本数量: {X.shape[0]}")


# 这个tts函数是专门用来划分数据集的，参数 arrays: 需要分割的数据集，通常是特征矩阵 X 和标签 y。
# test_size: 测试集的大小。train_size: 训练集的大小。random_state: 随机数种子。shuffle: 是否在分割前打乱数据。默认为 True。
# stratify: 是否进行分层分割。
# 返回值：返回一个列表，其中包含分割后的训练集和测试集。
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# PART2. 数据预处理

# 标准化:让所有特征数值范围统一(均值0,方差1)
# ss函数是用来做标准化的，减去特征的均值，除以特征的标准差，结果：数据均值为0，标准差为1
# 注意：ss()函数是一个类，要把创建的实例赋给scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 训练集:计算均值方差并转换
X_test = scaler.transform(X_test)        # 测试集:只用训练集的参数转换

print('reshape前的测试集标签\n',y_test)
# torch.FloatTensor() 将数据转换为pytorch浮点数张量
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)  # 变成(16512,1)列向量  reshape让y变列向量,方便算loss（因为原始数据是一个数组，要把它变成1列的矩阵）
y_test = torch.FloatTensor(y_test).reshape(-1, 1)
print('reshape后的测试集标签\n',y_test)

# PART3.搭建MLP神经网络

class MLP(nn.Module):   # 注意，这里创建继承nn.Module的类，nn.Moudle是提前设置好的一个神经网络的基类，继承它能自动获得很多功能:参数管理、GPU加速、保存/加载模型等
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()  # 调用父类nn.Module的初始化方法，必须写!不然PyTorch不知道这是个神经网络
        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层→隐藏层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层→输出层
        # 可以看到类里有很多属性，包括fc1、fc2、relu


    def forward(self, x):
        # 这就是前向传播!数据像流水一样经过网络
        # 对应还有backward(反向传播) PyTorch会自动计算backward,你只需定义forward!
        out = self.fc1(x)  # 第一层线性变换
        out = self.relu(out)  # 激活
        out = self.fc2(out)  # 第二层线性变换
        return out


# 创建模型实例
input_size = 8  # 8个特征
hidden_size = 64  # 隐藏层64个神经元(可调)
output_size = 1  # 输出1个值(房价)

model = MLP(input_size, hidden_size, output_size)
print(model)
print(f"\n模型参数总数: {sum(p.numel() for p in model.parameters())}")

# PART4.定义损失函数和优化器

# 损失函数:用来衡量预测值和真实值差多少
criterion = nn.MSELoss()  # Mean Squared Error,均方误差

# 优化器:用来更新模型参数
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"损失函数: {criterion}")
print(f"优化器: {optimizer}")


# PART5.模型训练

# 训练参数
num_epochs = 100  # 训练100轮
batch_size = 64  # 每次喂64个样本

# 记录loss变化(用来画图)
train_losses = []

print("开始训练...")
for epoch in range(num_epochs):
    model.train()  # 设置为训练模式

    # 前向传播
    outputs = model(X_train)  # 模型预测
    loss = criterion(outputs, y_train)  # 计算loss 注意这里criterion是个nn.MSELoss对象 loss = 计算出来的损失值(结果)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空上一轮的梯度
    # 为什么要清零? PyTorch默认会累加梯度，如果不清零,这轮的梯度会叠加到上一轮,导致错误!
    loss.backward()  # 自动计算所有参数的梯度
    # 这就是PyTorch的魔法!不用手动求导
    # 内部用链式法则计算loss对每个参数的偏导数
    # 注意：loss是张量，pytorch的张量自带.backward()函数
    optimizer.step()  # 根据梯度更新参数
    # 每个参数 = 旧值 - 学习率 × 梯度
    # 让loss变小的方向前进一小步

    # 记录loss
    train_losses.append(loss.item())  #把PyTorch张量转成普通Python数字 方便打印和记录

    # 每10轮打印一次
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("训练完成!")

model.eval()  # 切换到评估模式
with torch.no_grad():  # 测试时不需要计算梯度
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'测试集Loss: {test_loss.item():.4f}')
