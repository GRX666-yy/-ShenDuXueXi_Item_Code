import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#设置超参数和设备
BATCH_SIZE=128
EPOCHS=20
ALPHA=0.001
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#--------------------------------------全连接神经网络MLP--------------------------------------
#加载CSV数据
#file_path = r"D:\Pytorch\digit_mnist_test.csv"
#data = pd.read_csv(file_path)
#提取标签和像素
#abels = data.iloc[:,0].values
#pixels = data.iloc[:,1:].values
# 1. 先用 ToTensor 加载，计算 mean 和 std
temp_transform = transforms.Compose([
    transforms.ToTensor()
])
train_dataset=datasets.MNIST(
    root='D:\Pytorch\data',train=True,download=True,transform=temp_transform
)
all_imgs = torch.stack([img for img, _ in train_dataset])
mean = all_imgs.mean().item()
std = all_imgs.std().item()
print(f'Mean: {mean:.4f}, Std: {std:.4f}')
# 2. 用带 Normalize 的完整 transform 重新加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,),(std,))
])

train_dataset=datasets.MNIST(
    root='D:\Pytorch\data',train=True,download=False,transform=transform
)
test_dataset=datasets.MNIST(
    root='D:\Pytorch\data',train=False,download=False,transform=transform
)
train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

#全连接模型定义
class MLP(nn.Module):#nn.Module：所有网络的基类，提供参数管理、设备切换等基础设施，必须调用
    def __init__(self):
        super(MLP,self).__init__()#super().__init__()调用父类的构造函数，初始化 nn.Module 的内部机制
        self.flatten=nn.Flatten()# 把 28×28 的图像展平成 784 的向量;.Flatten将多维展为一维，保持batch维不变；输入[batch,1,28,28] → 输出[batch,784]
        self.fc1=nn.Linear(28*28,256)# 全连接层：784 → 256；全连接层（线性层），公式为 y = xW^T + b
        self.relu1=nn.ReLU()# 激活函数
        self.fc2=nn.Linear(256,64)# 全连接层：256 → 64
        self.relu2=nn.ReLU()# 激活函数；ReLU(x) = max(0, x)
        self.out=nn.Linear(64,10)# 全连接层：64 → 10(输出层的10个类别)
    #前向传播
    def forward(self,x):
        x=self.flatten(x)#调用上面的self.flatten(),将[batch,1,28,28] → [batch,784]
        x=self.relu1(self.fc1(x))# 784 → 256，ReLU 激活
        x=self.relu2(self.fc2(x))
        x=self.out(x)# 64 → 10（原始分数 logits：表示输出层数据为加softmax）
        return x#原始分数（logits）不应用 softmax，因为 CrossEntropyLoss 内部的LogSoftmax + NLLLoss会处理
model=MLP().to(DEVICE)

#损失函数与优化器
criterion=nn.CrossEntropyLoss()#.CrossEntropyLoss要求输入形为：[形状：batch,取值范围：10]
#交叉熵损失函数，适用于多分类问题；输入为原始分数（logits），标签为类别索引
#交叉熵对正确类别的概率取负对数。公式： Loss = -log(p_正确类别:P越接近1越准，越接近0越不准)
#因为分类问题要的是概率分布，交叉熵对「分错但很自信」的情况惩罚极大，梯度也更大、收敛更快。
optimizer=optim.Adam(model.parameters(),lr=ALPHA)#Adam优化器，自动调整学习率；model.parameters()返回模型的可训练参数
#Adam 控制「怎么调参数」，交叉熵告诉 Adam「调得对不对」。两者配合，就是模型训练的核心循环。

#训练循环
def train_one_epoch(model,loader,optimizer,criterion,device):
    model.train()#设置模型为训练模式，启用 dropout 和 batch normalization
    running_loss=0.0
    correct=0
    total=0
    for images,labels in loader:
        images,labels=images.to(device),labels.to(device)#将数据移动到指定设备（CPU 或 GPU）

        outputs=model(images)#前向传播，得到原始分数（logits）
        loss=criterion(outputs,labels)#计算损失，输入为原始分数（logits）和标签索引

        optimizer.zero_grad()#清除之前的梯度
        loss.backward()#反向传播，计算当前损失的梯度
        optimizer.step()#更新模型参数

        #累加损失和准确率
        running_loss+=loss.item()*images.size(0)#累加损失，乘以批量大小
        _,predicted=torch.max(outputs,1)#获取预测类别，torch.max返回最大值和索引，这里取索引
        total+=labels.size(0)#累加总样本数
        correct+=(predicted==labels).sum().item()#累加正确预测的数量

    epoch_loss=running_loss/total#计算平均损失
    epoch_acc=correct/total#计算准确率
    return epoch_loss,epoch_acc

#测试循环
@torch.no_grad()#在测试阶段不需要计算梯度，节省内存和计算资源
def evaluate(model,loader,criterion,device):
    model.eval()#设置模型为评估模式，禁用 dropout 和 batch normalization
    running_loss=0.0
    correct=0
    total=0
    for images,labels in loader:
        images,labels=images.to(device),labels.to(device)

        outputs=model(images)#前向传播，得到原始分数（logits）
        loss=criterion(outputs,labels)#计算损失

        #累加损失和准确率
        running_loss+=loss.item()*images.size(0)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

    test_loss=running_loss/total
    test_acc=correct/total
    return test_loss,test_acc

#训练循环和记录
train_losses,train_accs=[],[]
test_losses,test_accs=[],[]

for epoch in range(EPOCHS):
    train_loss,train_acc=train_one_epoch(model,train_loader,optimizer,criterion,DEVICE)
    test_loss,test_acc=evaluate(model,test_loader,criterion,DEVICE)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f'循环 {epoch+1}/{EPOCHS}, 训练损失: {train_loss:.4f}, 训练准确率: {train_acc*100:.2f}%, 测试损失: {test_loss:.4f}, 测试准确率: {test_acc*100:.2f}%')

#可视化训练过程
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_losses,label='Training Loss')
plt.plot(test_losses,label='Test Loss')
plt.legend()
plt.title('loss curve')

plt.subplot(1,2,2)
plt.plot(train_accs,label='Training Acc')
plt.plot(test_accs,label='Test Acc')
plt.legend()
plt.title('accuracy curve')
plt.show()