import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#------------------------------卷积神经网络CNN--------------------------------------
#设置超参数和设备
BATCH_SIZE=128
EPOCHS=20
ALPHA=0.001
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#加载数据
# 1. 先用 ToTensor 加载，计算 mean 和 std
temp_transform=transforms.Compose([
    transforms.ToTensor()
])
train_dataset=datasets.MNIST(
    root='D:\Pytorch\data',train=True,download=True,transform=temp_transform
)
all_imgs=torch.stack([img for img,_ in train_dataset])
mean=all_imgs.mean().item()
std=all_imgs.std().item()
print(f'Mean: {mean:.4f}, Std: {std:.4f}')
# 2. 用带 Normalize 的完整 transform 重新加载
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,),(std,))
])

#读取文件
train_dataset=datasets.MNIST(
    root='D:\Pytorch\data',train=True,download=False,transform=transform
)
test_dataset=datasets.MNIST(
    root='D:\Pytorch\data',train=False,download=False,transform=transform
)
train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

#卷积神经网络CNN定义
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #对于MNIST数据集，2层卷积层和2层全连接层的结构已经足够，过多的层可能导致过拟合
        #卷积层1：输入通道1，输出通道32，卷积核大小3x3，padding=1保持输入输出尺寸不变
        self.conv1=nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.relu1=nn.ReLU()#激活函数
        self.pool1=nn.MaxPool2d(2)#池化层，2x2的窗口，步长为2，将特征图尺寸减半
        #卷积层2：输入通道32，输出通道64，卷积核大小3x3，padding=1保持输入输出尺寸不变
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.relu2=nn.ReLU()#激活函数
        self.pool2=nn.MaxPool2d(2)#池化层，2x2的窗口，步长为2，将特征图尺寸减半
        #卷积层3：输入通道64，输出通道128，卷积核大小3x3，padding=1保持输入输出尺寸不变
        self.conv3=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.relu3=nn.ReLU()#激活函数
        self.pool3=nn.MaxPool2d(2)#池化层，2x2的窗口，步长为2，将特征图尺寸减半
        #卷积层4：输入通道128，输出通道256，卷积核大小3x3，padding=1保持输入输出尺寸不变
        self.conv4=nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.relu4=nn.ReLU()#激活函数
        self.pool4=nn.MaxPool2d(2)#池化层，2x2的窗口，步长为2，将特征图尺寸减半
        self.flatten=nn.Flatten()#展平层，将多维特征图展平为一维向量
        self.fc1=nn.Linear(256*1*1,128)#全连接层：输入256*1*1，输出128
        self.relu5=nn.ReLU()#激活函数
        self.out=nn.Linear(128,10)#全连接层：输入128，输出10（类别数）

    def forward(self,x):
        x=self.pool1(self.relu1(self.conv1(x)))#卷积层1 → 激活函数 → 池化层1
        x=self.pool2(self.relu2(self.conv2(x)))#卷积层2 → 激活函数 → 池化层2
        x=self.pool3(self.relu3(self.conv3(x)))#卷积层3 → 激活函数 → 池化层3
        x=self.pool4(self.relu4(self.conv4(x)))#卷积层4 → 激活函数 → 池化层4
        x=self.flatten(x)#展平层
        x=self.relu5(self.fc1(x))#全连接层1 → 激活函数
        x=self.out(x)#输出层
        return x#原始分数（logits）不应用 softmax，因为 CrossEntropyLoss 内部的LogSoftmax + NLLLoss会处理
model=CNN().to(DEVICE)

#损失函数与优化器
criterion=nn.CrossEntropyLoss()#.CrossEntropyLoss要求输入形为：[形状：batch,取值范围：10]
optimizer=optim.Adam(model.parameters(),lr=ALPHA)#Adam优化器，适用于大多数情况，自动调整学习率

#训练循环
def train_one_epoch(model,loader,optimizer,criterion,DEVICE):
    model.train()
    running_loss=0.0
    total=0
    correct=0

    for images,labels in loader:
        images,labels=images.to(DEVICE),labels.to(DEVICE)
        outputs=model(images)
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #累加损失和准确率
        running_loss+=loss.item()*images.size(0)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()

    epoch_loss=running_loss/total
    epoch_acc=correct/total
    return epoch_loss,epoch_acc

#测试循环
@torch.no_grad()#在测试阶段不需要计算梯度，节省内存和计算资源
def evaluate(model,loader,criterion,DEVICE):
    model.eval()
    running_loss=0.0
    total=0
    correct=0

    for images,labels in loader:
        images,labels=images.to(DEVICE),labels.to(DEVICE)
        outputs=model(images)
        loss=criterion(outputs,labels)

        running_loss+=loss.item()*images.size(0)
        _,predicted=torch.max(outputs,1)
        correct+=(predicted==labels).sum().item()
        total+=labels.size(0)
    
    test_loss=running_loss/total
    test_acc=correct/total
    return test_loss,test_acc

#训练和评估模型
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
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses,label='train loss')
plt.plot(test_losses,label='test loss')
plt.legend()
plt.title('epochs curves')

plt.subplot(1,2,2)
plt.plot(train_accs,label='train acc')
plt.plot(test_accs,label='test acc')
plt.legend()
plt.title('accuracy curves')
plt.show()