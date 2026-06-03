import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.onnx

#------------------------------CIFAR10——卷积神经网络CNN--------------------------------------
#设置超参数和设备
BATCH_SIZE=256
EPOCHS=30
ALPHA=0.001
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#加载数据
# 1. 先用 ToTensor 加载，计算 mean 和 std
temp_transform=transforms.Compose([
    transforms.ToTensor()
])
train_data=datasets.CIFAR10(root='D:\Pytorch\data',train=True,download=False,transform=temp_transform)
all_images=torch.stack([img for img,_ in train_data])# 三通道，通道分开计算[50000, 3, 32, 32]
mean=all_images.mean(dim=(0,2,3))# 计算每个通道的均值，保持维度[3]
std=all_images.std(dim=(0,2,3))# 计算每个通道的方差，保持维度[3]
print(f'Mean: {mean.tolist()}, Std: {std.tolist()}')
# 2. 用带 Normalize 的完整 transform 重新加载
transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),#数据增强，随机水平翻转图像，增加训练数据的多样性，减少过拟合；在训练时以0.5的概率翻转图像，在测试时不翻转
    transforms.RandomCrop(32,padding=4),#数据增强，随机裁剪图像，增加训练数据的多样性，减少过拟合；在训练时随机裁剪图像，在测试时中心裁剪
    #上面两行为数据增强，增加训练数据的多样性，减少过拟合；在训练时随机水平翻转和裁剪图像
    transforms.ToTensor(),#将PIL图像或NumPy数组转换为PyTorch张量，并将像素值缩放到[0,1]范围内；输入图像形状为[H,W,C]，输出张量形状为[C,H,W]
    transforms.Normalize(mean.tolist(), std.tolist())#使用计算得到的均值和方差对每个通道进行标准化output[channel] = (input[channel] - mean[channel]) / std[channel]
])
test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean.tolist(), std.tolist())
])
train_dataset=datasets.CIFAR10(root='D:\Pytorch\data',train=True,download=False,transform=transform)
#测试集不能用 RandomHorizontalFlip 和 RandomCrop，因为测试时需要保持图像的一致性，不能引入随机变换；测试集只使用 ToTensor 和 Normalize 进行预处理
test_dataset=datasets.CIFAR10(root='D:\Pytorch\data',train=False,download=False,transform=test_transform)
train_loader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

#卷积神经网络CNN定义
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #卷积层1：输入通道3，输出通道32，卷积核大小3x3，padding=1保持输入输出尺寸不变
        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(32)#批归一化层，标准化卷积层的输出，稳定训练过程，加速收敛；输入输出通道数相同
        self.relu1=nn.ReLU()#激活函数
        self.pool1=nn.MaxPool2d(2)#池化层，2x2的窗口，步长为2，将特征图尺寸减半
        #卷积层2：输入通道32，输出通道64，卷积核大小3x3，padding=1保持输入输出尺寸不变
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(64)#批归一化层，标准化卷积层的输出，稳定训练过程，加速收敛；输入输出通道数相同
        self.relu2=nn.ReLU()#激活函数
        self.pool2=nn.MaxPool2d(2)#池化层，2x2的窗口，步长为2，将特征图尺寸减半
        #卷积层3：输入通道64，输出通道128，卷积核大小3x3，padding=1保持输入输出尺寸不变
        self.conv3=nn.Conv2d(64,256,kernel_size=3,padding=1)
        self.bn3=nn.BatchNorm2d(256)
        self.relu3=nn.ReLU()#激活函数
        self.conv4=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.bn4=nn.BatchNorm2d(256)
        self.pool3=nn.MaxPool2d(2)#conv3和conv4共用池化层，2x2的窗口，步长为2，将特征图尺寸减半
        self.conv1x1=nn.Conv2d(256,256,kernel_size=1)#1x1卷积层，调整通道数，保持空间尺寸不变；输入通道数为256，输出通道数为256
        self.relu1x1=nn.ReLU()#激活函数
        self.flatten=nn.Flatten()#展平层，将多维特征图展平为一维向量
        self.fc1=nn.Linear(256*4*4,256)#全连接层：256*4*4 → 256；全连接层（线性层），公式为 y = xW^T + b
        self.relu4=nn.ReLU()#激活函数
        self.fc2=nn.Linear(256,10)#全连接层：256 → 10(输出层的10个类别)
    #前向传播
    def forward(self,x):
        x=self.pool1(self.relu1(self.bn1(self.conv1(x))))#卷积层1 + 激活 + 池化
        x=self.pool2(self.relu2(self.bn2(self.conv2(x))))#卷积层2 + 激活 + 池化
        #x=self.pool3(self.relu3(self.bn3(self.conv3(x))))#卷积层3 + 激活 + 池化
        x=self.relu4(self.bn4(self.conv4(self.relu3(self.bn3(self.conv3(x))))))#卷积层4 + 激活
        x=self.pool3(x)#卷积层4 + 激活 + 池化;尺寸链应该为：conv3→conv4→bn4→relu4→pool3
        x=self.relu1x1(self.conv1x1(x))#1x1卷积层 + 激活
        x=self.flatten(x)#展平
        x=self.relu4(self.fc1(x))#全连接层1 + 激活
        x=self.fc2(x)#全连接层2（输出层）
        #x=nn.Dropout(0.01)(x) # Dropout层，随机丢弃部分神经元，防止过拟合；在训练时以0.5的概率丢弃神经元，在测试时不丢弃
        return x
model=CNN().to(DEVICE)

#损失函数与优化器
criterion=nn.CrossEntropyLoss()#.CrossEntropyLoss要求输入形为：[形状：batch,取值范围：10]
optimizer=optim.AdamW(model.parameters(),lr=ALPHA,weight_decay=1e-4)#Adam优化器，结合了动量和RMSProp的优点，适用于大多数情况

#训练循环
def train(model,train_loader,criterion,optimizer,DEVICE):
    model.train()
    running_loss=0.0
    total=0
    correct=0

    for images,labels in train_loader:
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
def evaluate(model,test_loader,criterion,DEVICE):
    model.eval()
    running_loss=0.0
    total=0
    correct=0

    for images,labels in test_loader:
        images,labels=images.to(DEVICE),labels.to(DEVICE)
        outputs=model(images)
        loss=criterion(outputs,labels)

        #累加损失和准确率
        running_loss+=loss.item()*images.size(0)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
    
    test_loss=running_loss/total
    test_acc=correct/total
    return test_loss,test_acc

#训练和评估模型
train_losses,train_accs=[],[]
test_losses,test_accs=[],[]
for epoch in range(EPOCHS):
    train_loss,train_acc=train(model,train_loader,criterion,optimizer,DEVICE)
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
torch.onnx.export(model, dummy_input, "model.onnx")