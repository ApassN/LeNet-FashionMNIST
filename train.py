import time
import copy
import os
from torchvision.datasets import FashionMNIST
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import LeNet
import torch.nn as nn
import pandas as pd

## 处理训练集和验证集的代码 二八划分
def train_val_data_process():
    dataset = FashionMNIST(
        "./dataset_MNIST",
        train=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]),
        download=True
    )
    train_data, val_data = torch.utils.data.random_split(
        dataset, 
        [round(0.8*len(dataset)), round(0.2*len(dataset))]
    )
    
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=64,
        shuffle=True,
        num_workers=0  # Windows系统必须设为0，Linux/Mac可以设为4-8
    )
    
    val_dataloader = DataLoader(
        val_data,
        batch_size=64,
        shuffle=False,  # 验证集不需要打乱
        num_workers=0
    )
    
    return train_dataloader, val_dataloader

def train_model_process(model, train_dataloader, val_dataloader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    ## 使用adam优化器 学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ## 损失函数为交叉熵函数
    loss_fcn = nn.CrossEntropyLoss()
    model = model.to(device)
    
    # 初始化最优模型权重
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 初始化记录列表
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []
    
    # 记录开始时间
    since = time.time()

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")
        print("-" * 10)
        
        # 初始化当前epoch的统计变量
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0
        
        # --------------------------
        # 训练阶段
        # --------------------------
        model.train()  # 设置为训练模式
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = loss_fcn(output, b_y)
            
            # 反向传播与参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累计损失和正确数
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)
        
        # 计算训练集epoch损失和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        
        # --------------------------
        # 验证阶段
        # --------------------------
        model.eval()  # 设置为评估模式
        with torch.no_grad():  # 关闭梯度计算，节省显存
            for step, (b_x, b_y) in enumerate(val_dataloader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                
                output = model(b_x)
                pre_lab = torch.argmax(output, dim=1)
                loss = loss_fcn(output, b_y)
                
                # 累计损失和正确数
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)
        
        # 计算验证集epoch损失和准确率
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        
        # 打印当前epoch结果
        print(f"训练集：损失 {train_loss_all[-1]:.4f}，准确率 {train_acc_all[-1]:.4f}")
        print(f"验证集：损失 {val_loss_all[-1]:.4f}，准确率 {val_acc_all[-1]:.4f}")
        
        # 更新最优模型
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"发现新的最优模型，验证集准确率: {best_acc:.4f}")
        
        # 计算并打印已用时间
        time_use = time.time() - since
        print(f"已用时间: {time_use//60:.0f}m {time_use%60:.0f}s\n")

    # --------------------------
    # 训练结束
    # --------------------------
    time_use = time.time() - since
    print(f"总耗时: {time_use//60:.0f}m {time_use%60:.0f}s")
    print(f"最高验证集准确率: {best_acc:.4f}")
    
    # 加载最优模型权重
    model.load_state_dict(best_model_weights)
    
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(best_model_weights, "LeNet/outputs/best_model.pth")
    print("最优模型已保存到: LeNet/outputs/best_model.pth")
    train_process=pd.DataFrame(data={"epoch":range(epochs),
                                     "train_loss_all":train_loss_all,
                                     "val_loss_all":val_loss_all,
                                     "train_acc_all":train_acc_all,
                                     "val_acc_all":val_acc_all})
    return train_process
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))  # 修正这里
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process.train_loss_all, 'ro-', label="train loss")
    plt.plot(train_process["epoch"], train_process.val_loss_all, 'bs-', label="val loss")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bs-', label="val acc")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("acc")
    
    plt.tight_layout()  # 加上这一行，防止标签重叠
    plt.savefig("LeNet/outputs/training_curve.png")
    plt.show()

##将模型实例化
leNet=LeNet()
train_dataloader,val_dataloader=train_val_data_process()
train_process=train_model_process(leNet,train_dataloader,val_dataloader,20)
matplot_acc_loss(train_process)