from torchvision.datasets import FashionMNIST
import numpy as np
import torch,torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
train_data=FashionMNIST("./dataset_MNIST",
                        train=True,
                        transform=torchvision.transforms.Compose([torchvision.transforms.Resize(size=224),torchvision.transforms.ToTensor()]),
                        download=True)
train_loader = DataLoader(train_data,
                          batch_size=64,
                          shuffle=True,
                          num_workers=0)
##获取一个batch的数据
for step,(b_x,b_y) in enumerate(train_loader):
    if step>0:
        break
batch_x=b_x.squeeze().numpy()
batch_y=b_y.numpy()
class_label=train_data.classes
plt.figure(figsize=(12,5))
for i in np.arange(len(batch_y)):
    plt.subplot(4,16,i+1)
    plt.imshow(batch_x[i,:,:],cmap=plt.cm.gray)
    plt.title(class_label[batch_y[i]],size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()

