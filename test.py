import torch
import torchvision
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torch.nn as nn
from model import LeNet


# ==================================================
# FashionMNIST 的 10 个类别名称
# ==================================================
class_names = [
    "T-shirt/top",  # 0：T恤 / 上衣
    "Trouser",      # 1：裤子
    "Pullover",     # 2：套头衫
    "Dress",        # 3：连衣裙
    "Coat",         # 4：外套
    "Sandal",       # 5：凉鞋
    "Shirt",        # 6：衬衫
    "Sneaker",      # 7：运动鞋
    "Bag",          # 8：包
    "Ankle boot"    # 9：短靴
]


# ==================================================
# 处理测试集数据
# ==================================================
def test_data_process():
    """
    加载 FashionMNIST 官方测试集。

    注意：
    train=False 表示使用官方测试集。
    测试集只用于最终模型性能评估，不参与训练，也不参与调参。
    """

    test_dataset = FashionMNIST(
        "./dataset_MNIST",
        train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ]),
        download=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,   # 测试集不需要打乱
        num_workers=0    # Windows 系统建议设为 0
    )

    return test_dataloader


# ==================================================
# 测试模型性能
# ==================================================
def test_model_process(model, test_dataloader):
    """
    在测试集上评估模型性能。

    评估指标：
    1. 测试集平均损失 test_loss
    2. 测试集准确率 test_acc
    3. 每个类别的分类准确率
    """

    # 选择设备：优先使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 将模型放到对应设备上
    model = model.to(device)

    # 加载训练阶段保存的最优模型参数
    model.load_state_dict(torch.load("LeNet/outputs/best_model.pth", map_location=device))
    print("已加载模型参数: LeNet/outputs/best_model.pth")

    # 交叉熵损失函数
    loss_fcn = nn.CrossEntropyLoss()

    # 设置模型为评估模式
    # eval() 会关闭 Dropout、BatchNorm 等训练行为
    model.eval()

    # 初始化整体统计变量
    test_loss = 0.0
    test_corrects = 0
    test_num = 0

    # 初始化每个类别的统计变量
    # class_correct[i] 表示第 i 类预测正确的数量
    # class_total[i] 表示第 i 类样本总数量
    class_correct = [0 for _ in range(10)]
    class_total = [0 for _ in range(10)]

    # 测试阶段不需要计算梯度，可以节省显存和加快速度
    with torch.no_grad():
        for step, (b_x, b_y) in enumerate(test_dataloader):
            # 将数据放到对应设备上
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 前向传播，得到模型输出
            output = model(b_x)

            # 计算损失
            loss = loss_fcn(output, b_y)

            # 取每一行最大值对应的类别作为预测类别
            pre_lab = torch.argmax(output, dim=1)

            # 累计整体损失
            test_loss += loss.item() * b_x.size(0)

            # 累计整体预测正确的数量
            test_corrects += torch.sum(pre_lab == b_y.data)

            # 累计测试样本数量
            test_num += b_x.size(0)

            # 统计每个类别的准确率
            for i in range(b_y.size(0)):
                label = b_y[i].item()
                pred = pre_lab[i].item()

                class_total[label] += 1

                if pred == label:
                    class_correct[label] += 1

    # 计算测试集平均损失和准确率
    test_loss_avg = test_loss / test_num
    test_acc = test_corrects.double().item() / test_num

    # ==================================================
    # 打印整体测试结果
    # ==================================================
    print("-" * 40)
    print("测试集整体结果")
    print("-" * 40)
    print(f"测试集平均损失: {test_loss_avg:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    # ==================================================
    # 打印每个类别的测试结果
    # ==================================================
    print("\n每个类别的准确率")
    print("-" * 40)

    for i in range(10):
        if class_total[i] == 0:
            acc = 0.0
        else:
            acc = class_correct[i] / class_total[i]

        print(
            f"{i}: {class_names[i]:12s} "
            f"准确率: {acc:.4f} "
            f"({class_correct[i]}/{class_total[i]})"
        )

    return test_loss_avg, test_acc


# ==================================================
# 主函数入口
# ==================================================
if __name__ == "__main__":
    # 实例化模型
    leNet = LeNet()

    # 加载测试集
    test_dataloader = test_data_process()

    # 测试模型
    test_loss, test_acc = test_model_process(leNet, test_dataloader)