import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import cv2
import numpy as np
import os


# 这是我们之前定义的网络结构，无需改变
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # 输入通道为3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # ImageFolder 加载的是灰度图，需要复制通道以匹配模型
        data = data.repeat(1, 3, 1, 1)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    print(f"训练轮次 {epoch} 完成。")


# 验证函数
def validate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.repeat(1, 3, 1, 1)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'验证集: 平均损失: {test_loss:.4f}, 准确率: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')


# 实时检测函数
def realtime_test(model, device):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    cap = cv2.VideoCapture(0)
    model.eval()
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)[1]
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            roi = thresh[y:y + h, x:x + w]
            if roi.size > 0:
                resized_digit = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                tensor_img = transform(resized_digit).repeat(3, 1, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(tensor_img)
                    pred = output.argmax(dim=1).item()
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f'Prediction: {pred}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)
        cv2.imshow('实时数字识别 - 按 Q 退出', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()


def main():
    # --- 训练设置 ---
    parser = argparse.ArgumentParser(description='用自定义数据训练并测试PyTorch模型')
    parser.add_argument('--data-path', type=str, default='my_digits', help='自定义数据集的路径')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮次 (默认: 20)')
    parser.add_argument('--lr', type=float, default=1.0, help='学习率 (默认: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, help='学习率衰减率 (默认: 0.7)')
    parser.add_argument('--save-model', action='store_true', default=True, help='保存最终模型')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # --- 1. 加载您自己的数据集 ---
    if not os.path.exists(args.data_path):
        print(f"错误: 数据集文件夹 '{args.data_path}' 不存在。")
        print("请先运行 collect_data.py 来创建您的数据集。")
        return

    # 定义图像变换
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 确保是单通道
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 使用MNIST的均值和标准差
    ])

    # 使用 ImageFolder 加载数据集
    full_dataset = datasets.ImageFolder(root=args.data_path, transform=transform)

    # 划分训练集和验证集 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print(f"数据集加载完毕：总样本数 {len(full_dataset)}, 训练样本 {len(train_dataset)}, 验证样本 {len(test_dataset)}")

    # --- 2. 训练模型 ---
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    print("\n--- 开始训练 ---")
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        validate(model, device, test_loader)
        scheduler.step()
    print("--- 训练完成 ---\n")

    if args.save_model:
        torch.save(model.state_dict(), "my_digits_cnn.pt")
        print("模型已保存为 my_digits_cnn.pt")

    # --- 3. 启动实时检测 ---
    input("按回车键启动实时摄像头检测...")
    realtime_test(model, device)


if __name__ == '__main__':
    main()
