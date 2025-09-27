import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import cv2  # 导入 OpenCV
import numpy as np  # 导入 Numpy


# --- 1. 修改后的网络定义 ---
# 唯一的改动在 self.conv1，将 nn.Conv2d(1, ...) 改为 nn.Conv2d(3, ...)
# 目的是为了接收来自摄像头的 3 通道彩色图像。
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 将输入通道 in_channels 从 1 改为 3
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
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
        output = F.log_softmax(x, dim=1)
        return output


# --- 原始的训练和测试函数 (有微小调整以适应新模型) ---
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 尽管模型现在能接收 3 通道图像，我们仍然在单通道的 MNIST 数据集上训练。
        # 因此，我们需要将单通道数据复制三次来匹配模型的输入维度。
        data = data.repeat(1, 3, 1, 1)  # 复制通道维度
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('训练轮次: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.repeat(1, 3, 1, 1)  # 测试时同样需要复制通道
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\n测试集: 平均损失: {:.4f}, 准确率: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# --- 2. 用于实时摄像头测试的新函数 ---
def realtime_test(model, device):
    """
    从摄像头捕获视频，处理每一帧图像以寻找数字，
    并使用训练好的模型进行实时预测。
    """
    # 定义与训练时相同的图像变换，用于处理单张图像
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 启动摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头。")
        return

    model.eval()  # 将模型设置为评估模式

    while True:
        # 逐帧捕获
        ret, frame = cap.read()
        if not ret:
            break

        # 水平翻转图像，使其看起来像镜子，更直观
        frame = cv2.flip(frame, 1)

        # 复制一份原始帧，用于最后绘制结果
        display_frame = frame.copy()

        # --- 针对模型的图像预处理流程 ---
        # 1. 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. 使用高斯模糊和阈值处理来获得二值图像
        # 这有助于将数字从背景中分离出来
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # 使用反向二值化（THRESH_BINARY_INV），使数字部分变为白色(255)，背景为黑色(0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

        # 3. 寻找图像中的轮廓
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 找到面积最大的轮廓，我们假设它就是我们写的数字
            largest_contour = max(contours, key=cv2.contourArea)

            # 获取这个最大轮廓的边界框 (x, y, 宽度, 高度)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # 为边界框添加一些内边距 (padding)
            padding = 15
            x, y, w, h = x - padding, y - padding, w + 2 * padding, h + 2 * padding

            # 确保边界框不会超出画面范围
            x, y = max(0, x), max(0, y)

            # 从二值图像中提取出数字所在的区域 (Region of Interest, ROI)
            digit_roi = thresh[y:y + h, x:x + w]

            if digit_roi.size > 0:
                # 4. 将ROI缩放到28x28像素，这是MNIST模型的标准输入尺寸
                resized_digit = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)

                # 5. 将处理后的图像（Numpy数组）转换为PyTorch张量
                tensor_img = transform(resized_digit)

                # 6. 模型需要3通道输入，所以我们将单通道张量复制三次
                tensor_img = tensor_img.repeat(3, 1, 1)

                # 7. 添加一个批次维度（模型期望的输入是[批次数, 通道, 高, 宽]）
                # 最终形状应为 [1, 3, 28, 28]
                tensor_img = tensor_img.unsqueeze(0).to(device)

                # 8. 使用模型进行预测
                with torch.no_grad():
                    output = model(tensor_img)
                    pred = output.argmax(dim=1, keepdim=True).item()

                # 在用于显示的图像上绘制矩形框和预测结果
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_frame, f'Prediction: {pred}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 显示最终处理后的画面
        cv2.imshow('Camera_torch - Press Q to exit', display_frame)

        # 如果按下 'q' 键，则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()


def main():
    # 训练设置
    parser = argparse.ArgumentParser(description='PyTorch MNIST 示例')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='训练时的批次大小 (默认: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='测试时的批次大小 (默认: 1000)')
    # 增加了默认的训练轮次以获得更好的准确率
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='训练的总轮次 (默认: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='学习率 (默认: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='学习率步进 gamma (默认: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='禁用 CUDA 训练')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='快速检查单个批次的训练过程')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='随机种子 (默认: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='每隔多少批次记录一次训练状态')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='保存当前训练好的模型')
    # --- 3. 为摄像头模式添加的新参数 ---
    parser.add_argument('--use-camera', action='store_true', default=False,
                        help='使用摄像头进行实时测试，而不是使用MNIST测试集')
    parser.add_argument('--model-path', type=str, default="mnist_cnn.pt",
                        help='用于摄像头测试的已保存模型的路径')

    args = parser.parse_args()

    # 设置设备 (优先使用CUDA)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    model = Net().to(device)

    # --- 4. 主逻辑分支：选择摄像头模式或训练/测试模式 ---
    if args.use_camera:
        try:
            # 加载已训练的模型
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"模型已从 {args.model_path} 加载")
            # 启动实时测试函数
            realtime_test(model, device)
        except FileNotFoundError:
            print(f"错误: 在 '{args.model_path}' 未找到模型文件。")
            print("请先不带 '--use-camera' 参数运行脚本来训练并保存一个模型。")
        except Exception as e:
            print(f"发生错误: {e}")

    else:  # 原始的训练和测试逻辑
        train_kwargs = {'batch_size': args.batch_size}
        test_kwargs = {'batch_size': args.test_batch_size}
        if use_cuda:
            cuda_kwargs = {'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True}
            train_kwargs.update(cuda_kwargs)
            test_kwargs.update(cuda_kwargs)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset1 = datasets.MNIST('../data', train=True, download=True,
                                  transform=transform)
        dataset2 = datasets.MNIST('../data', train=False,
                                  transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")
            print("模型已保存至 mnist_cnn.pt")


if __name__ == '__main__':
    main()
