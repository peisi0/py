import argparse

parser = argparse.ArgumentParser(description='PyTorch 训练脚本')
parser.add_argument('--lr', type=float, default=0.01, help='学习率')
parser.add_argument('--epochs', type=int, default=10, help='训练周期数')

args = parser.parse_args()

print(f"学习率: {args.lr}")
print(f"训练周期数: {args.epochs}")
