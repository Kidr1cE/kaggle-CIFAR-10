import torch
import data_loader
import utils.utils as utils
from torch import nn
from model import resnet
from utils.visualizer import TrainingVisualizer

def train(net, train_iter, test_iter, num_epoch, lr, wd, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    loss = nn.CrossEntropyLoss()
    viz = TrainingVisualizer(title="CIFAR10")

    for epoch in range(num_epoch):
        # 训练阶段
        net.train()
        train_loss = 0.0    # 训练损失
        correct = 0         # 预测正确数量
        total = 0           # 样本总数
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)

            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            # 累加训练损失和正确预测的样本数
            train_loss += l.item() * X.size(0)      # 累积总损失
            _, predicted = y_hat.max(1)             # 预测结果
            total += y.size(0)
            correct += predicted.eq(y).sum().item() # 预测正确数量

        train_accuracy = 100.0 * correct / total    # 训练正确率
        train_loss = train_loss / total             # 训练损失

        # 验证阶段
        net.eval()
        val_correct = 0 # 验证正确数
        val_total = 0   # 验证总数
        with torch.no_grad():
            for X, y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)

                # 计算正确率
                _, predicted = y_hat.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()
        
        # 计算验证准确率
        val_accuracy = 100.0 * val_correct / val_total

        # 更新可视化图表
        viz.update(epoch+1, train_accuracy, val_accuracy, train_loss)

    # 保存最终结果
    viz.save_figure('final_training_result.png')

    return

lr, wd = 2e-4, 5e-4
batch_size, epoch_num = 32, 30

save_path = "./data/resnet18.pth"

if __name__ == "__main__":
    train_iter, test_iter = data_loader.load_dataset(batch_size, 0.8, 114514)

    net = resnet.resnet18(10, 3)
    train(net, train_iter, test_iter, epoch_num, lr, wd, utils.try_gpu())
    torch.save(net.state_dict(), save_path)
