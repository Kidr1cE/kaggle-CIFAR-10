import data_loader
import utils.utils as utils

from model.resnet import resnet18
from train import train

batch_size, epoch_num, lr = 32, 30, 0.1

if __name__ == "__main__":
    train_iter, test_iter = data_loader.load_dataset(batch_size, 0.8, 114514)

    net = resnet18(10, 3)
    train(net, train_iter, test_iter, epoch_num, lr, utils.try_gpu())
