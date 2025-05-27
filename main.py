import torch
import data_loader
import predict
import utils.utils as utils

from model.resnet import resnet18
from configs import config
from train import train

if __name__ == "__main__":
        # 读取设置
    conf = config.load_config()
    data_loader_conf = conf['data_loader']
    train_conf = conf['training']

    # 数据迭代器
    train_iter, test_iter = data_loader.load_dataset(
        data_loader_conf['batch_size'],
        data_loader_conf['train_ratio'],
        data_loader_conf['seed'])

    # 获取网络
    net = resnet18(10, 3)

    # 训练
    train(net, train_iter, test_iter, train_conf, utils.try_gpu())
    torch.save(net.state_dict(), train_conf['model_path'])

    predict.predict(net, data_loader_conf['batch_size'], utils.try_gpu())
