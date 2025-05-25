
import os
import pandas
import torch
import numpy
import torchvision

from PIL import Image
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader, Dataset

"""
dataset
    image size: 32*32
    length: 50000
    label: trainLabels.csv
"""

batch_size = 32
data_path = "../data/cifar-10/"  # 50000

class CIFARDataset(Dataset):
    def __init__(self, path, transform):
        """
        Args:
            path: 数据集路径
            transform: 图像转换
        """
        self.path = path
        self.transform = transform
    
        # 获取 labels标签
        self.labels_info = pandas.read_csv(path+"trainLabels.csv")   # labels文件

        self.labels = sorted(list(set(self.labels_info['label'])))
        self.n_classes = len(self.labels)

        self.class_to_idx = dict(zip(self.labels, range(self.n_classes)))   # labels与id映射字典
        self.idx_to_class = {v : k for k,v in self.class_to_idx.items()}
        print(self.class_to_idx)                                            # TODO: log打印日志

    def __len__(self):
        return len(self.labels_info)

    def __getitem__(self, index):
        id = self.labels_info['id'][index]
        label = self.labels_info['label'][index]
        image = self.transform(Image.open(self.path + f"train/{id}.png"))

        return image, self.class_to_idx[label]
    
class CIFARTestDataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.transform = transform

        self.file_num = len(os.listdir(path))

    def __len__(self):
        return len(self.labels_info)

    def __getitem__(self, index):
        image = self.transform(Image.open(self.path + f"test/{index}.png"))
        return image


def split_cifar_dataset(dataset, train_ratio=0.8, 
                        random_seed=None, shuffle=True):
    if random_seed is not None:
        numpy.random.seed(random_seed)
        torch.manual_seed(random_seed)

    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)

    # 子集数据索引
    indices = list(range(dataset_size))
    if shuffle:
        numpy.random.shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # 根据数据索引分割数据集
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    print(f"train_dataset_len: {len(train_dataset)}, test_dataset_len: {len(test_dataset)}") # TODO: log打印日志

    return train_dataset, test_dataset

def load_dataset(batch_size, train_ratio, seed=None):
    transform = torchvision.transforms.Compose([
        # 在高度和宽度上将图像放大到40像素的正方形
        torchvision.transforms.Resize(40),
        # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
        # 生成一个面积为原始图像面积0.64～1倍的小正方形，
        # 然后将其缩放为高度和宽度均为32像素的正方形
        torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # 标准化图像的每个通道
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
    dataset = CIFARDataset(data_path, transform)
    train, test = split_cifar_dataset(dataset, train_ratio, seed)
    train_iter = DataLoader(train, batch_size, num_workers=0, pin_memory=True)
    test_iter = DataLoader(test, batch_size, num_workers=0, pin_memory=True)
    return train_iter, test_iter

def get_test_data(batch_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(40),
        torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                   ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
    dataset = CIFARTestDataset(data_path, transform)
    data_iter = DataLoader(dataset, batch_size, num_workers=0, pin_memory=True)
    
    return data_iter
