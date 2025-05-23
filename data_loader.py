import pandas
import torch
import numpy
import torchvision
import test_data_loader

from PIL import Image
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader, Dataset

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

    def __len__(self):
        return len(self.labels_info)
    
    def __getitem__(self, index):
        id = self.labels_info['id'][index]
        label = self.labels_info['label'][index]
        image = self.transform(Image.open(self.path + f"train/{id}.png"))

        return image, self.class_to_idx[label]

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
    print(len(train_dataset), len(test_dataset))

    return train_dataset, test_dataset

def load_dataset(batch_size, train_ratio, seed):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    dataset = CIFARDataset(data_path, transform)
    train, test = split_cifar_dataset(dataset, train_ratio, seed)
    train_iter = DataLoader(train, batch_size, num_workers=0)
    test_iter = DataLoader(test, batch_size, num_workers=0)
    return train_iter, test_iter

def test_load_dataset():
    train_iter, test_iter = load_dataset(32, 0.8)
    test_data_loader.test_data_iterator(
        data_loader=train_iter,
        num_batches=3,
        visualize=True,
        print_batch_info=True,
        check_labels=True
    )
    test_data_loader.test_data_iterator(
        data_loader=test_iter,
        num_batches=3,
        visualize=True,
        print_batch_info=True,
        check_labels=True
    )

if __name__ == "__main__":
    test_load_dataset()
