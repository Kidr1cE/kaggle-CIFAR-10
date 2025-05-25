import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_loader import load_dataset


# TODO: 将图像移动至显存
def test_data_iterator(data_loader, num_batches=5, visualize=True, 
                       print_batch_info=True, check_labels=True):
    """
    测试数据迭代器的功能和性能
    
    参数:
        data_loader: 数据加载器实例
        num_batches: 测试的批次数量
        visualize: 是否可视化样本
        print_batch_info: 是否打印批次信息
        check_labels: 是否检查标签分布
    """
    # 记录开始时间
    start_time = time.time()
    
    # 初始化标签计数器
    label_counts = {}
    
    # 遍历批次
    for batch_idx, (images, labels) in enumerate(data_loader):
        # 只测试指定数量的批次
        if batch_idx >= num_batches:
            break
        
        # 打印批次信息
        if print_batch_info:
            print(f"\n批次 #{batch_idx+1}")
            print(f"  图像形状: {images.shape}")
            print(f"  标签形状: {labels.shape}")
            print(f"  数据类型: {images.dtype}, {labels.dtype}")
            print(f"  图像范围: [{images.min():.4f}, {images.max():.4f}]")
            
            # 检查标签分布
            if check_labels:
                unique_labels, counts = np.unique(labels.numpy(), return_counts=True)
                for label, count in zip(unique_labels, counts):
                    label_name = data_loader.dataset.dataset.idx_to_class.get(label, f"类别_{label}")
                    if label not in label_counts:
                        label_counts[label] = 0
                    label_counts[label] += count
                    print(f"  标签 {label} ({label_name}): {count} 个样本")
        
        # 可视化样本
        if visualize and batch_idx == 0:  # 只可视化第一个批次
            plt.figure(figsize=(12, 8))
            num_samples = min(8, len(images))  # 最多显示8个样本
            
            for i in range(num_samples):
                img = images[i].permute(1, 2, 0).numpy()  # 调整通道顺序：CxHxW -> HxWxC
                
                # 如果图像已经归一化，恢复到[0,1]范围进行可视化
                if img.min() < 0:
                    img = (img - img.min()) / (img.max() - img.min())
                
                plt.subplot(2, 4, i+1)
                plt.imshow(img)
                
                # 获取标签名称
                label = labels[i].item()
                label_name = data_loader.dataset.dataset.idx_to_class.get(label, f"类别_{label}")
                
                plt.title(f"label: {label} ({label_name})")
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
    
    # 计算并打印性能指标
    elapsed_time = time.time() - start_time
    batches_per_second = num_batches / elapsed_time if elapsed_time > 0 else 0
    samples_per_second = num_batches * len(images) / elapsed_time if elapsed_time > 0 else 0
    
    print(f"\n性能指标:")
    print(f"  处理时间: {elapsed_time:.2f} 秒")
    print(f"  批次/秒: {batches_per_second:.2f}")
    print(f"  样本/秒: {samples_per_second:.2f}")
    
    # 打印总体标签分布
    if check_labels and label_counts:
        print("\n总体标签分布:")
        for label, count in sorted(label_counts.items()):
            label_name = data_loader.dataset.dataset.idx_to_class.get(label, f"类别_{label}")
            print(f"  {label_name} ({label}): {count} 个样本")

def test_load_dataset():
    train_iter, test_iter = load_dataset(32, 0.8)
    # 测试数据迭代器
    test_data_iterator(
        data_loader=train_iter,
        num_batches=3,
        visualize=True,
        print_batch_info=True,
        check_labels=True
    )
    test_data_iterator(
        data_loader=test_iter,
        num_batches=3,
        visualize=True,
        print_batch_info=True,
        check_labels=True
    )

"""
性能指标:
  处理时间: 1.72 秒
  批次/秒: 1.75
  样本/秒: 55.88
"""

if __name__ == "__main__":
    test_load_dataset()
