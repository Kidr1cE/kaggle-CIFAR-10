import os
import matplotlib.pyplot as plt
import numpy as np

class TrainingVisualizer:
    def __init__(self, title="visualizer", figsize=(10, 6), log_dir="logs"):
        """
        初始化训练可视化工具
        
        参数:
            title: 图表标题
            figsize: 图表尺寸
        """
        # 确保中文显示正常
        plt.rcParams["font.family"] = ["SimHei"]
        self.log_dir = log_dir
        self._ensure_log_dir_exists()

        # 创建图表和双Y坐标轴
        self.fig, self.ax1 = plt.subplots(figsize=figsize)
        self.ax2 = self.ax1.twinx()


        # 设置图表标题和坐标轴标签
        self.fig.suptitle(title, fontsize=16)
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('准确率 (%)', color='blue')
        self.ax2.set_ylabel('损失值', color='red')
        
        # 初始化数据列表
        self.epochs = []
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        
        # 初始化线条和标注对象
        self.lines = []
        self.annotations = []
        
        # 设置图表样式
        self.ax1.grid(True, linestyle='--', alpha=0.7)
        
 
        # 显示图表但不阻塞
        plt.ion()
        plt.show()
    
    def update(self, epoch, train_accuracy, val_accuracy, train_loss):
        """
        更新图表数据
        
        参数:
            epoch: 当前轮次
            train_accuracy: 训练准确率 (0-100)
            val_accuracy: 验证准确率 (0-100)
            train_loss: 训练损失值
        """
        # 添加新数据点
        self.epochs.append(epoch)
        self.train_acc.append(train_accuracy)
        self.val_acc.append(val_accuracy)
        self.train_loss.append(train_loss)
        
        # 清除现有线条和标注
        for line in self.lines:
            line.remove()
        self.lines = []
        
        for annotation in self.annotations:
            annotation.remove()
        self.annotations = []
        
        # 绘制训练准确率曲线 (左侧Y轴)
        line1, = self.ax1.plot(self.epochs, self.train_acc, 'b-o', label='训练准确率', markersize=4)
        self.lines.append(line1)
        
        # 绘制验证准确率曲线 (左侧Y轴)
        line2, = self.ax1.plot(self.epochs, self.val_acc, 'g-o', label='验证准确率', markersize=4)
        self.lines.append(line2)
        
        # 绘制训练损失曲线 (右侧Y轴)
        line3, = self.ax2.plot(self.epochs, self.train_loss, 'r-o', label='训练损失', markersize=4)
        self.lines.append(line3)
        
        # 只在最新点上标注具体数值
        if len(self.epochs) > 0:
            # 训练准确率标注
            self.annotations.append(self.ax1.annotate(
                f'{train_accuracy:.2f}%',
                xy=(epoch, train_accuracy),
                xytext=(5, 5),
                textcoords='offset points',
                ha='left',
                va='bottom',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8)
            ))
            
            # 验证准确率标注
            self.annotations.append(self.ax1.annotate(
                f'{val_accuracy:.2f}%',
                xy=(epoch, val_accuracy),
                xytext=(5, 30),  # 垂直方向上错开显示
                textcoords='offset points',
                ha='left',
                va='bottom',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8)
            ))
            
            # 训练损失标注
            self.annotations.append(self.ax2.annotate(
                f'{train_loss:.4f}',
                xy=(epoch, train_loss),
                xytext=(5, 5),
                textcoords='offset points',
                ha='left',
                va='bottom',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8)
            ))
        
        # 设置Y轴范围
        self.ax1.set_ylim(0, 100)
        self.ax2.set_ylim(0, max(self.train_loss) * 1.1 if self.train_loss else 1)
        
        # 设置X轴范围
        if len(self.epochs) > 1:
            self.ax1.set_xlim(min(self.epochs) - 0.5, max(self.epochs) + 0.5)
        
        # 添加图例
        lines = self.ax1.get_lines() + self.ax2.get_lines()
        labels = [line.get_label() for line in lines]
        self.ax1.legend(lines, labels, loc='upper left')
        
        # 更新图表
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def save_figure(self, filename='training_visualization.png'):
        """保存当前图表到日志目录"""
        filepath = os.path.join(self.log_dir, filename)
        self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {filepath}")

    def close(self):
        """关闭图表"""
        plt.close(self.fig)

    def _ensure_log_dir_exists(self):
        """确保日志目录存在"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"创建日志目录: {self.log_dir}")
