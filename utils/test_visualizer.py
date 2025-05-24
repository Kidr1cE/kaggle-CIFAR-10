import visualizer

# 初始化可视化器
viz = visualizer.TrainingVisualizer(title="CIFAR10 模型训练")

# 模拟训练过程
epochs = 10
for epoch in range(1, epochs + 1):
    # 模拟训练结果
    train_accuracy = 70 + epoch * 2
    val_accuracy = 65 + epoch * 1.5
    train_loss = 2.0 - epoch * 0.15
    
    # 更新可视化
    viz.update(epoch, train_accuracy, val_accuracy, train_loss)

    # 可选：保存图表
    if epoch % 5 == 0:
        viz.save_figure(f'training_epoch_{epoch}.png')
    
    # 模拟训练时间
    import time
    time.sleep(1)

# 保存最终结果
viz.save_figure('final_training_result.png')

# 关闭图表
viz.close()
