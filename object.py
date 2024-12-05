import matplotlib.pyplot as plt
import os
import numpy as np  # 导入 numpy

# 绘制图表的函数
def plot_graph(x, y1, y2, xlabel, ylabel, title, label1, label2, file_name, file_path):
    plt.figure(figsize=(10, 6))

    # 将 y1 和 y2 转换为 numpy 数组，并限制它们的值在 0 到 10 之间
    y1 = np.array(y1).clip(0, 10)
    y2 = np.array(y2).clip(0, 10)

    # 绘制曲线
    plt.plot(x, y1, label=label1, marker='o', linestyle='-', color='b')
    plt.plot(x, y2, label=label2, marker='o', linestyle='-', color='r')

    # 设置标签和标题
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # 添加图例
    plt.legend()

    # 将 x 转换为 numpy 数组并确保它是整数类型
    x = np.array(x).astype(int)

    # 设置 x 轴刻度范围和标签
    plt.xticks(locations=x, labels=x, rotation=45)  # 使用locations和labels

    # 设置 y 轴范围
    plt.ylim(0, 10)  # y 轴范围限制在 0 到 10 之间
    
    # 保存图形到文件
    output_file = os.path.join(file_path, file_name)
    plt.savefig(output_file)
    plt.close()  # 关闭图形，释放内存

# 示例：绘制准确率图表
epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
training_accuracy = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.1, 1.2, 12.0]  # 假设有一些数据
validation_accuracy = [0.1, 0.15, 0.2, 0.3, 0.6, 1.0, 0.9, 1.0, 1.1, 8.0]

# 绘制准确率图表
file_path = "/path/to/save"  # 请替换为你实际的文件路径
plot_graph(epochs, training_accuracy, validation_accuracy, "Epoch", "Accuracy", 
           "Training and Validation Accuracy vs Epoch", 'Training_Accuracy', 'Validation Accuracy', 
           'accuracy_vs_epoch.png', file_path)

# 绘制损失图表
Training_Loss = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
Validation_Loss = [0.9, 0.85, 0.8, 0.7, 0.6, 0.4, 0.3, 0.25, 0.2, 0.15]
plot_graph(epochs, Training_Loss, Validation_Loss, "Epoch", "Loss", 
           "Training Loss and Validation Loss vs Epoch", 'Training_Loss', 'Validation_Loss', 
           'loss_vs_epoch.png', file_path)
