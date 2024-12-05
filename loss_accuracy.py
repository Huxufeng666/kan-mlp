import csv
import matplotlib.pyplot as plt



# 保存结果到 CSV 文件
def save_results_to_csv(filename, results):
    keys = results[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


# 绘制损失和准确率曲线
def plot_results(train_results, test_results,save_path="cifar100_training_results.png"):
    train_epochs = [r["epoch"] for r in train_results]
    train_losses = [r["loss"] for r in train_results]
    train_accuracies = [r["accuracy"] for r in train_results]

    test_epochs = [r["epoch"] for r in test_results]
    test_losses = [r["loss"] for r in test_results]
    test_accuracies = [r["accuracy"] for r in test_results]

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_epochs, train_losses, label="Train Loss", color="blue")
    plt.plot(test_epochs, test_losses, label="Test Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_epochs, train_accuracies, label="Train Accuracy", color="blue")
    plt.plot(test_epochs, test_accuracies, label="Test Accuracy", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    # 保存图像到文件
    plt.tight_layout()  # 确保子图之间不重叠
    plt.savefig(save_path)  # 保存图像
    plt.close()  # 关闭图像，以释放内存

    # plt.tight_layout()
    # plt.show()
