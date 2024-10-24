import matplotlib.pyplot as plt
import re

def extract_data_from_log(log_file_path):
    epochs = []
    train_loss = []
    val_loss = []
    top1_acc = []
    top5_acc = []
    superclass_acc = []

    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            if "Epoch" in line and "Results" not in line:
                epoch = int(re.search(r"Epoch (\d+)/", line).group(1))
                epochs.append(epoch)
            elif "Training Loss" in line:
                train_loss.append(float(re.search(r"Training Loss: ([\d.]+)", line).group(1)))
            elif "Validation Loss" in line:
                val_loss.append(float(re.search(r"Validation Loss: ([\d.]+)", line).group(1)))
            elif "Top-1 Accuracy" in line:
                top1_acc.append(float(re.search(r"Top-1 Accuracy: ([\d.]+)", line).group(1)))
            elif "Top-5 Accuracy" in line:
                top5_acc.append(float(re.search(r"Top-5 Accuracy: ([\d.]+)", line).group(1)))
            elif "Superclass Accuracy" in line:
                superclass_acc.append(float(re.search(r"Superclass Accuracy: ([\d.]+)", line).group(1)))

    return epochs, train_loss, val_loss, top1_acc, top5_acc, superclass_acc

def plot_loss(epochs, train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    min_len = min(len(epochs), len(train_loss), len(val_loss))
    plt.plot(epochs[:min_len], train_loss[:min_len], label="Training Loss")
    plt.plot(epochs[:min_len], val_loss[:min_len], label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_over_epochs1.png")
    plt.close()


def plot_accuracy(epochs, top1_acc, top5_acc, superclass_acc):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, top1_acc[:len(epochs)], label="Top 1 Accuracy")
    plt.plot(epochs, top5_acc[:len(epochs)], label="Top 5 Accuracy")
    plt.plot(epochs, superclass_acc[:len(epochs)], label="Superclass Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_over_epochs1.png")
    plt.close()

if __name__ == "__main__":
    log_file_path = './logs/wideResNet_efficientNet_train_output1.log'  # 경로 수정

    # 로그 파일에서 데이터를 추출
    epochs, train_loss, val_loss, top1_acc, top5_acc, superclass_acc = extract_data_from_log(log_file_path)

    # 로그 데이터가 있으면 플롯 생성
    if train_loss and val_loss:
        plot_loss(epochs, train_loss, val_loss)
        plot_accuracy(epochs, top1_acc, top5_acc, superclass_acc)
    else:
        print("No data found in the log file.")
