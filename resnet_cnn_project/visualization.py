import matplotlib.pyplot as plt

# 로그 파일에서 Loss 및 Accuracy 불러오기
def load_log_data(log_file):
    epochs = []
    train_loss = []
    top1_acc = []
    top5_acc = []
    superclass_acc = []

    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'Results' in line:
                parts = line.strip().split()
                epoch_num = int(parts[1].split('/')[0])
                epochs.append(epoch_num)
            if 'Validation Loss' in line:
                train_loss.append(float(line.split(":")[1].strip()))
            if 'Top-1 Accuracy' in line:
                top1_acc.append(float(line.split(":")[1].strip()))
            if 'Top-5 Accuracy' in line:
                top5_acc.append(float(line.split(":")[1].strip()))
            if 'Superclass Accuracy' in line:
                superclass_acc.append(float(line.split(":")[1].strip()))

    # 데이터의 길이가 맞지 않으면 조정
    min_length = min(len(epochs), len(train_loss), len(top1_acc), len(top5_acc), len(superclass_acc))
    
    return epochs[:min_length], train_loss[:min_length], top1_acc[:min_length], top5_acc[:min_length], superclass_acc[:min_length]


# Loss 시각화
def plot_loss(epochs, train_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_loss, label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Accuracy 시각화
def plot_accuracy(epochs, top1_acc, top5_acc, superclass_acc):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, top1_acc, label="Top 1 Accuracy")
    plt.plot(epochs, top5_acc, label="Top 5 Accuracy")
    plt.plot(epochs, superclass_acc, label="Superclass Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    log_file = "./resnet_efficientnet_train_output.log"  # 학습 로그 파일

    # 데이터 로드
    epochs, train_loss, top1_acc, top5_acc, superclass_acc = load_log_data(log_file)

    # 시각화
    plot_loss(epochs, train_loss)
    plot_accuracy(epochs, top1_acc, top5_acc, superclass_acc)
