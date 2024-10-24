import matplotlib.pyplot as plt
import re

# 로그 파일에서 데이터 추출 함수
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
            elif "Train Loss" in line:
                train_loss.append(float(re.search(r"Train Loss: ([\d.]+)", line).group(1)))
                top1_acc.append(float(re.search(r"Top-1 Accuracy: ([\d.]+)", line).group(1)))
                top5_acc.append(float(re.search(r"Top-5 Accuracy: ([\d.]+)", line).group(1)))
                superclass_acc.append(float(re.search(r"Superclass Accuracy: ([\d.]+)", line).group(1)))
            elif "Validation Loss" in line:
                val_loss.append(float(re.search(r"Validation Loss: ([\d.]+)", line).group(1)))

    return epochs, train_loss, val_loss, top1_acc, top5_acc, superclass_acc

# 로그 파일 경로 설정
log_file_path = './logs/train_output.log'

# 로그 데이터 추출
epochs, train_loss, val_loss, top1_acc, top5_acc, superclass_acc = extract_data_from_log(log_file_path)

# Loss 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label="Training Loss")
plt.plot(epochs, val_loss, label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_over_epochs.png")
plt.show()

# Accuracy 그래프 그리기
plt.figure(figsize=(10, 5))
plt.plot(epochs, top1_acc, label="Top-1 Accuracy")
plt.plot(epochs, top5_acc, label="Top-5 Accuracy")
plt.plot(epochs, superclass_acc, label="Superclass Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_over_epochs.png")
plt.show()
