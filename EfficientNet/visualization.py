import matplotlib.pyplot as plt
import numpy as np

# Load the log data
data = np.load('./logs/train_results.npz')

# Plot Loss over Epochs
plt.figure(figsize=(10, 5))
plt.plot(data['train_loss'], label="Training Loss")
plt.plot(data['val_loss'], label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_over_epochs.png")
plt.show()

# Plot Accuracy over Epochs
plt.figure(figsize=(10, 5))
plt.plot(data['top1_acc'], label="Top-1 Accuracy")
plt.plot(data['top5_acc'], label="Top-5 Accuracy")
plt.plot(data['superclass_acc'], label="Superclass Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_over_epochs.png")
plt.show()
