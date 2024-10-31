## CIFAR-100 Image Classification Project
<img width="458" alt="Screenshot 2024-10-31 at 7 26 12 PM" src="https://github.com/user-attachments/assets/b022102b-62af-42ab-ad45-3c5136e37774">

### Project Overview
In this project, I explored various neural network architectures for image classification on the CIFAR-100 dataset, aiming to achieve high accuracy across diverse image classes. I experimented with several architectures, including DINO, ResNet (as the baseline model), ResNet+EfficientNet, and WideResNet+EfficientNet, to identify the most effective setup for this classification task.

### Dataset
- **Name:** CIFAR-100
- **Size:** 60,000 images, each 32x32 pixels
- **Classes:** 100 distinct classes

### Final Results
- **DINO:**
  - Top-1 Accuracy: 72.8%
  - Top-5 Accuracy: 91.5%
  - Superclass Accuracy: 80.4%

- **ResNet (Baseline Model):**
  - Top-1 Accuracy: 69.5%
  - Top-5 Accuracy: 89.8%
  - Superclass Accuracy: 77.2%

- **ResNet+EfficientNet:**
  - Top-1 Accuracy: 74.0%
  - Top-5 Accuracy: 92.3%
  - Superclass Accuracy: 83.0%

- **WideResNet+EfficientNet:**
  - Top-1 Accuracy: 76.3%
  - Top-5 Accuracy: 93.4%
  - Superclass Accuracy: 84.6%

### Project Schedule
1. **Weeks 2-4:** Researched and selected models.
2. **Weeks 4-6:** Implemented models and tested configurations.
3. **Weeks 6-8:** Trained models on CIFAR-100 dataset, monitored performance.
4. **Weeks 8-9:** Evaluated final model performance and prepared presentation.

### Conclusion
This exploration confirmed the strength of combining different architectures, such as WideResNet with EfficientNet, for achieving higher classification accuracy on complex datasets. Future work could involve model ensembling or further optimizing hyperparameters to improve results.

### References
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [ResNet Paper](https://arxiv.org/pdf/1505.00393)
- [EfficientNet Paper](https://arxiv.org/pdf/1905.11946)
- [WideResNet Paper](https://arxiv.org/pdf/1605.07146)
