## EfficientNet  
**EfficientNet model used**: We used efficientnet-b0, and loaded it with pre-trained weights for 100 CIFAR-100 classes.  
**Data loader**: Load the CIFAR-100 dataset as training and validation sets, and apply data augmentation techniques.  
**Training and validation**: Compute Top-1, Top-5, Superclass accuracy and loss during training.  
**Save results**: Periodically save logs and model checkpoints, and finally display the training results as graphs in the visualization script.  
